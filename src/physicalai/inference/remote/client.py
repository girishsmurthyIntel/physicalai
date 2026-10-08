# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lazy client proxy for a remotely served inference model."""

from __future__ import annotations

import copy
import logging
import math
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Self

import numpy as np
import zenoh

from physicalai.config import export_config
from physicalai.inference.constants import ACTION
from physicalai.inference.model import InferenceModel
from physicalai.transport._zenoh import endpoint_for_key, make_zenoh_config, model_key_prefix, open_zenoh_session

from ._protocol import (
    DEFAULT_IMAGE_JPEG_QUALITY,
    PROTOCOL_VERSION,
    ImageCodec,
    RemoteInferenceError,
    RemoteInferenceModelMismatchError,
    RemoteInferenceProtocolError,
    RemoteInferenceTimeoutError,
    RemoteInferenceUnavailableError,
    RemoteTiming,
    decode_error,
    decode_payload,
    decode_response,
    encode_payload,
    encode_request,
)

logger = logging.getLogger(__name__)


@export_config(class_path="physicalai.inference.RemoteInferenceModel")
class RemoteInferenceModel(InferenceModel):
    """Proxy an inference model using a lazy, persistent Zenoh client session."""

    def __init__(
        self,
        name: str,
        *,
        endpoint: str | None = None,
        expected_policy: str | None = None,
        expected_manifest_sha256: str | None = None,
        request_timeout_s: float = 2.0,
        image_codec: ImageCodec = "jpeg",
        jpeg_quality: int = DEFAULT_IMAGE_JPEG_QUALITY,
        max_image_side: int | None = None,
        zenoh_config: str | Path | None = None,
    ) -> None:
        self._key_prefix = model_key_prefix(name)
        if endpoint is not None and (
            not isinstance(endpoint, str) or not endpoint or any(c.isspace() for c in endpoint)
        ):
            raise ValueError("endpoint must be a non-empty Zenoh endpoint")
        if (
            isinstance(request_timeout_s, bool)
            or not math.isfinite(request_timeout_s)
            or request_timeout_s <= 0
            or int(request_timeout_s * 1000) < 1
        ):
            raise ValueError("request_timeout_s must be finite and at least 0.001 seconds")
        if image_codec not in ("jpeg", "raw"):
            raise ValueError("image_codec must be 'jpeg' or 'raw'")
        if not isinstance(jpeg_quality, int) or isinstance(jpeg_quality, bool) or not 0 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 0 and 100")
        if max_image_side is not None and (
            not isinstance(max_image_side, int) or isinstance(max_image_side, bool) or max_image_side < 1
        ):
            raise ValueError("max_image_side must be positive")
        if expected_policy is not None and not isinstance(expected_policy, str):
            raise TypeError("expected_policy must be a string or None")
        if expected_manifest_sha256 is not None and not isinstance(expected_manifest_sha256, str):
            raise TypeError("expected_manifest_sha256 must be a string or None")

        self._name = name
        self.policy_name = expected_policy or name
        self._endpoint = endpoint or endpoint_for_key(self._key_prefix, "127.0.0.1")
        self._expected_policy = expected_policy
        self._expected_manifest_sha256 = expected_manifest_sha256
        self._request_timeout_s = float(request_timeout_s)
        self._image_codec = image_codec
        self._jpeg_quality = jpeg_quality
        self._max_image_side = max_image_side
        self._zenoh_config_path = Path(zenoh_config) if zenoh_config is not None else None
        self._builtin_config = None
        if self._zenoh_config_path is None:
            self._builtin_config = make_zenoh_config(
                "client",
                connect_endpoints=[self._endpoint],
                listen_endpoints=[],
                multicast_enabled=False,
                gossip_enabled=False,
            )

        self._lock = threading.RLock()
        self._session: zenoh.Session | None = None
        self._queriers: dict[str, Any] = {}
        self._seq = 0
        self._metadata: dict[str, Any] = {}
        self._server_id: str | None = None
        self._last_timing: RemoteTiming | None = None
        self._action_buffer: deque[np.ndarray] = deque()

    def connect(self) -> None:
        """Open the session, declare queriers, and validate the server handshake."""
        with self._lock:
            if self._session is not None and self._server_id is not None:
                return
            self._close_locked()
            try:
                if self._zenoh_config_path is None:
                    self._session = open_zenoh_session(
                        "client",
                        connect_endpoints=[self._endpoint],
                        listen_endpoints=[],
                        multicast_enabled=False,
                        gossip_enabled=False,
                        config=self._builtin_config,
                    )
                else:
                    custom_config = zenoh.Config.from_file(str(self._zenoh_config_path))
                    self._session = zenoh.open(custom_config)
                for key in (f"{self._key_prefix}/metadata", f"{self._key_prefix}/predict", f"{self._key_prefix}/reset"):
                    self._queriers[key.rsplit("/", maxsplit=1)[-1]] = self._session.declare_querier(
                        key,
                        consolidation=zenoh.ConsolidationMode.NONE,
                        timeout=self._request_timeout_s,
                    )
                deadline = time.monotonic() + self._request_timeout_s
                while not self._all_queries_match():
                    if time.monotonic() >= deadline:
                        raise RemoteInferenceUnavailableError(
                            f"Inference server {self._name!r} at {self._endpoint} is unavailable; is "
                            f"`physicalai inference serve --name {self._name}` running, and is the SSH tunnel up / "
                            "address reachable?"
                        )
                    time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
                request_seq = self._next_seq()
                request = encode_payload(
                    {
                        "protocol_version": PROTOCOL_VERSION,
                        "command": "metadata",
                        "seq": request_seq,
                        "budget_ms": int(self._request_timeout_s * 1000),
                    },
                    [],
                )
                started = time.monotonic()
                metadata = self._query_metadata(request, request_seq)
                self._validate_handshake(metadata)
                self._metadata = metadata
                self._server_id = metadata["server_id"]
                self.policy_name = str(metadata.get("policy_name") or self._name)
                self._last_timing = None
                elapsed = time.monotonic() - started
                rtc_supported = bool(metadata.get("rtc_supported"))
                rtc = metadata.get("rtc", {})
                chunk_size = (
                    rtc.get("chunk_size") if rtc_supported and isinstance(rtc, dict) else metadata.get("chunk_size")
                )
                logger.info(
                    "Connected to inference server %r: policy=%r, manifest_sha256=%r, chunk_size=%s "
                    "rtc_supported=%s (handshake %.3fs)",
                    self._name,
                    metadata.get("policy_name"),
                    metadata.get("manifest_sha256"),
                    chunk_size,
                    rtc_supported,
                    elapsed,
                )
            except RemoteInferenceError:
                self._close_locked()
                raise
            except Exception as error:
                self._close_locked()
                raise RemoteInferenceUnavailableError(
                    f"Inference server {self._name!r} at {self._endpoint} is unavailable; "
                    f"is `physicalai inference serve --name {self._name}` running, and is the SSH tunnel up / "
                    f"address reachable? ({error})"
                ) from error

    def __call__(self, inputs: dict[str, np.ndarray | list[str]]) -> dict[str, np.ndarray]:
        """Run the generic InferenceModel call interface remotely."""
        with self._lock:
            self.connect()
            metadata, outputs = self._predict(inputs, api="call")
            del metadata
            return outputs

    def predict_action_chunk(self, observation: dict[str, Any]) -> np.ndarray:
        """Return a two-dimensional action chunk, stripping a singleton batch."""
        with self._lock:
            self.connect()
            _, outputs = self._predict(observation, api="action_chunk")
            if ACTION not in outputs:
                raise RemoteInferenceProtocolError("Remote response has no action output")
            actions = outputs[ACTION]
            if actions.ndim == 3:
                if actions.shape[0] != 1:
                    raise RemoteInferenceProtocolError(
                        f"Batched action inference requires batch size 1, got {actions.shape[0]}"
                    )
                actions = actions[0]
            actions = np.atleast_2d(actions)
            if actions.ndim != 2:
                raise RemoteInferenceProtocolError(f"Expected a 2-D action chunk, got shape {actions.shape}")
            return actions

    def select_action(self, observation: dict[str, Any]) -> np.ndarray:
        """Return the next action from a locally buffered action chunk."""
        with self._lock:
            if not self._action_buffer:
                self._action_buffer.extend(self.predict_action_chunk(observation))
            return self._action_buffer.popleft()

    def reset(self) -> None:
        """Reset the remote policy on the model-owning server worker."""
        with self._lock:
            self.connect()
            seq = self._next_seq()
            request = encode_payload(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "command": "reset",
                    "seq": seq,
                    "budget_ms": int(self._request_timeout_s * 1000),
                },
                [],
            )
            self._send("reset", request, seq)
            self._action_buffer.clear()
            self._last_timing = None

    def close(self) -> None:
        """Close queriers and the persistent Zenoh session."""
        with self._lock:
            self._close_locked()

    def __enter__(self) -> Self:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"RemoteInferenceModel(name={self._name!r}, endpoint={self._endpoint!r}, "
            f"connected={self._server_id is not None})"
        )

    @property
    def endpoint(self) -> str:
        """The configured Zenoh endpoint."""
        return self._endpoint

    @property
    def chunk_size(self) -> int:
        """Action chunk size from the server metadata, connecting if required."""
        with self._lock:
            self.connect()
            rtc = self._metadata.get("rtc", {})
            if self._metadata.get("rtc_supported") and isinstance(rtc, dict) and "chunk_size" in rtc:
                return int(rtc["chunk_size"])
            return int(self._metadata["chunk_size"])

    @property
    def metadata(self) -> dict[str, Any]:
        """Copy of the last validated server handshake metadata."""
        with self._lock:
            return copy.deepcopy(self._metadata)

    @property
    def last_timing(self) -> RemoteTiming | None:
        """Timing information for the most recent completed request."""
        return self._last_timing

    @property
    def last_server_latency_s(self) -> float | None:
        """Compatibility view used by existing execution latency trackers."""
        return self._last_timing.compute_s if self._last_timing is not None else None

    def _all_queries_match(self) -> bool:
        for querier in self._queriers.values():
            status = querier.matching_status
            status = status() if callable(status) else status
            if not bool(getattr(status, "matching", False)):
                return False
        return bool(self._queriers)

    def _validate_handshake(self, metadata: dict[str, Any]) -> None:
        version = metadata.get("protocol_version")
        if not isinstance(version, int) or isinstance(version, bool) or version != PROTOCOL_VERSION:
            raise RemoteInferenceProtocolError(
                f"Remote protocol version {metadata.get('protocol_version')!r} does not match {PROTOCOL_VERSION}"
            )
        if metadata.get("name") != self._name:
            raise RemoteInferenceModelMismatchError(
                f"Server name {metadata.get('name')!r} does not match requested name {self._name!r}"
            )
        server_id = metadata.get("server_id")
        chunk_size = metadata.get("chunk_size")
        rtc = metadata.get("rtc", {})
        if not isinstance(server_id, str) or not server_id:
            raise RemoteInferenceProtocolError("Server metadata has no server_id")
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
            raise RemoteInferenceProtocolError("Server metadata has an invalid chunk_size")
        if not isinstance(rtc, dict) or not isinstance(metadata.get("rtc_supported"), bool):
            raise RemoteInferenceProtocolError("Server metadata has invalid RTC information")
        rtc_chunk_size = rtc.get("chunk_size")
        if (
            metadata["rtc_supported"]
            and rtc_chunk_size is not None
            and (not isinstance(rtc_chunk_size, int) or isinstance(rtc_chunk_size, bool) or rtc_chunk_size < 1)
        ):
            raise RemoteInferenceProtocolError("Server metadata has an invalid RTC chunk_size")
        if self._expected_policy is not None and metadata.get("policy_name") != self._expected_policy:
            raise RemoteInferenceModelMismatchError(
                f"Expected policy {self._expected_policy!r}, got {metadata.get('policy_name')!r}"
            )
        if (
            self._expected_manifest_sha256 is not None
            and metadata.get("manifest_sha256") != self._expected_manifest_sha256
        ):
            raise RemoteInferenceModelMismatchError("Remote manifest SHA-256 does not match the expected value")

    def _query_metadata(self, payload: bytes, seq: int) -> dict[str, Any]:
        reply = self._receive_matching_reply("metadata", payload, seq)
        if reply.err is not None:
            self._raise_remote_error(bytes(reply.err.payload), seq)
        if reply.ok is None:
            raise RemoteInferenceProtocolError("Metadata query returned an empty reply")
        metadata, frames = decode_payload(bytes(reply.ok.payload))
        if frames or metadata.get("seq") != seq or metadata.get("ok") is not True:
            raise RemoteInferenceProtocolError("Invalid metadata handshake response")
        return metadata

    def _predict(self, inputs: dict[str, Any], *, api: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        seq = self._next_seq()
        payload = encode_request(
            inputs,
            seq=seq,
            budget_ms=int(self._request_timeout_s * 1000),
            api=api,  # type: ignore[arg-type]
            image_codec=self._image_codec,
            jpeg_quality=self._jpeg_quality,
            max_image_side=self._max_image_side,
        )
        started = time.monotonic()
        reply = self._receive_matching_reply("predict", payload, seq)
        if reply.err is not None:
            self._raise_remote_error(bytes(reply.err.payload), seq)
        if reply.ok is None:
            raise RemoteInferenceProtocolError("Prediction query returned an empty reply")
        metadata, outputs = decode_response(bytes(reply.ok.payload))
        self._validate_response(metadata, seq)
        self._last_timing = RemoteTiming(
            seq=seq,
            round_trip_s=time.monotonic() - started,
            queue_s=float(metadata.get("queue_ms", 0)) / 1000,
            compute_s=float(metadata.get("compute_ms", 0)) / 1000,
        )
        return metadata, outputs

    def _receive_matching_reply(self, name: str, payload: bytes, seq: int) -> Any:
        if self._session is None:
            raise RemoteInferenceUnavailableError("Remote inference session is not connected")
        key_expr = f"{self._key_prefix}/{name}"
        receiver = self._session.get(
            key_expr,
            payload=payload,
            timeout=self._request_timeout_s,
            consolidation=zenoh.ConsolidationMode.NONE,
        )
        while True:
            try:
                reply = receiver.recv()
            except Exception as error:
                if "timeout" in str(error).lower():
                    raise RemoteInferenceTimeoutError(
                        f"Remote inference request {seq} timed out after {self._request_timeout_s:.3f}s"
                    ) from error
                raise RemoteInferenceUnavailableError(f"Remote inference query failed: {error}") from error
            if reply is None:
                raise RemoteInferenceTimeoutError(f"Remote inference request {seq} timed out")
            if reply.err is not None:
                error_payload = bytes(reply.err.payload)
                if error_payload.strip().lower() == b"timeout":
                    raise RemoteInferenceTimeoutError(
                        f"Remote inference request {seq} timed out after {self._request_timeout_s:.3f}s"
                    )
                error_metadata = decode_error(error_payload)
                if error_metadata.get("seq") != seq:
                    continue
                return reply
            if reply.ok is None:
                continue
            metadata, _ = decode_payload(bytes(reply.ok.payload))
            if metadata.get("seq") != seq:
                continue
            return reply

    def _validate_response(self, metadata: dict[str, Any], seq: int) -> None:
        version = metadata.get("protocol_version")
        response_seq = metadata.get("seq")
        if not isinstance(version, int) or isinstance(version, bool) or version != PROTOCOL_VERSION:
            raise RemoteInferenceProtocolError("Prediction response protocol version mismatch")
        if not isinstance(response_seq, int) or isinstance(response_seq, bool) or response_seq != seq:
            raise RemoteInferenceProtocolError("Prediction response sequence mismatch")
        for field in ("queue_ms", "compute_ms"):
            value = metadata.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise RemoteInferenceProtocolError(f"Prediction response has invalid {field}")
        server_id = metadata.get("server_id")
        if not isinstance(server_id, str) or server_id != self._server_id:
            raise RemoteInferenceModelMismatchError("Inference server restarted or was replaced")

    def _send(self, name: str, payload: bytes, seq: int) -> dict[str, Any]:
        reply = self._receive_matching_reply(name, payload, seq)
        if reply.err is not None:
            self._raise_remote_error(bytes(reply.err.payload), seq)
        if reply.ok is None:
            raise RemoteInferenceProtocolError("Remote request returned an empty reply")
        metadata, frames = decode_payload(bytes(reply.ok.payload))
        if (
            frames
            or metadata.get("protocol_version") != PROTOCOL_VERSION
            or metadata.get("seq") != seq
            or metadata.get("ok") is not True
        ):
            raise RemoteInferenceProtocolError("Invalid remote request response")
        self._validate_response(metadata, seq)
        return metadata

    def _raise_remote_error(self, payload: bytes, seq: int) -> None:
        metadata = decode_error(payload)
        if metadata.get("protocol_version") != PROTOCOL_VERSION:
            raise RemoteInferenceProtocolError("Remote error protocol version mismatch")
        if metadata.get("seq") != seq:
            raise RemoteInferenceProtocolError("Remote error sequence mismatch")
        code = metadata.get("code")
        if code == "expired":
            raise RemoteInferenceTimeoutError(f"Remote inference request {seq} expired in the server queue")
        if code == "unsupported_version":
            raise RemoteInferenceProtocolError("Remote server does not support this protocol version")
        if code == "invalid_request":
            raise RemoteInferenceProtocolError("Remote server rejected the request as invalid")
        if code == "shutting_down":
            raise RemoteInferenceUnavailableError("Remote inference server is shutting down")
        if code == "superseded":
            raise RemoteInferenceError(f"Remote inference request {seq} was superseded by a newer request")
        if code == "model_error":
            raise RemoteInferenceError("Remote model failed to process the request")
        raise RemoteInferenceProtocolError(f"Unknown remote error code: {code!r}")

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _close_locked(self) -> None:
        for querier in self._queriers.values():
            try:
                querier.undeclare()
            except Exception:
                pass
        self._queriers.clear()
        if self._session is not None:
            try:
                self._session.close()
            finally:
                self._session = None
        self._server_id = None
        self._metadata = {}
