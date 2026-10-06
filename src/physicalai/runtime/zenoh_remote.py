"""Zenoh remote inference model proxy and server for physicalai.

The client proxy implements the InferenceModel call interfaces. Scheduling is
delegated to the existing SyncExecution, AsyncExecution, or RTCExecution.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import cv2
import msgpack
import numpy as np
import zenoh

from physicalai._zenoh import endpoint_for_key, open_zenoh_session
from physicalai.config import export_config
from physicalai.inference.constants import ACTION
from physicalai.inference.model import InferenceModel

logger = logging.getLogger(__name__)

_PROTOCOL_VERSION = 2
_DEFAULT_IMAGE_JPEG_QUALITY = 90
_MODEL_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z", re.ASCII)


class ZenohRemoteInferenceError(RuntimeError):
    """Raised when a Zenoh remote inference request fails."""


def _encode_payload(metadata: dict[str, Any], frames: list[bytes]) -> bytes:
    return msgpack.packb({"metadata": metadata, "frames": frames}, use_bin_type=True)


def _decode_payload(payload: bytes) -> tuple[dict[str, Any], list[bytes]]:
    try:
        unpacked = msgpack.unpackb(payload, raw=False, strict_map_key=False)
        if not isinstance(unpacked, dict):
            raise ZenohRemoteInferenceError("Payload must be a MessagePack map")
        metadata = unpacked.get("metadata")
        frames = unpacked.get("frames")
        if not isinstance(metadata, dict) or not isinstance(frames, list):
            raise ZenohRemoteInferenceError("Payload must contain metadata and frames")
        if not all(isinstance(frame, bytes) for frame in frames):
            raise ZenohRemoteInferenceError("Payload frames must be binary data")
        return metadata, frames
    except ZenohRemoteInferenceError:
        raise
    except (msgpack.UnpackException, TypeError, ValueError, OverflowError) as error:
        raise ZenohRemoteInferenceError("Invalid MessagePack inference payload") from error


def _array_descriptor(name: str, array: np.ndarray, encoding: str = "raw") -> dict[str, Any]:
    return {
        "name": name,
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "encoding": encoding,
    }


def _is_image_input(name: str) -> bool:
    return name == "images" or name.startswith("images.")


def _encode_image_jpeg(image: np.ndarray, quality: int) -> bytes:
    if image.dtype != np.uint8:
        raise ZenohRemoteInferenceError("Image observations must use uint8 pixels for JPEG encoding")
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    elif image.ndim == 3 and image.shape[0] == 1 and image.shape[-1] not in (1, 3):
        image = image[0]
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[-1] != 3):
        raise ZenohRemoteInferenceError(f"Unsupported image shape for JPEG encoding: {image.shape}")
    # Camera inputs are RGB; OpenCV's JPEG encoder expects BGR.
    encoder_input = image[..., ::-1] if image.ndim == 3 else image
    success, encoded = cv2.imencode(".jpg", encoder_input, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ZenohRemoteInferenceError("OpenCV failed to encode image observation as JPEG")
    return encoded.tobytes()


def _decode_image_jpeg(frame: bytes, dtype: np.dtype[Any], shape: tuple[int, ...]) -> np.ndarray:
    encoded = np.frombuffer(frame, dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ZenohRemoteInferenceError("OpenCV failed to decode JPEG image observation")
    if decoded.ndim == 3 and decoded.shape[-1] == 3:
        decoded = decoded[..., ::-1]
    if len(shape) == 4 and shape[0] == 1:
        decoded = decoded[np.newaxis]
    elif len(shape) == 3 and shape[0] == 1 and decoded.ndim == 2:
        decoded = decoded[np.newaxis]
    if len(shape) == 3 and shape[-1] == 1 and decoded.ndim == 2:
        decoded = decoded[..., np.newaxis]
    if decoded.shape != shape or decoded.dtype != dtype:
        raise ZenohRemoteInferenceError("Decoded JPEG image does not match its declared shape and dtype")
    return np.ascontiguousarray(decoded)


def _encode_observation(
    observation: dict[str, Any],
    image_jpeg_quality: int = _DEFAULT_IMAGE_JPEG_QUALITY,
    command: str = "predict",
) -> bytes:
    if not 0 <= image_jpeg_quality <= 100:
        raise ValueError("image_jpeg_quality must be between 0 and 100")
    arrays: list[dict[str, Any]] = []
    frames: list[bytes] = []
    for name, value in observation.items():
        array = np.ascontiguousarray(np.asarray(value))
        if array.dtype.hasobject:
            raise ZenohRemoteInferenceError(f"Observation {name!r} has unsupported object dtype")
        if _is_image_input(name):
            frame = _encode_image_jpeg(array, image_jpeg_quality)
            encoding = "jpeg"
        elif array.dtype.kind == "U":
            frame = array.tobytes()
            encoding = "text"
        else:
            frame = array.tobytes()
            encoding = "raw"
        arrays.append(_array_descriptor(name, array, encoding))
        frames.append(frame)
    return _encode_payload({"version": _PROTOCOL_VERSION, "command": command, "arrays": arrays}, frames)


def _decode_observation_parts(
    metadata: dict[str, Any], frames: list[bytes], command: str = "predict"
) -> dict[str, Any]:
    if metadata.get("version") != _PROTOCOL_VERSION or metadata.get("command") != command:
        raise ZenohRemoteInferenceError("Unsupported inference request")
    descriptors = metadata.get("arrays")
    if not isinstance(descriptors, list) or len(descriptors) != len(frames):
        raise ZenohRemoteInferenceError("Observation metadata does not match data frames")
    observation: dict[str, Any] = {}
    for descriptor, frame in zip(descriptors, frames, strict=True):
        name = descriptor.get("name")
        dtype = np.dtype(descriptor.get("dtype"))
        shape = tuple(descriptor.get("shape", ()))
        if not isinstance(name, str) or dtype.hasobject or any(not isinstance(size, int) or size < 0 for size in shape):
            raise ZenohRemoteInferenceError("Invalid observation array descriptor")
        encoding = descriptor.get("encoding", "raw")
        if encoding == "jpeg":
            if not _is_image_input(name):
                raise ZenohRemoteInferenceError("JPEG encoding is only allowed for image inputs")
            observation[name] = _decode_image_jpeg(frame, dtype, shape)
        elif encoding == "text":
            expected_size = int(np.prod(shape, dtype=np.intp)) * dtype.itemsize
            if dtype.kind != "U" or len(frame) != expected_size:
                raise ZenohRemoteInferenceError(f"Observation {name!r} has invalid text data")
            observation[name] = np.frombuffer(frame, dtype=dtype).reshape(shape).tolist()
        elif encoding == "raw":
            expected_size = int(np.prod(shape, dtype=np.intp)) * dtype.itemsize
            if len(frame) != expected_size:
                raise ZenohRemoteInferenceError(f"Observation {name!r} has invalid data length")
            observation[name] = np.frombuffer(frame, dtype=dtype).reshape(shape).copy()
        else:
            raise ZenohRemoteInferenceError(f"Unsupported observation encoding: {encoding!r}")
    return observation


def _encode_model_outputs(outputs: dict[str, Any], server_latency_s: float) -> bytes:
    arrays: list[dict[str, Any]] = []
    frames: list[bytes] = []
    for name, value in outputs.items():
        array = np.ascontiguousarray(np.asarray(value))
        if array.dtype.hasobject:
            raise ZenohRemoteInferenceError(f"Model output {name!r} has unsupported object dtype")
        arrays.append(_array_descriptor(name, array))
        frames.append(array.tobytes())
    return _encode_payload(
        {
            "version": _PROTOCOL_VERSION,
            "ok": True,
            "server_inference_latency_s": server_latency_s,
            "arrays": arrays,
        },
        frames,
    )


def _decode_model_outputs(payload: bytes) -> tuple[dict[str, np.ndarray], float | None]:
    try:
        metadata, frames = _decode_payload(payload)
        if not metadata.get("ok"):
            raise ZenohRemoteInferenceError(str(metadata.get("error", "Remote inference failed")))
        if metadata.get("version") != _PROTOCOL_VERSION:
            raise ZenohRemoteInferenceError("Unsupported model output protocol version")
        descriptors = metadata.get("arrays")
        if not isinstance(descriptors, list) or len(descriptors) != len(frames):
            raise ZenohRemoteInferenceError("Model output metadata does not match its data frames")
        outputs: dict[str, np.ndarray] = {}
        for descriptor, frame in zip(descriptors, frames, strict=True):
            name = descriptor.get("name")
            dtype = np.dtype(descriptor.get("dtype"))
            shape = tuple(descriptor.get("shape", ()))
            if (
                not isinstance(name, str)
                or dtype.hasobject
                or any(not isinstance(size, int) or size < 0 for size in shape)
            ):
                raise ZenohRemoteInferenceError("Invalid model output descriptor")
            if descriptor.get("encoding") != "raw":
                raise ZenohRemoteInferenceError("Model outputs must use raw encoding")
            expected_size = int(np.prod(shape, dtype=np.intp)) * dtype.itemsize
            if len(frame) != expected_size:
                raise ZenohRemoteInferenceError(f"Model output {name!r} has invalid data length")
            outputs[name] = np.frombuffer(frame, dtype=dtype).reshape(shape).copy()
        latency = metadata.get("server_inference_latency_s")
        if latency is not None and (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not np.isfinite(latency)
            or latency < 0
        ):
            raise ZenohRemoteInferenceError("Invalid server inference latency")
        return outputs, latency
    except ZenohRemoteInferenceError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as error:
        raise ZenohRemoteInferenceError("Invalid remote model output") from error


def _decode_observation(payload: bytes) -> dict[str, Any]:
    """Decode an observation payload for callers that have not parsed it yet."""
    metadata, frames = _decode_payload(payload)
    return _decode_observation_parts(metadata, frames, command="predict")


def _encode_actions(actions: np.ndarray, server_latency_s: float | None = None) -> bytes:
    array = np.ascontiguousarray(np.asarray(actions))
    if array.dtype.hasobject:
        raise ZenohRemoteInferenceError("Actions have unsupported object dtype")
    metadata = {
        "version": _PROTOCOL_VERSION,
        "ok": True,
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "server_inference_latency_s": server_latency_s,
        "arrays": [_array_descriptor("actions", array)],
    }
    return _encode_payload(metadata, [array.tobytes()])


def _decode_actions(payload: bytes) -> tuple[np.ndarray, float | None]:
    try:
        metadata, frames = _decode_payload(payload)
        if not metadata.get("ok"):
            raise ZenohRemoteInferenceError(str(metadata.get("error", "Remote inference failed")))
        if len(frames) != 1:
            raise ZenohRemoteInferenceError("Action response must contain one data frame")
        dtype = np.dtype(metadata["dtype"])
        shape = tuple(metadata["shape"])
        if (
            len(shape) < 2
            or shape[0] == 0
            or dtype.hasobject
            or any(not isinstance(size, int) or size < 0 for size in shape)
        ):
            raise ZenohRemoteInferenceError("Invalid action array descriptor")
        expected_size = int(np.prod(shape, dtype=np.intp)) * dtype.itemsize
        _DEFAULT_IMAGE_JPEG_QUALITY = 90

        _MODEL_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z", re.ASCII)
        if len(frames[0]) != expected_size:
            raise ZenohRemoteInferenceError("Action response has invalid data length")
        latency = metadata.get("server_inference_latency_s")
        if latency is not None and (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not np.isfinite(latency)
            or latency < 0
        ):
            raise ZenohRemoteInferenceError("Invalid server inference latency")
        return np.frombuffer(frames[0], dtype=dtype).reshape(shape).copy(), latency
    except ZenohRemoteInferenceError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as error:
        raise ZenohRemoteInferenceError("Invalid action response") from error


def _model_key_prefix(model_name: str) -> str:
    if not _MODEL_NAME_RE.fullmatch(model_name) or model_name in {".", ".."}:
        raise ValueError("model_name must be a non-empty key segment containing only letters, digits, '.', '_' or '-'")
    return f"physicalai/inference/{model_name}"


def _payload_bytes(payload: Any) -> bytes:
    return bytes(payload)


class RemoteInferenceServer:
    """Serve an inference model through a Zenoh queryable."""

    def __init__(
        self,
        model: InferenceModel,
        model_name: str | None = None,
        *,
        listen_endpoint: str | None = None,
        listen_host: str = "0.0.0.0",
        listen_port: int | None = None,
        config: zenoh.Config | None = None,
        max_workers: int = 1,
    ) -> None:
        served_model_name = model_name or getattr(model, "policy_name", None)
        if served_model_name is None:
            raise ValueError("model_name is required when the model does not expose policy_name")
        loaded_model_name = getattr(model, "policy_name", None)
        if loaded_model_name is not None and loaded_model_name != served_model_name:
            raise ValueError(f"model_name {served_model_name!r} does not match loaded model {loaded_model_name!r}")
        self._key_prefix = _model_key_prefix(served_model_name)
        self._model_name = served_model_name
        self._model = model
        self._handshake_key = f"{self._key_prefix}/handshake"
        self._predict_key = f"{self._key_prefix}/predict"
        self._call_key = f"{self._key_prefix}/call"
        self._reset_key = f"{self._key_prefix}/reset"
        self._listen_endpoint = listen_endpoint or endpoint_for_key(self._key_prefix, listen_host, listen_port)
        self._config = config
        self._max_workers = max_workers
        self._session: zenoh.Session | None = None
        self._queryables: list[Any] = []
        self._executor: ThreadPoolExecutor | None = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()

    def serve_forever(self) -> None:
        self._session = open_zenoh_session(
            mode="router",
            connect_endpoints=[],
            listen_endpoints=[self._listen_endpoint],
            multicast_enabled=False,
            gossip_enabled=False,
            config=self._config,
        )
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers, thread_name_prefix="ZenohInference")
        self._queryables = [
            self._session.declare_queryable(
                key,
                lambda query, reply_key=key, expected_command=command: self._on_query(
                    query, reply_key, expected_command
                ),
                complete=True,
            )
            for key, command in (
                (self._handshake_key, "handshake"),
                (self._predict_key, "predict"),
                (self._call_key, "call"),
                (self._reset_key, "reset"),
            )
        ]
        self._ready_event.set()
        try:
            self._stop_event.wait()
        finally:
            self.stop()

    def wait_until_ready(self, timeout_s: float | None = None) -> bool:
        """Wait until the router session and all model queryables are ready."""
        return self._ready_event.wait(timeout_s)

    @property
    def listen_endpoint(self) -> str:
        """Return the actual endpoint bound by this server."""
        return self._listen_endpoint

    def stop(self) -> None:
        self._stop_event.set()
        self._ready_event.clear()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        for queryable in self._queryables:
            import contextlib

            with contextlib.suppress(Exception):
                queryable.undeclare()
        self._queryables.clear()
        if self._session is not None:
            self._session.close()
            self._session = None

    def _on_query(self, query: zenoh.Query, reply_key: str, expected_command: str) -> None:
        if self._executor is None:
            query.reply_err(b"Server is not running")
            return
        self._executor.submit(self._handle_query, query, reply_key, expected_command)

    def _handle_query(self, query: zenoh.Query, reply_key: str, expected_command: str) -> None:
        try:
            payload = _payload_bytes(query.payload) if query.payload is not None else b""
            metadata, frames = _decode_payload(payload)
            command = metadata.get("command")
            if metadata.get("version") != _PROTOCOL_VERSION or command != expected_command:
                raise ZenohRemoteInferenceError("Unsupported request protocol or command")
            if command == "handshake":
                if frames or metadata.get("model_name") != self._model_name:
                    raise ZenohRemoteInferenceError("Handshake model name does not match this server")
                model_extra = getattr(getattr(self._model, "manifest", None), "model_extra", {})
                rtc_config = model_extra.get("rtc", {}) if isinstance(model_extra, dict) else {}
                if not isinstance(rtc_config, dict):
                    rtc_config = {}
                chunk_size = getattr(self._model, "chunk_size", 1)
                if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
                    chunk_size = 1
                response = _encode_payload(
                    {
                        "version": _PROTOCOL_VERSION,
                        "ok": True,
                        "command": "handshake",
                        "model_name": self._model_name,
                        "chunk_size": chunk_size,
                        "rtc_config": rtc_config,
                    },
                    [],
                )
            elif command == "predict":
                observation = _decode_observation_parts(metadata, frames)
                started_at = time.perf_counter()
                actions = self._model.predict_action_chunk(observation)
                latency = time.perf_counter() - started_at
                response = _encode_actions(actions, latency)
            elif command == "call":
                inputs = _decode_observation_parts(metadata, frames, command="call")
                started_at = time.perf_counter()
                outputs = self._model(inputs)
                latency = time.perf_counter() - started_at
                if not isinstance(outputs, dict):
                    raise ZenohRemoteInferenceError("Model __call__ must return a mapping of output arrays")
                response = _encode_model_outputs(outputs, latency)
            else:
                self._model.reset()
                response = _encode_payload({"version": _PROTOCOL_VERSION, "ok": True, "command": "reset"}, [])
            query.reply(reply_key, response)
        except Exception as error:
            logger.exception("Zenoh remote inference request failed")
            query.reply_err(str(error).encode("utf-8"))


@export_config(class_path="physicalai.runtime.RemoteInferenceModel")
class RemoteInferenceModel(InferenceModel):
    """InferenceModel-compatible client proxy for a policy served over Zenoh.

    It implements both ``predict_action_chunk`` and ``__call__`` so the existing
    SyncExecution, AsyncExecution, and RTCExecution can use it unchanged.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        model_name: str | None = None,
        *,
        server_host: str = "127.0.0.1",
        server_port: int | None = None,
        image_jpeg_quality: int = _DEFAULT_IMAGE_JPEG_QUALITY,
        request_timeout_s: float = 30.0,
        config_path: str | None = None,
    ) -> None:
        if model_name is None:
            raise ValueError("model_name is required")
        if not 0 <= image_jpeg_quality <= 100:
            raise ValueError("image_jpeg_quality must be between 0 and 100")
        self._key_prefix = _model_key_prefix(model_name)
        self._model_name = model_name
        self._handshake_key = f"{self._key_prefix}/handshake"
        self._predict_key = f"{self._key_prefix}/predict"
        self._call_key = f"{self._key_prefix}/call"
        self._reset_key = f"{self._key_prefix}/reset"
        self._endpoint = endpoint or endpoint_for_key(self._key_prefix, server_host, server_port)
        self._image_jpeg_quality = image_jpeg_quality
        self._request_timeout_s = request_timeout_s
        self._config = zenoh.Config.from_file(config_path) if config_path else None
        self._request_lock = threading.RLock()
        self._session: zenoh.Session | None = None
        self._last_server_latency_s: float | None = None
        self._chunk_size = 1
        self._rtc_config: dict[str, Any] = {}
        self.policy_name = model_name
        self._manifest = SimpleNamespace(model_extra={"rtc": self._rtc_config})
        self._action_buffer: deque[np.ndarray] = deque()
        self._connect_and_handshake()

    def __call__(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Run remote inference with the generic InferenceModel call interface."""
        with self._request_lock:
            outputs, latency = _decode_model_outputs(self._request("call", inputs))
            if ACTION not in outputs:
                raise ZenohRemoteInferenceError("Remote model response has no action output")
            self._last_server_latency_s = latency
            return outputs

    def predict_action_chunk(self, observation: dict[str, Any]) -> np.ndarray:
        """Request one action chunk from the configured remote model."""
        with self._request_lock:
            response = self._request("predict", observation)
            actions, latency = _decode_actions(response)
            if actions.shape[0] == 0:
                raise ZenohRemoteInferenceError("Remote server returned an empty action chunk")
            self._last_server_latency_s = latency
            return actions

    def select_action(self, observation: dict[str, Any]) -> np.ndarray:
        """Return the first action from a remote action chunk."""
        if not self._action_buffer:
            self._action_buffer.extend(self.predict_action_chunk(observation))
        return self._action_buffer.popleft()

    def reset(self) -> None:
        """Reset policy state on the remote inference server."""
        with self._request_lock:
            self._request("reset", None)
            self._last_server_latency_s = None
            self._action_buffer.clear()

    def close(self) -> None:
        """Close the persistent Zenoh client session."""
        with self._request_lock:
            if self._session is not None:
                self._session.close()
                self._session = None

    @property
    def endpoint(self) -> str:
        """Return the single endpoint this model proxy connects to."""
        return self._endpoint

    @property
    def chunk_size(self) -> int:
        """Action chunk size advertised by the server during handshake."""
        return self._chunk_size

    @property
    def manifest(self) -> SimpleNamespace:
        """Minimal manifest metadata needed by RTCExecution."""
        return self._manifest

    @property
    def last_server_latency_s(self) -> float | None:
        """Server-side model compute time from the most recent prediction."""
        return self._last_server_latency_s

    def _connect_and_handshake(self) -> None:
        try:
            self._session = open_zenoh_session(
                mode="client",
                connect_endpoints=[self._endpoint],
                listen_endpoints=[],
                multicast_enabled=False,
                gossip_enabled=False,
                config=self._config,
            )
            payload = _encode_payload(
                {"version": _PROTOCOL_VERSION, "command": "handshake", "model_name": self._model_name},
                [],
            )
            reply = self._session.get(
                self._handshake_key,
                payload=payload,
                timeout=self._request_timeout_s,
            ).recv()
            if reply.err is not None:
                detail = _payload_bytes(reply.err.payload).decode("utf-8", errors="replace")
                raise ZenohRemoteInferenceError(f"Zenoh server handshake failed: {detail}")
            if reply.ok is None:
                raise ZenohRemoteInferenceError("Zenoh server returned an empty handshake response")
            metadata, frames = _decode_payload(_payload_bytes(reply.ok.payload))
            if (
                frames
                or metadata.get("version") != _PROTOCOL_VERSION
                or metadata.get("command") != "handshake"
                or metadata.get("ok") is not True
                or metadata.get("model_name") != self._model_name
            ):
                raise ZenohRemoteInferenceError("Zenoh server handshake protocol or model mismatch")
            chunk_size = metadata.get("chunk_size", 1)
            rtc_config = metadata.get("rtc_config", {})
            if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
                raise ZenohRemoteInferenceError("Zenoh server advertised an invalid chunk size")
            if not isinstance(rtc_config, dict):
                raise ZenohRemoteInferenceError("Zenoh server advertised invalid RTC metadata")
            self._chunk_size = chunk_size
            self._rtc_config = rtc_config
            self._manifest = SimpleNamespace(model_extra={"rtc": rtc_config})
        except Exception as error:
            if self._session is not None:
                self._session.close()
                self._session = None
            if isinstance(error, ZenohRemoteInferenceError):
                raise
            raise ZenohRemoteInferenceError(
                f"Could not verify Zenoh server model {self._model_name!r} at {self._endpoint}: {error}"
            ) from error

    def _request(self, command: str, observation: dict[str, Any] | None) -> bytes:
        if self._session is None:
            self._connect_and_handshake()
        if command == "predict":
            if observation is None:
                raise ValueError("An observation is required for prediction")
            payload = _encode_observation(observation, self._image_jpeg_quality)
            key_expr = self._predict_key
        elif command == "call":
            if observation is None:
                raise ValueError("Model inputs are required for remote __call__")
            payload = _encode_observation(observation, self._image_jpeg_quality, command="call")
            key_expr = self._call_key
        else:
            payload = _encode_payload({"version": _PROTOCOL_VERSION, "command": command}, [])
            key_expr = self._reset_key
        try:
            replies = self._session.get(key_expr, payload=payload, timeout=self._request_timeout_s)
            reply = replies.recv()
            if reply.err is not None:
                raise ZenohRemoteInferenceError(_payload_bytes(reply.err.payload).decode("utf-8", errors="replace"))
            if reply.ok is None:
                raise ZenohRemoteInferenceError("Zenoh returned an empty reply")
            return _payload_bytes(reply.ok.payload)
        except ZenohRemoteInferenceError:
            raise
        except Exception as error:
            raise ZenohRemoteInferenceError(f"Zenoh request to {key_expr} failed: {error}") from error
