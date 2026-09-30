"""Zenoh query/reply remote execution for physicalai.

This module is intentionally separate from the installed ZeroMQ implementation.
It preserves the Execution lifecycle while using Zenoh Session.get() on the
client and Session.declare_queryable() on the server.
"""

from __future__ import annotations

import json
import logging
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import zenoh

from physicalai.config import export_config
from physicalai.runtime.execution.base import NOT_STARTED, Execution, WorkerDiedError

if TYPE_CHECKING:
    from physicalai.inference.model import InferenceModel
    from physicalai.runtime._callback_bus import _CallbackBus
    from physicalai.runtime.execution.queue import ActionQueue, ChunkedActionQueue

logger = logging.getLogger(__name__)

_PROTOCOL_VERSION = 1
_HEADER_SIZE = struct.Struct("!I")
_JOIN_TIMEOUT_S = 10.0


class ZenohRemoteInferenceError(RuntimeError):
    """Raised when a Zenoh remote inference request fails."""


def _encode_payload(metadata: dict[str, Any], frames: list[bytes]) -> bytes:
    header = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    return _HEADER_SIZE.pack(len(header)) + header + b"".join(frames)


def _decode_payload(payload: bytes) -> tuple[dict[str, Any], list[bytes]]:
    try:
        if len(payload) < _HEADER_SIZE.size:
            raise ZenohRemoteInferenceError("Payload has no metadata header")
        header_size = _HEADER_SIZE.unpack_from(payload)[0]
        header_end = _HEADER_SIZE.size + header_size
        metadata = json.loads(payload[_HEADER_SIZE.size:header_end])
        if not isinstance(metadata, dict):
            raise ZenohRemoteInferenceError("Payload metadata must be an object")
        data = payload[header_end:]
        frames: list[bytes] = []
        for descriptor in metadata.get("arrays", []):
            offset = int(descriptor["offset"])
            size = int(descriptor["nbytes"])
            if offset < 0 or size < 0 or offset + size > len(data):
                raise ZenohRemoteInferenceError("Invalid array frame bounds")
            frames.append(data[offset:offset + size])
        if metadata.get("arrays") and sum(int(item["nbytes"]) for item in metadata["arrays"]) != len(data):
            raise ZenohRemoteInferenceError("Array metadata does not match payload size")
        return metadata, frames
    except ZenohRemoteInferenceError:
        raise
    except (IndexError, KeyError, TypeError, ValueError, OverflowError, json.JSONDecodeError) as error:
        raise ZenohRemoteInferenceError("Invalid Zenoh inference payload") from error


def _array_descriptor(name: str, array: np.ndarray, offset: int) -> dict[str, Any]:
    return {
        "name": name,
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "offset": offset,
        "nbytes": array.nbytes,
    }


def _encode_observation(observation: dict[str, Any]) -> bytes:
    arrays: list[dict[str, Any]] = []
    frames: list[bytes] = []
    offset = 0
    for name, value in observation.items():
        array = np.ascontiguousarray(np.asarray(value))
        if array.dtype.hasobject:
            raise ZenohRemoteInferenceError(f"Observation {name!r} has unsupported object dtype")
        frame = array.tobytes()
        arrays.append(_array_descriptor(name, array, offset))
        frames.append(frame)
        offset += len(frame)
    return _encode_payload({"version": _PROTOCOL_VERSION, "command": "predict", "arrays": arrays}, frames)


def _decode_observation_parts(metadata: dict[str, Any], frames: list[bytes]) -> dict[str, np.ndarray]:
    if metadata.get("version") != _PROTOCOL_VERSION or metadata.get("command") != "predict":
        raise ZenohRemoteInferenceError("Unsupported inference request")
    descriptors = metadata.get("arrays")
    if not isinstance(descriptors, list) or len(descriptors) != len(frames):
        raise ZenohRemoteInferenceError("Observation metadata does not match data frames")
    observation: dict[str, np.ndarray] = {}
    for descriptor, frame in zip(descriptors, frames, strict=True):
        name = descriptor.get("name")
        dtype = np.dtype(descriptor.get("dtype"))
        shape = tuple(descriptor.get("shape", ()))
        if not isinstance(name, str) or dtype.hasobject or any(not isinstance(size, int) or size < 0 for size in shape):
            raise ZenohRemoteInferenceError("Invalid observation array descriptor")
        expected_size = int(np.prod(shape, dtype=np.intp)) * dtype.itemsize
        if len(frame) != expected_size:
            raise ZenohRemoteInferenceError(f"Observation {name!r} has invalid data length")
        observation[name] = np.frombuffer(frame, dtype=dtype).reshape(shape).copy()
    return observation


def _decode_observation(payload: bytes) -> dict[str, np.ndarray]:
    """Decode an observation payload for callers that have not parsed it yet."""
    metadata, frames = _decode_payload(payload)
    return _decode_observation_parts(metadata, frames)


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
        "arrays": [_array_descriptor("actions", array, 0)],
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
        if len(shape) < 2 or shape[0] == 0 or dtype.hasobject or any(
            not isinstance(size, int) or size < 0 for size in shape
        ):
            raise ZenohRemoteInferenceError("Invalid action array descriptor")
        expected_size = int(np.prod(shape, dtype=np.intp)) * dtype.itemsize
        if len(frames[0]) != expected_size:
            raise ZenohRemoteInferenceError("Action response has invalid data length")
        latency = metadata.get("server_inference_latency_s")
        if latency is not None and (
            isinstance(latency, bool) or not isinstance(latency, (int, float))
            or not np.isfinite(latency) or latency < 0
        ):
            raise ZenohRemoteInferenceError("Invalid server inference latency")
        return np.frombuffer(frames[0], dtype=dtype).reshape(shape).copy(), latency
    except ZenohRemoteInferenceError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError, json.JSONDecodeError) as error:
        raise ZenohRemoteInferenceError("Invalid action response") from error


def _make_config(connect_endpoint: str | None = None, listen_endpoint: str | None = None) -> zenoh.Config:
    config = zenoh.Config()
    if connect_endpoint:
        config.insert_json5("connect/endpoints", json.dumps([connect_endpoint]))
    if listen_endpoint:
        config.insert_json5("listen/endpoints", json.dumps([listen_endpoint]))
    return config


def _payload_bytes(payload: Any) -> bytes:
    return bytes(payload)


class ZenohRemoteInferenceServer:
    """Serve an inference model through a Zenoh queryable."""

    def __init__(
        self,
        model: InferenceModel,
        key_expr: str = "physicalai/inference/predict",
        *,
        listen_endpoint: str | None = None,
        config: zenoh.Config | None = None,
        max_workers: int = 1,
    ) -> None:
        self._model = model
        self._key_expr = key_expr
        self._config = config or _make_config(listen_endpoint=listen_endpoint)
        self._max_workers = max_workers
        self._session: zenoh.Session | None = None
        self._queryable: Any = None
        self._executor: ThreadPoolExecutor | None = None
        self._stop_event = threading.Event()

    def serve_forever(self) -> None:
        self._session = zenoh.open(self._config)
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers, thread_name_prefix="ZenohInference")
        self._queryable = self._session.declare_queryable(self._key_expr, self._on_query, complete=True)
        try:
            self._stop_event.wait()
        finally:
            self.stop()

    def stop(self) -> None:
        self._stop_event.set()
        if self._queryable is not None:
            self._queryable.undeclare()
            self._queryable = None
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        if self._session is not None:
            self._session.close()
            self._session = None

    def _on_query(self, query: zenoh.Query) -> None:
        if self._executor is None:
            query.reply_err(b"Server is not running")
            return
        self._executor.submit(self._handle_query, query)

    def _handle_query(self, query: zenoh.Query) -> None:
        try:
            payload = _payload_bytes(query.payload) if query.payload is not None else b""
            metadata, frames = _decode_payload(payload)
            command = metadata.get("command")
            if command == "predict":
                observation = _decode_observation_parts(metadata, frames)
                started_at = time.perf_counter()
                actions = self._model.predict_action_chunk(observation)
                latency = time.perf_counter() - started_at
                response = _encode_actions(actions, latency)
            elif command == "reset":
                self._model.reset()
                response = _encode_payload({"version": _PROTOCOL_VERSION, "ok": True, "command": "reset"}, [])
            else:
                raise ZenohRemoteInferenceError(f"Unsupported command: {command!r}")
            query.reply(self._key_expr, response)
        except Exception as error:
            logger.exception("Zenoh remote inference request failed")
            query.reply_err(str(error).encode("utf-8"))


@export_config(class_path="physicalai.runtime.ZenohRemoteExecution")
class ZenohRemoteExecution(Execution):
    """Asynchronously request policy inference through Zenoh query/reply."""

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        key_expr: str = "physicalai/inference/predict",
        request_threshold: float = 0.5,
        request_timeout_s: float = 30.0,
        config_path: str | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._key_expr = key_expr
        self._request_threshold = request_threshold
        self._request_timeout_s = request_timeout_s
        self._config = zenoh.Config.from_file(config_path) if config_path else _make_config(connect_endpoint=endpoint)
        self._session: zenoh.Session | None = None
        self._queue: ChunkedActionQueue | None = None
        self._chunk_size = 0
        self._threshold_count = 0
        self._lock = threading.Lock()
        self._obs_slot: tuple[dict[str, Any], int] | None = None
        self._obs_ready = threading.Event()
        self._running_inference = False
        self._request_time = 0.0
        self._pops_at_request = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._death_cause: BaseException | None = None
        self._inference_count = 0
        self._incarnation = 0
        self._bus: _CallbackBus | None = None
        self._session_id = ""

    def start(self, model: InferenceModel, action_queue: ActionQueue) -> None:
        self.stop()
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Zenoh inference worker is still running")
        self._session = zenoh.open(self._config)
        self._queue = cast("ChunkedActionQueue", action_queue)
        self._stop_event = threading.Event()
        self._obs_ready = threading.Event()
        self._death_cause = None
        self._inference_count = 0
        with self._lock:
            self._incarnation += 1
            self._obs_slot = None
            self._running_inference = False
            self._request_time = 0.0
            self._pops_at_request = 0
        self._thread = threading.Thread(target=self._run, args=(self._stop_event, self._obs_ready), daemon=True)
        self._thread.start()

    def warmup(self, sample_observation: dict[str, Any]) -> None:
        if self._queue is None:
            raise RuntimeError(NOT_STARTED)
        actions, _ = self._request("predict", sample_observation)
        if actions.ndim < 2 or actions.shape[0] == 0:
            raise ZenohRemoteInferenceError("Remote server returned an empty action chunk")
        with self._lock:
            self._chunk_size = actions.shape[0]
            self._threshold_count = max(1, int(self._chunk_size * self._request_threshold))
        self._queue.push_chunk(actions, offset=0)

    def maybe_request(self, observation: dict[str, Any]) -> None:
        if self._queue is None:
            raise RuntimeError(NOT_STARTED)
        if self._thread is not None and not self._thread.is_alive() and self._death_cause is not None:
            raise WorkerDiedError(f"Zenoh inference thread died: {self._death_cause}") from self._death_cause
        if self._queue.below_threshold(self._threshold_count) and not self._busy:
            snapshot = {name: value.copy() if isinstance(value, np.ndarray) else value for name, value in observation.items()}
            with self._lock:
                self._obs_slot = (snapshot, self._incarnation)
                self._request_time = time.perf_counter()
                self._pops_at_request = self._queue.total_pops
            self._obs_ready.set()

    def reset(self, *, reset_model: bool = True) -> None:
        if self._queue is None:
            raise RuntimeError(NOT_STARTED)
        with self._lock:
            self._incarnation += 1
            self._obs_slot = None
        if reset_model:
            self._request("reset", None)

    def stop(self) -> None:
        self._stop_event.set()
        self._obs_ready.set()
        if self._thread is not None:
            self._thread.join(timeout=_JOIN_TIMEOUT_S)
        if self._session is not None:
            self._session.close()
            self._session = None

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def inference_count(self) -> int:
        return self._inference_count

    @property
    def _busy(self) -> bool:
        with self._lock:
            return self._obs_slot is not None or self._running_inference

    def _request(self, command: str, observation: dict[str, Any] | None) -> tuple[np.ndarray, float | None]:
        if self._session is None:
            raise RuntimeError(NOT_STARTED)
        if command == "predict":
            if observation is None:
                raise ValueError("An observation is required for prediction")
            payload = _encode_observation(observation)
        else:
            payload = _encode_payload({"version": _PROTOCOL_VERSION, "command": command}, [])
        try:
            replies = self._session.get(self._key_expr, payload=payload, timeout=self._request_timeout_s)
            reply = replies.recv()
            if reply.err is not None:
                raise ZenohRemoteInferenceError(_payload_bytes(reply.err.payload).decode("utf-8", errors="replace"))
            if reply.ok is None:
                raise ZenohRemoteInferenceError("Zenoh returned an empty reply")
            if command == "predict":
                return _decode_actions(_payload_bytes(reply.ok.payload))
            return np.empty((0,)), None
        except ZenohRemoteInferenceError:
            raise
        except Exception as error:
            raise ZenohRemoteInferenceError(f"Zenoh request to {self._key_expr} failed: {error}") from error

    def _run(self, stop_event: threading.Event, obs_ready: threading.Event) -> None:
        try:
            while not stop_event.is_set():
                obs_ready.wait()
                obs_ready.clear()
                if stop_event.is_set():
                    return
                with self._lock:
                    request = self._obs_slot
                    self._obs_slot = None
                    if request is None:
                        continue
                    observation, incarnation = request
                    self._running_inference = True
                started_at = time.perf_counter()
                actions, server_latency = self._request("predict", observation)
                latency = time.perf_counter() - started_at
                if stop_event.is_set():
                    return
                with self._lock:
                    if incarnation != self._incarnation or self._queue is None:
                        self._running_inference = False
                        continue
                    pops_since = self._queue.total_pops - self._pops_at_request
                    offset = min(max(pops_since, 0), len(actions) - 1)
                    self._queue.push_chunk(actions, offset=offset)
                    self._running_inference = False
                self._inference_count += 1
                if self._bus:
                    from physicalai.runtime.events import InferenceEvent
                    self._bus.emit_inference(
                        InferenceEvent(
                            session_id=self._session_id,
                            timestamp=time.time(),
                            latency_s=latency,
                            offset=offset,
                            chunk=actions,
                            server_inference_latency_s=server_latency,
                        )
                    )
        except Exception as error:
            self._death_cause = error
            logger.exception("Zenoh inference thread died")
