# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for model-namespaced Zenoh remote inference."""

from __future__ import annotations

import inspect
import socket
import threading
import time
from typing import Any

import msgpack
import numpy as np
import pytest

zenoh = pytest.importorskip("zenoh")

from physicalai.inference.constants import ACTION  # noqa: E402
from physicalai.inference.remote import (  # noqa: E402
    InferenceServer,
    RemoteInferenceModel,
    RemoteInferenceModelMismatchError,
)
from physicalai.inference.remote import client as client_module  # noqa: E402
from physicalai.inference.remote._protocol import (  # noqa: E402
    RemoteInferenceError,
    RemoteInferenceProtocolError,
    decode_observation,
    decode_payload,
    decode_request,
    encode_observation,
    encode_payload,
    encode_request,
)
from physicalai.inference.remote.server import _Request  # noqa: E402
from physicalai.runtime.execution.async_execution import AsyncExecution  # noqa: E402
from physicalai.runtime.execution.queue import ChunkedActionQueue  # noqa: E402
from physicalai.runtime.execution.rtc import RTCExecution  # noqa: E402
from physicalai.runtime.execution.rtc_queue import RTCActionQueue  # noqa: E402
from physicalai.runtime.execution.sync import SyncExecution  # noqa: E402
from physicalai.transport._zenoh import endpoint_for_key  # noqa: E402


class _Model:
    def __init__(self, policy_name: str) -> None:
        self.policy_name = policy_name
        self.predict_count = 0
        self.call_count = 0
        self.reset_count = 0
        self.reset_thread_id: int | None = None
        self.chunk_size = 4
        self.manifest = type("Manifest", (), {"model_extra": {"rtc": {}}})()

    def predict_action_chunk(self, observation: dict[str, Any]) -> np.ndarray:
        self.predict_count += 1
        return np.ones((4, 3), dtype=np.float32)

    def reset(self) -> None:
        self.reset_count += 1
        self.reset_thread_id = threading.get_ident()

    def __call__(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        self.call_count += 1
        return {ACTION: self.predict_action_chunk(inputs)}


def _free_endpoint() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"tcp/127.0.0.1:{port}"


def test_client_rejects_server_with_different_model_namespace() -> None:
    model_b = _Model("model-b")
    endpoint_b = _free_endpoint()
    server_b = InferenceServer(model_b, "model-b", listen=endpoint_b)
    server_b.start()
    assert len(server_b._queryables) == 3  # noqa: SLF001
    mismatched_client = RemoteInferenceModel(
        "model-b",
        endpoint=endpoint_b,
        expected_policy="another-policy",
        request_timeout_s=0.5,
    )

    with pytest.raises(RemoteInferenceModelMismatchError):
        mismatched_client.connect()
    mismatched_client.close()

    client_for_b = RemoteInferenceModel("model-b", endpoint=endpoint_b)
    try:
        actions = client_for_b.predict_action_chunk({"state": np.zeros(3, dtype=np.float32)})
        assert actions.shape == (4, 3)
        assert model_b.predict_count == 1
        assert client_for_b.last_timing is not None
        assert client_for_b.metadata["name"] == "model-b"
        assert client_for_b.metadata["server_id"] == server_b._server_id  # noqa: SLF001
        assert client_for_b.last_timing.seq == 2
    finally:
        client_for_b.close()
        server_b.stop()


def test_server_error_reply_does_not_expose_exception_text() -> None:
    class _FailingModel(_Model):
        def predict_action_chunk(self, observation: dict[str, Any]) -> np.ndarray:
            raise RuntimeError("private model failure details")

    name = "error-reply-test"
    server = InferenceServer(_FailingModel(name), name, listen=_free_endpoint())
    server.start()
    client = RemoteInferenceModel(name, endpoint=server.endpoint, request_timeout_s=0.5)
    try:
        with pytest.raises(RemoteInferenceError) as error:
            client.predict_action_chunk({"state": np.zeros(3, dtype=np.float32)})
        assert "private model failure details" not in str(error.value)
    finally:
        client.close()
        server.stop()


def test_remote_model_default_request_timeout_is_two_seconds() -> None:
    timeout_parameter = inspect.signature(RemoteInferenceModel.__init__).parameters["request_timeout_s"]

    assert timeout_parameter.default == 2.0


def test_client_constructor_is_lazy_and_exports_its_public_config_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_opened(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("constructor must not open a Zenoh session")

    monkeypatch.setattr(client_module, "open_zenoh_session", fail_if_opened)
    model = RemoteInferenceModel("lazy-construction", endpoint="tcp/127.0.0.1:14567")

    assert not hasattr(model, "manifest")
    assert model.endpoint == "tcp/127.0.0.1:14567"
    assert model.metadata == {}

    from physicalai.config import Config
    from physicalai.inference import InferenceModel
    from physicalai.runtime.action_sources.policy import PolicySource

    recipe = Config.from_instance(model)
    assert recipe.class_path == "physicalai.inference.RemoteInferenceModel"
    restored = recipe.instantiate(expected_type=InferenceModel)
    assert isinstance(restored, InferenceModel)
    assert not hasattr(restored, "manifest")
    source_recipe = Config.from_instance(PolicySource(model=model))
    restored_source = source_recipe.instantiate()
    assert isinstance(restored_source, PolicySource)


def test_custom_zenoh_config_replaces_the_built_in_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    def fail_if_built(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("custom config must replace built-in config construction")

    monkeypatch.setattr(client_module, "make_zenoh_config", fail_if_built)
    model = RemoteInferenceModel("custom-config", zenoh_config=tmp_path / "custom.json5")

    assert model._builtin_config is None  # noqa: SLF001


def test_client_fails_unavailable_when_no_queryable_matches() -> None:
    model = RemoteInferenceModel("unavailable-test", endpoint=_free_endpoint(), request_timeout_s=0.05)

    with pytest.raises(RemoteInferenceError, match="SSH tunnel up / address reachable"):
        model.connect()
    model.close()


def test_string_lists_remain_msgpack_strings_and_other_arrays_are_raw() -> None:
    payload = encode_request(
        {"task": ["pick", "place"], "state": np.asarray([1, 2], dtype=np.int16)},
        seq=17,
        budget_ms=2000,
        api="call",
    )
    metadata, frames = decode_payload(payload)
    decoded_metadata, inputs = decode_request(payload)

    assert decoded_metadata["seq"] == 17
    assert metadata["arrays"][0]["encoding"] == "msgpack_strings"
    assert metadata["arrays"][0]["value"] == ["pick", "place"]
    assert metadata["arrays"][1]["encoding"] == "raw"
    assert len(frames) == 1
    assert inputs["task"] == ["pick", "place"]
    np.testing.assert_array_equal(inputs["state"], [1, 2])


def test_request_decoder_rejects_oversized_and_non_numeric_arrays() -> None:
    payload = encode_request(
        {"state": np.zeros(1024, dtype=np.float32)},
        seq=1,
        budget_ms=1000,
        api="call",
    )
    with pytest.raises(RemoteInferenceProtocolError, match="max_request_bytes"):
        decode_request(payload, max_bytes=256)

    invalid = encode_payload(
        {
            "protocol_version": 1,
            "command": "predict",
            "api": "call",
            "seq": 1,
            "budget_ms": 1000,
            "arrays": [{"name": "state", "dtype": "|O", "shape": [1], "encoding": "raw"}],
        },
        [b"x"],
    )
    with pytest.raises(RemoteInferenceProtocolError, match="numeric or bool"):
        decode_request(invalid)


def test_new_predict_supersedes_the_only_pending_prediction() -> None:
    class _Query:
        def __init__(self) -> None:
            self.error: bytes | None = None

        def reply_err(self, payload: bytes) -> None:
            self.error = payload

    server = InferenceServer(_Model("newest-wins"), "newest-wins")
    old_query = _Query()
    latest_query = _Query()
    now = time.monotonic()
    server._enqueue(_Request(old_query, "predict", 10, 1000, now, 12, "call", {}))  # noqa: SLF001
    server._enqueue(_Request(latest_query, "predict", 11, 1000, now, 13, "call", {}))  # noqa: SLF001

    assert old_query.error is not None
    assert msgpack.unpackb(old_query.error, raw=False)["code"] == "superseded"
    assert len(server._queue) == 1  # noqa: SLF001
    assert server._queue[0].seq == 11  # noqa: SLF001
    server._queue.clear()  # noqa: SLF001
    server.stop()


def test_worker_expires_a_request_before_calling_the_model() -> None:
    class _Query:
        def __init__(self) -> None:
            self.event = threading.Event()
            self.error: bytes | None = None

        def reply_err(self, payload: bytes) -> None:
            self.error = payload
            self.event.set()

    model = _Model("deadline-test")
    server = InferenceServer(model, "deadline-test")
    server._last_summary = time.monotonic()  # noqa: SLF001
    server._worker = threading.Thread(target=server._worker_loop, daemon=True)  # noqa: SLF001
    server._worker.start()  # noqa: SLF001
    query = _Query()
    server._enqueue(  # noqa: SLF001
        _Request(query, "predict", 15, 1, time.monotonic() - 1, 10, "action_chunk", {"state": np.zeros(3)})
    )

    assert query.event.wait(1.0)
    assert query.error is not None
    assert msgpack.unpackb(query.error, raw=False)["code"] == "expired"
    assert model.predict_count == 0
    server.stop()


def test_server_and_client_derive_port_from_model_namespace() -> None:
    model_name = "pi05-port-test"
    key_prefix = f"physicalai/inference/{model_name}"
    server = InferenceServer(_Model(model_name), model_name)
    client = RemoteInferenceModel(model_name)

    assert server.endpoint == endpoint_for_key(key_prefix, "127.0.0.1")
    assert client.endpoint == endpoint_for_key(key_prefix, "127.0.0.1")


def test_reset_roundtrip_runs_on_the_server_worker() -> None:
    name = "reset-roundtrip"
    model = _Model(name)
    server = InferenceServer(model, name, listen=_free_endpoint())
    server.start()
    client = RemoteInferenceModel(name, endpoint=server.endpoint)
    try:
        client.predict_action_chunk({"state": np.zeros(3, dtype=np.float32)})
        client.reset()
        assert model.reset_count == 1
        assert model.reset_thread_id == server._worker.ident  # noqa: SLF001
    finally:
        client.close()
        server.stop()


def test_predict_querier_can_be_reused() -> None:
    name = "repeat-predict"
    server = InferenceServer(_Model(name), name, listen=_free_endpoint())
    server.start()
    client = RemoteInferenceModel(name, endpoint=server.endpoint)
    try:
        for _ in range(3):
            assert client.predict_action_chunk({"state": np.zeros(3, dtype=np.float32)}).shape == (4, 3)
    finally:
        client.close()
        server.stop()


def test_client_uses_rtc_chunk_size_from_handshake_without_manifest() -> None:
    name = "rtc-handshake"
    model = _Model(name)
    model.manifest.model_extra["rtc"] = {"chunk_size": 6}
    server = InferenceServer(model, name, listen=_free_endpoint())
    server.start()
    client = RemoteInferenceModel(name, endpoint=server.endpoint)
    try:
        assert client.chunk_size == 6
        assert not hasattr(client, "manifest")
    finally:
        client.close()
        server.stop()


def test_client_rejects_action_batch_larger_than_one() -> None:
    class _BatchedModel(_Model):
        def __call__(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
            return {ACTION: np.ones((2, 4, 3), dtype=np.float32)}

    name = "batched-actions"
    server = InferenceServer(_BatchedModel(name), name, listen=_free_endpoint())
    server.start()
    client = RemoteInferenceModel(name, endpoint=server.endpoint)
    try:
        with pytest.raises(RemoteInferenceProtocolError, match="batch size 1"):
            client.predict_action_chunk({"state": np.zeros(3, dtype=np.float32)})
    finally:
        client.close()
        server.stop()


def test_sync_async_and_rtc_use_the_same_remote_model_interface() -> None:
    model_name = "execution-mode-check"
    served_model = _Model(model_name)
    server = InferenceServer(served_model, model_name, listen=_free_endpoint())
    server.start()

    model = RemoteInferenceModel(
        model_name,
        endpoint=server.endpoint,
        request_timeout_s=0.5,
    )
    observation = {"state": np.zeros(3, dtype=np.float32)}
    try:
        sync_queue = ChunkedActionQueue()
        sync = SyncExecution()
        sync.start(model, sync_queue)
        sync.warmup(observation)
        assert sync_queue.remaining == model.chunk_size
        sync_queue.clear()
        sync.stop()

        async_queue = ChunkedActionQueue()
        asynchronous = AsyncExecution(request_threshold=0.75)
        asynchronous.start(model, async_queue)
        asynchronous.warmup(observation)
        for _ in range(3):
            async_queue.pop()
        asynchronous.maybe_request(observation)
        assert _wait_until(lambda: asynchronous.inference_count > 0 and async_queue.remaining > 0, 5.0)
        asynchronous.stop()

        rtc_queue = RTCActionQueue()
        rtc = RTCExecution(chunk_size=None, execution_horizon=2, fps=30.0, queue_threshold=2)
        rtc.start(model, rtc_queue)
        rtc.warmup(observation)
        assert rtc_queue.remaining == 4
        assert served_model.call_count > 0
        assert model.last_timing is not None
        rtc.stop()
    finally:
        model.close()
        server.stop()


def test_observations_use_msgpack_jpeg_images_and_lossless_other_arrays() -> None:
    image = np.empty((1, 128, 160, 3), dtype=np.uint8)
    image[:] = [220, 35, 18]
    state = np.array([[1.25, -2.5]], dtype=np.float32)
    task = ["pick the red cube"]

    payload = encode_observation({"images.front": image, "state": state, "task": task})
    unpacked = msgpack.unpackb(payload, raw=False, strict_map_key=False)
    metadata, frames = decode_payload(payload)
    decoded = decode_observation(payload)

    assert unpacked["metadata"] == metadata
    assert metadata["arrays"][0]["encoding"] == "jpeg"
    assert metadata["arrays"][1]["encoding"] == "raw"
    assert frames[0].startswith(b"\xff\xd8")
    assert len(frames[0]) < image.nbytes
    assert decoded["images.front"].shape == image.shape
    assert decoded["images.front"][0, 32, 32, 0] > 200
    assert decoded["images.front"][0, 32, 32, 2] < 40
    np.testing.assert_array_equal(decoded["state"], state)
    assert decoded["task"] == task


def test_batched_rgb_images_roundtrip_shape() -> None:
    image = np.full((1, 72, 96, 3), 127, dtype=np.uint8)
    decoded = decode_observation(encode_observation({"images": image}))

    assert decoded["images"].shape == image.shape
    assert abs(int(decoded["images"][0, 20, 20, 0]) - 127) <= 3


def test_max_image_side_resizes_and_preserves_rgb_shape() -> None:
    image = np.zeros((96, 128, 3), dtype=np.uint8)
    payload = encode_request(
        {"images.front": image},
        seq=1,
        budget_ms=1000,
        api="call",
        max_image_side=32,
    )
    _, inputs = decode_request(payload)

    assert inputs["images.front"].shape == (24, 32, 3)


def _wait_until(predicate: Any, timeout_s: float) -> bool:
    end_time = time.monotonic() + timeout_s
    while time.monotonic() < end_time:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())
