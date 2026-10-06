# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for model-namespaced Zenoh remote inference."""

from __future__ import annotations

import socket
import threading
import time
from typing import Any

import numpy as np
import msgpack
import pytest

zenoh = pytest.importorskip("zenoh")

from physicalai._zenoh import endpoint_for_key  # noqa: E402
from physicalai.inference.constants import ACTION  # noqa: E402
from physicalai.runtime.execution.async_execution import AsyncExecution  # noqa: E402
from physicalai.runtime.execution.queue import ChunkedActionQueue  # noqa: E402
from physicalai.runtime.execution.rtc import RTCExecution  # noqa: E402
from physicalai.runtime.execution.rtc_queue import RTCActionQueue  # noqa: E402
from physicalai.runtime.execution.sync import SyncExecution  # noqa: E402
from physicalai.runtime.zenoh_remote import (  # noqa: E402
    RemoteInferenceModel,
    ZenohRemoteInferenceError,
    RemoteInferenceServer,
    _decode_observation,
    _decode_payload,
    _encode_observation,
)


class _Model:
    def __init__(self, policy_name: str) -> None:
        self.policy_name = policy_name
        self.predict_count = 0
        self.call_count = 0
        self.reset_count = 0
        self.chunk_size = 4
        self.manifest = type("Manifest", (), {"model_extra": {"rtc": {}}})()

    def predict_action_chunk(self, observation: dict[str, Any]) -> np.ndarray:
        self.predict_count += 1
        return np.ones((4, 3), dtype=np.float32)

    def reset(self) -> None:
        self.reset_count += 1

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
    server_b = RemoteInferenceServer(model_b, listen_endpoint=endpoint_b)
    server_thread = threading.Thread(target=server_b.serve_forever, daemon=True)
    server_thread.start()
    assert server_b.wait_until_ready(timeout_s=5.0)

    with pytest.raises(ZenohRemoteInferenceError):
        RemoteInferenceModel(endpoint_b, "model-a", request_timeout_s=0.5)

    client_for_b = RemoteInferenceModel(endpoint_b, "model-b", request_timeout_s=0.5)
    try:
        actions = client_for_b.predict_action_chunk({"state": np.zeros(3, dtype=np.float32)})
        assert actions.shape == (4, 3)
        assert model_b.predict_count == 1
        assert client_for_b.last_server_latency_s is not None
    finally:
        client_for_b.close()
        server_b.stop()
        server_thread.join(timeout=2.0)

    assert not server_thread.is_alive()


def test_server_and_client_derive_port_from_model_namespace() -> None:
    model_name = "pi05-port-test"
    key_prefix = f"physicalai/inference/{model_name}"
    server = RemoteInferenceServer(
        _Model(model_name),
        model_name=model_name,
        listen_host="127.0.0.1",
    )
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    assert server.wait_until_ready(timeout_s=5.0)
    client = None
    try:
        client = RemoteInferenceModel(model_name=model_name, server_host="127.0.0.1")
        assert server.listen_endpoint == endpoint_for_key(key_prefix, "127.0.0.1")
        assert client.endpoint == endpoint_for_key(key_prefix, "127.0.0.1")
    finally:
        if client is not None:
            client.close()
        server.stop()
        server_thread.join(timeout=2.0)
    assert not server_thread.is_alive()


def test_sync_async_and_rtc_use_the_same_remote_model_interface() -> None:
    model_name = "execution-mode-check"
    served_model = _Model(model_name)
    server = RemoteInferenceServer(served_model, model_name=model_name, listen_host="127.0.0.1")
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    assert server.wait_until_ready(timeout_s=5.0)

    model = RemoteInferenceModel(
        model_name=model_name,
        server_host="127.0.0.1",
        server_port=int(server.listen_endpoint.rsplit(":", 1)[1]),
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
        rtc = RTCExecution(chunk_size=4, execution_horizon=2, fps=30.0, queue_threshold=2)
        rtc.start(model, rtc_queue)
        rtc.warmup(observation)
        assert rtc_queue.remaining == 4
        assert served_model.call_count > 0
        assert model.last_server_latency_s is not None
        rtc.stop()
    finally:
        model.close()
        server.stop()
        server_thread.join(timeout=2.0)
    assert not server_thread.is_alive()


def test_observations_use_msgpack_jpeg_images_and_lossless_other_arrays() -> None:
    image = np.empty((1, 128, 160, 3), dtype=np.uint8)
    image[:] = [220, 35, 18]
    state = np.array([[1.25, -2.5]], dtype=np.float32)
    task = np.array(["pick the red cube"])

    payload = _encode_observation({"images.front": image, "state": state, "task": task})
    unpacked = msgpack.unpackb(payload, raw=False, strict_map_key=False)
    metadata, frames = _decode_payload(payload)
    decoded = _decode_observation(payload)

    assert unpacked["metadata"] == metadata
    assert metadata["arrays"][0]["encoding"] == "jpeg"
    assert metadata["arrays"][1]["encoding"] == "raw"
    assert frames[0].startswith(b"\xff\xd8")
    assert len(frames[0]) < image.nbytes
    assert decoded["images.front"].shape == image.shape
    assert decoded["images.front"][0, 32, 32, 0] > 200
    assert decoded["images.front"][0, 32, 32, 2] < 40
    np.testing.assert_array_equal(decoded["state"], state)
    assert decoded["task"] == task.tolist()


def test_grayscale_batched_images_roundtrip_shape() -> None:
    image = np.full((1, 72, 96), 127, dtype=np.uint8)
    decoded = _decode_observation(_encode_observation({"images": image}))

    assert decoded["images"].shape == image.shape
    assert abs(int(decoded["images"][0, 20, 20]) - 127) <= 3


def _wait_until(predicate: Any, timeout_s: float) -> bool:
    end_time = time.monotonic() + timeout_s
    while time.monotonic() < end_time:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())
