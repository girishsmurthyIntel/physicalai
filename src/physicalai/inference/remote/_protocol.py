# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MessagePack and JPEG wire protocol for remote inference."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any, Literal

import cv2
import msgpack
import numpy as np

from physicalai.inference.constants import IMAGES

PROTOCOL_VERSION = 1
DEFAULT_IMAGE_JPEG_QUALITY = 90
DEFAULT_MAX_REQUEST_BYTES = 32 * 2**20
ImageCodec = Literal["jpeg", "raw"]


class RemoteInferenceError(RuntimeError):
    """Base class for remote inference failures."""


class RemoteInferenceUnavailableError(RemoteInferenceError):
    """The configured remote inference server is unavailable."""


class RemoteInferenceTimeoutError(RemoteInferenceError):
    """A remote inference request exceeded its deadline."""


class RemoteInferenceProtocolError(RemoteInferenceError):
    """The peer sent a malformed or incompatible protocol message."""


class RemoteInferenceModelMismatchError(RemoteInferenceError):
    """The peer serves a different or replaced model."""


@dataclass(frozen=True, slots=True)
class RemoteTiming:
    """Timing information for the last completed remote request."""

    seq: int
    round_trip_s: float
    queue_s: float
    compute_s: float


def encode_payload(metadata: dict[str, Any], frames: list[bytes]) -> bytes:
    """Pack metadata and binary frames into a MessagePack envelope."""
    return msgpack.packb({"metadata": metadata, "frames": frames}, use_bin_type=True)


def decode_payload(payload: bytes) -> tuple[dict[str, Any], list[bytes]]:
    """Unpack and validate a MessagePack envelope."""
    try:
        unpacked = msgpack.unpackb(payload, raw=False, strict_map_key=False)
    except (msgpack.UnpackException, TypeError, ValueError, OverflowError) as error:
        raise RemoteInferenceProtocolError("Invalid MessagePack inference payload") from error
    if not isinstance(unpacked, dict):
        raise RemoteInferenceProtocolError("Payload must be a MessagePack map")
    metadata = unpacked.get("metadata")
    frames = unpacked.get("frames")
    if not isinstance(metadata, dict) or not isinstance(frames, list):
        raise RemoteInferenceProtocolError("Payload must contain metadata and frames")
    if not all(isinstance(frame, bytes) for frame in frames):
        raise RemoteInferenceProtocolError("Payload frames must be binary data")
    return metadata, frames


def decode_error(payload: bytes) -> dict[str, Any]:
    """Decode the protocol's compact query.reply_err map."""
    try:
        value = msgpack.unpackb(payload, raw=False, strict_map_key=False)
    except (msgpack.UnpackException, TypeError, ValueError, OverflowError) as error:
        raise RemoteInferenceProtocolError("Invalid remote error reply") from error
    if not isinstance(value, dict):
        raise RemoteInferenceProtocolError("Remote error reply must be a MessagePack map")
    return value


def array_descriptor(name: str, array: np.ndarray, encoding: str = "raw") -> dict[str, Any]:
    """Describe a binary array frame."""
    return {"name": name, "dtype": array.dtype.str, "shape": list(array.shape), "encoding": encoding}


def _is_image_input(name: str) -> bool:
    return name == IMAGES or name.startswith(f"{IMAGES}.")


def _scaled_image(image: np.ndarray, max_side: int | None) -> np.ndarray:
    if image.dtype != np.uint8 or image.ndim not in (3, 4) or image.shape[-1] != 3:
        raise RemoteInferenceProtocolError("Image inputs must be uint8 with a trailing channel dimension of 3")
    height, width = image.shape[-3:-1]
    if max_side is None or max(height, width) <= max_side:
        return image
    scale = max_side / max(height, width)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    if image.ndim == 3:
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return np.stack([cv2.resize(item, new_size, interpolation=cv2.INTER_AREA) for item in image])


def _encode_jpeg(image: np.ndarray, quality: int) -> bytes:
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise RemoteInferenceProtocolError("JPEG image batch dimension must be one")
        image = image[0]
    rgb = image[..., ::-1]
    success, encoded = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise RemoteInferenceProtocolError("Image encoder failed")
    return encoded.tobytes()


def encode_request(
    inputs: dict[str, Any],
    *,
    seq: int,
    budget_ms: int,
    api: Literal["call", "action_chunk"],
    image_codec: ImageCodec = "jpeg",
    jpeg_quality: int = DEFAULT_IMAGE_JPEG_QUALITY,
    max_image_side: int | None = None,
) -> bytes:
    """Encode a predict request with sequence, deadline, and array descriptors."""
    if image_codec not in ("jpeg", "raw"):
        raise ValueError("image_codec must be 'jpeg' or 'raw'")
    if not 0 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 0 and 100")
    if max_image_side is not None and max_image_side < 1:
        raise ValueError("max_image_side must be positive")
    arrays: list[dict[str, Any]] = []
    frames: list[bytes] = []
    for name, value in inputs.items():
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            arrays.append({"name": name, "encoding": "msgpack_strings", "value": value})
            continue
        try:
            array = np.ascontiguousarray(np.asarray(value))
        except (TypeError, ValueError) as error:
            raise RemoteInferenceProtocolError(f"Input {name!r} is not a supported array") from error
        if array.dtype.kind not in "biufc":
            raise RemoteInferenceProtocolError(f"Input {name!r} must have a numeric or bool dtype")
        encoding = "raw"
        if _is_image_input(name):
            array = np.ascontiguousarray(_scaled_image(array, max_image_side))
            if image_codec == "jpeg":
                frame = _encode_jpeg(array, jpeg_quality)
                encoding = "jpeg"
            else:
                frame = array.tobytes()
        else:
            frame = array.tobytes()
        arrays.append(array_descriptor(name, array, encoding))
        frames.append(frame)
    return encode_payload(
        {
            "protocol_version": PROTOCOL_VERSION,
            "command": "predict",
            "api": api,
            "seq": seq,
            "budget_ms": budget_ms,
            "arrays": arrays,
        },
        frames,
    )


def _checked_shape_dtype(descriptor: object, *, max_bytes: int) -> tuple[str, np.dtype[Any], tuple[int, ...], int, str]:
    if not isinstance(descriptor, dict):
        raise RemoteInferenceProtocolError("Invalid array descriptor")
    name = descriptor.get("name")
    shape_value = descriptor.get("shape")
    encoding = descriptor.get("encoding")
    if not isinstance(name, str) or not isinstance(shape_value, list) or not isinstance(encoding, str):
        raise RemoteInferenceProtocolError("Invalid array descriptor")
    if any(not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in shape_value):
        raise RemoteInferenceProtocolError("Invalid array shape")
    try:
        dtype = np.dtype(descriptor.get("dtype"))
    except (TypeError, ValueError) as error:
        raise RemoteInferenceProtocolError("Invalid array dtype") from error
    if dtype.kind not in "biufc":
        raise RemoteInferenceProtocolError("Array dtype must be numeric or bool")
    shape = tuple(shape_value)
    elements = math.prod(shape)
    if elements > max_bytes // max(1, dtype.itemsize):
        raise RemoteInferenceProtocolError("Decoded arrays exceed max_request_bytes")
    return name, dtype, shape, elements * dtype.itemsize, encoding


def _jpeg_dimensions(frame: bytes) -> tuple[int, int]:
    if len(frame) < 4 or frame[:2] != b"\xff\xd8":
        raise RemoteInferenceProtocolError("Malformed JPEG image frame")
    sof_markers = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    offset = 2
    while offset + 4 <= len(frame):
        if frame[offset] != 0xFF:
            raise RemoteInferenceProtocolError("Malformed JPEG marker")
        while offset < len(frame) and frame[offset] == 0xFF:
            offset += 1
        if offset >= len(frame):
            break
        marker = frame[offset]
        offset += 1
        if marker in {0xD8, 0xD9, 0x01, *range(0xD0, 0xD8)}:
            continue
        if offset + 2 > len(frame):
            break
        segment_length = struct.unpack_from(">H", frame, offset)[0]
        if segment_length < 2 or offset + segment_length > len(frame):
            raise RemoteInferenceProtocolError("Malformed JPEG segment")
        if marker in sof_markers:
            if segment_length < 7:
                raise RemoteInferenceProtocolError("Malformed JPEG frame header")
            height, width = struct.unpack_from(">HH", frame, offset + 3)
            if height == 0 or width == 0:
                raise RemoteInferenceProtocolError("Invalid JPEG dimensions")
            return height, width
        offset += segment_length
    raise RemoteInferenceProtocolError("JPEG frame has no dimensions")


def decode_request(
    payload: bytes,
    *,
    max_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
) -> tuple[dict[str, Any], dict[str, np.ndarray | list[str]]]:
    """Validate and decode request arrays without allocating beyond max_bytes."""
    if len(payload) > max_bytes:
        raise RemoteInferenceProtocolError("Request exceeds max_request_bytes")
    metadata, frames = decode_payload(payload)
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        raise RemoteInferenceProtocolError("Unsupported request protocol version")
    if metadata.get("command") != "predict":
        raise RemoteInferenceProtocolError("Invalid predict command")
    seq = metadata.get("seq")
    budget_ms = metadata.get("budget_ms")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
        raise RemoteInferenceProtocolError("Invalid request sequence")
    if not isinstance(budget_ms, int) or isinstance(budget_ms, bool) or budget_ms < 1:
        raise RemoteInferenceProtocolError("Invalid request budget")
    if metadata.get("api") not in ("call", "action_chunk"):
        raise RemoteInferenceProtocolError("Invalid inference API")
    descriptors = metadata.get("arrays")
    if not isinstance(descriptors, list):
        raise RemoteInferenceProtocolError("Request arrays must be a list")
    binary_descriptors = [
        item for item in descriptors if isinstance(item, dict) and item.get("encoding") != "msgpack_strings"
    ]
    if len(binary_descriptors) != len(frames):
        raise RemoteInferenceProtocolError("Request metadata does not match data frames")
    inputs: dict[str, np.ndarray | list[str]] = {}
    frame_index = 0
    total_decoded_bytes = 0
    seen_names: set[str] = set()
    for descriptor in descriptors:
        if not isinstance(descriptor, dict):
            raise RemoteInferenceProtocolError("Invalid array descriptor")
        name = descriptor.get("name")
        if not isinstance(name, str) or not name or name in seen_names:
            raise RemoteInferenceProtocolError("Invalid or duplicate input name")
        seen_names.add(name)
        encoding = descriptor.get("encoding")
        if encoding == "msgpack_strings":
            values = descriptor.get("value")
            if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
                raise RemoteInferenceProtocolError("String inputs must be MessagePack string arrays")
            total_decoded_bytes += sum(len(item.encode("utf-8")) for item in values)
            if total_decoded_bytes > max_bytes:
                raise RemoteInferenceProtocolError("Decoded arrays exceed max_request_bytes")
            inputs[name] = values
            continue
        name, dtype, shape, decoded_size, encoding = _checked_shape_dtype(descriptor, max_bytes=max_bytes)
        total_decoded_bytes += decoded_size
        if total_decoded_bytes > max_bytes:
            raise RemoteInferenceProtocolError("Decoded arrays exceed max_request_bytes")
        frame = frames[frame_index]
        frame_index += 1
        if encoding == "jpeg":
            if not _is_image_input(name) or dtype != np.dtype(np.uint8) or len(shape) not in (3, 4) or shape[-1] != 3:
                raise RemoteInferenceProtocolError("JPEG image descriptor is invalid")
            if _jpeg_dimensions(frame) != shape[-3:-1]:
                raise RemoteInferenceProtocolError("JPEG dimensions do not match the declared shape")
            image = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise RemoteInferenceProtocolError("Image decoder rejected the JPEG frame")
            image = image[..., ::-1]
            if len(shape) == 4:
                if shape[0] != 1:
                    raise RemoteInferenceProtocolError("JPEG batch dimension must be one")
                image = image[np.newaxis]
            inputs[name] = np.ascontiguousarray(image)
        elif encoding == "raw":
            if _is_image_input(name) and (dtype != np.dtype(np.uint8) or len(shape) not in (3, 4) or shape[-1] != 3):
                raise RemoteInferenceProtocolError("Image inputs must be uint8 with a trailing channel dimension of 3")
            if len(frame) != decoded_size:
                raise RemoteInferenceProtocolError(f"Input {name!r} has invalid data length")
            inputs[name] = np.frombuffer(frame, dtype=dtype).reshape(shape).copy()
        else:
            raise RemoteInferenceProtocolError(f"Unsupported array encoding: {encoding!r}")
    return metadata, inputs


def encode_response(
    outputs: dict[str, Any],
    *,
    seq: int,
    server_id: str,
    queue_ms: float,
    compute_ms: float,
) -> bytes:
    """Encode output arrays and timing metadata."""
    descriptors: list[dict[str, Any]] = []
    frames: list[bytes] = []
    for name, value in outputs.items():
        array = np.ascontiguousarray(np.asarray(value))
        if array.dtype.kind not in "biufc":
            raise RemoteInferenceProtocolError(f"Output {name!r} must have a numeric or bool dtype")
        descriptors.append(array_descriptor(name, array))
        frames.append(array.tobytes())
    return encode_payload(
        {
            "protocol_version": PROTOCOL_VERSION,
            "seq": seq,
            "server_id": server_id,
            "ok": True,
            "queue_ms": queue_ms,
            "compute_ms": compute_ms,
            "arrays": descriptors,
        },
        frames,
    )


def decode_response(payload: bytes) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Decode and validate a successful inference response."""
    metadata, frames = decode_payload(payload)
    if metadata.get("protocol_version") != PROTOCOL_VERSION or metadata.get("ok") is not True:
        raise RemoteInferenceProtocolError("Invalid inference response")
    descriptors = metadata.get("arrays")
    if not isinstance(descriptors, list) or len(descriptors) != len(frames):
        raise RemoteInferenceProtocolError("Response metadata does not match data frames")
    outputs: dict[str, np.ndarray] = {}
    for descriptor, frame in zip(descriptors, frames, strict=True):
        name, dtype, shape, expected_size, encoding = _checked_shape_dtype(
            descriptor, max_bytes=DEFAULT_MAX_REQUEST_BYTES
        )
        if encoding != "raw" or len(frame) != expected_size:
            raise RemoteInferenceProtocolError(f"Output {name!r} has invalid data")
        outputs[name] = np.frombuffer(frame, dtype=dtype).reshape(shape).copy()
    return metadata, outputs


def encode_observation(
    observation: dict[str, Any],
    image_jpeg_quality: int = DEFAULT_IMAGE_JPEG_QUALITY,
    command: str = "predict",
) -> bytes:
    """Backward-compatible encoder used by small protocol tests."""
    return encode_request(
        observation,
        seq=0,
        budget_ms=2000,
        api="action_chunk" if command == "predict" else "call",
        jpeg_quality=image_jpeg_quality,
    )


def decode_observation_parts(metadata: dict[str, Any], frames: list[bytes], command: str = "predict") -> dict[str, Any]:
    """Decode observations from an already-unpacked legacy envelope."""
    wrapped = encode_payload(metadata, frames)
    _, inputs = decode_request(wrapped)
    if command != "predict":
        raise RemoteInferenceProtocolError("Unsupported inference command")
    return inputs


def decode_observation(payload: bytes) -> dict[str, Any]:
    """Decode a complete request payload."""
    _, inputs = decode_request(payload)
    return inputs


def encode_actions(actions: np.ndarray, server_latency_s: float | None = None) -> bytes:
    """Compatibility encoder for an action-only response."""
    latency_ms = 0.0 if server_latency_s is None else server_latency_s * 1000
    return encode_response({"action": actions}, seq=0, server_id="", queue_ms=0.0, compute_ms=latency_ms)


def decode_actions(payload: bytes) -> tuple[np.ndarray, float | None]:
    """Compatibility decoder for an action-only response."""
    metadata, outputs = decode_response(payload)
    if "action" not in outputs:
        raise RemoteInferenceProtocolError("Response has no action output")
    return outputs["action"], float(metadata.get("compute_ms", 0.0)) / 1000


def encode_model_outputs(outputs: dict[str, Any], server_latency_s: float) -> bytes:
    """Compatibility encoder for generic model outputs."""
    return encode_actions(outputs.get("action", np.empty((0,))), server_latency_s)


def decode_model_outputs(payload: bytes) -> tuple[dict[str, np.ndarray], float | None]:
    """Compatibility decoder for generic model outputs."""
    metadata, outputs = decode_response(payload)
    latency_ms = metadata.get("compute_ms")
    return outputs, None if latency_ms is None else float(latency_ms) / 1000
