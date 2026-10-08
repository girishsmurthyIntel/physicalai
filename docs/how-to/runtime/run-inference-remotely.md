# Run Inference Remotely over Zenoh

Use a `RemoteInferenceModel` on the robot client and a normal exported `InferenceModel` on the inference server. Choose `SyncExecution`, `AsyncExecution`, or `RTCExecution` independently in the client runtime configuration.

The client and server must use the same name. Zenoh keys are scoped below `physicalai/inference/{name}`, and the default TCP port is derived from that namespace. At startup, the client connects to one configured endpoint with multicast and gossip disabled, then checks the protocol version, policy identity, and server identity in a handshake.

## Start the Server

Install/update the repository on the inference host and install the transport extra:

```bash
cd ~/local_dev/physicalai
uv sync --extra transport
```

Load and serve the exported policy:

```bash
uv run physicalai inference serve --name pi05 \
  --export-dir ~/models/pi05/openvino \
  --policy-name pi05 \
  --backend openvino \
  --device GPU
```

By default the server listens on loopback and prints an SSH tunnel command. On the client host, use that command to forward the server's port. To find the deterministic port without starting the model server:

```bash
uv run physicalai inference port pi05
```

An explicit `--port PORT` changes the server's port. For direct LAN access, bind all interfaces explicitly and opt in:

```bash
uv run physicalai inference serve --name pi05 \
  --export-dir ~/models/pi05/openvino \
  --listen tcp/0.0.0.0:45000 \
  --allow-all-interfaces
```

Set the client model's endpoint to `tcp/192.0.2.10:45000` (replace the example address with the inference host). The server never selects a fallback port when the requested port is occupied.

## Configure the Robot Client

Install the transport, robot, camera, and optional Rerun dependencies on the robot host:

```bash
cd ~/local_dev/physicalai
uv sync --extra transport --extra so101 --extra capture --extra observer-rerun
```

Configure `PolicySource.model` as the remote model. The scheduler remains a standard runtime execution strategy:

```yaml
runtime:
  robot:
    class_path: physicalai.robot.SO101
    init_args:
      port: /dev/ttyACM0
      calibration: ./robot_calibration.json

  action_source:
    class_path: physicalai.runtime.PolicySource
    init_args:
      model:
        class_path: physicalai.inference.RemoteInferenceModel
        init_args:
          name: pi05
          request_timeout_s: 2.0
      execution:
        class_path: physicalai.runtime.AsyncExecution
        init_args:
          request_threshold: 0.5
      task: "pick up the red cube and place it in the blue bowl"

  cameras:
    wrist:
      class_path: physicalai.capture.UVCCamera
      init_args:
        device: /dev/video4
        width: 640
        height: 480
        fps: 30
  fps: 30.0
```

Run the configured runtime:

```bash
uv run physicalai run --config policy_runtime_remote_zenoh.yaml --run.duration_s=300
```

Use `SyncExecution` to compare blocking inference. For RTC, configure `RTCExecution` together with `RTCActionQueue`; chunk size and RTC metadata are advertised by the server handshake. The client constructor performs no network I/O, so its configuration can be round-tripped through YAML. The wire protocol uses MessagePack string arrays, JPEG-compressed RGB images by default, and lossless binary frames for other numeric inputs.
