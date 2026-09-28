# SONIC Teleoperation Quick Run Guide


## 0. Start the Robot (G1)

1. Secure the G1 robot to the gantry and make sure both feet are in contact with the ground.
2. Press the power button briefly, then press and hold it again until the blue light on the head turns on and stops blinking.
3. Wait for calibration to finish. The robot light should then turn **purple**.
## 1. Robot — Camera

SSH into the robot:

```bash
ssh unitree@192.168.123.164
```
Then enter:

```bash
123
```

for the password, and choose:

```bash
1
```

for ROS 2 Foxy if prompted.
Start the RealSense camera:

```bash
conda activate vision
cd ~/SONIC_psi0_release
python -m gear_sonic.camera.composed_camera --ego-view-camera realsense --port 5555
```

Check that port `5555` is running:

```bash
ss -ltnp | grep :5555
```

If an old process is using the port:

```bash
kill -9 <PID>
```

---

## 2. Robot — DEX1 Gripper

Open another robot terminal:

```bash
cd dex1_1_service/bin
sudo ./dex1_1_gripper_server --network eth0
```

Keep it running.

---

## 3. Workstation — Terminal 1

Start the SONIC controller:

```bash
cd ~/Psi0/third_party/GR00T-WholeBodyControl/gear_sonic_deploy
export TensorRT_ROOT=$HOME/TensorRT

./docker/run-ros2-dev.sh
```

Inside the container:

```bash
source scripts/setup_env.sh
./deploy.sh --input-type zmq_manager real --hand-type dex1
```

Wait until you see:

```text
Init done
```

---

## 4. Workstation — Terminal 2

Start PICO teleoperation:

```bash
cd ~/Psi0
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh pico
```

Alternative:

```bash
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh pico_mirror
```

---

## 5. Workstation — Terminal 3

Start the data exporter:

```bash
cd ~/Psi0
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh exporter
```

---


# Start Teleoperation

After everything above is running:

1. Put the robot in the calibration pose.
2. Press:

```text
A + B + X + Y
```

to calibrate.

3. Press:

```text
A + X
```

to enable teleoperation.

---

# Record an Episode

Start / stop recording:

```text
Left Grip + A
```

Discard the episode:

```text
Left Grip + B
```

---

# Simulation

Open the simulator:

```bash
cd ~/Psi0/third_party/GR00T-WholeBodyControl
source .venv_teleop/bin/activate
python gear_sonic/scripts/run_sim_loop.py
```

In the deploy terminal use:

```bash
source scripts/setup_env.sh

./deploy.sh --input-type zmq_manager sim
```

---

# Quick Checklist

```text
Robot:
[ ] Camera running
[ ] DEX1 server running

Workstation:
[ ] Terminal 1: controller -> Init done
[ ] Terminal 2: PICO
[ ] Terminal 3: exporter

```


| Action | Button | Notes |
|---|---|---|
| **Start / Stop policy** | **A+B+X+Y** | First press: engage + CALIB_FULL. Again: emergency stop → OFF. |
| **Toggle POSE** | **A+X** | Switches between PLANNER ↔ POSE. OR from VR_3PT (entered via PLANNER) → POSE. |
| **Toggle PLANNER_FROZEN_UPPER** | **B+Y** | Switches between POSE ↔ PLANNER_FROZEN_UPPER. OR from VR_3PT (entered via PLANNER_FROZEN_UPPER) → POSE. |

