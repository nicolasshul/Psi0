# Ψ0 Deployment Guide
---

## 0. Start the Robot (G1)

1. Secure the G1 robot to the gantry and make sure both feet are in contact with the ground.
2. Press the power button briefly, then press and hold it again until the blue light on the head turns on and stops blinking.
3. Wait for calibration to finish. The robot light should then turn **purple**.

---

## 1. Robot — RealSense Camera Server

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
Start the camera server:

```bash
conda activate vision
cd ~/SONIC_psi0_release
python realsense_server.py
```

If an old video hub process is running, stop it before restarting the camera server:

```bash
sudo pkill -f videohub_pc4
```

Keep this terminal running.

---

## 2. Robot — DEX1 Gripper Server

Open another terminal connected to the robot:

```bash
ssh unitree@192.168.123.164
```

Start the DEX1 gripper server:

```bash
cd dex1_1_service/bin
sudo ./dex1_1_gripper_server --network eth0
```

Keep this terminal running.

---

## 3. Workstation — Terminal 1: SONIC Controller

If Docker permission has not been configured for the current user, run once:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

Set the TensorRT path and launch the ROS 2 development container:

```bash
export TensorRT_ROOT=~/TensorRT
cd ~/Psi0/third_party/GR00T-WholeBodyControl/gear_sonic_deploy
./docker/run-ros2-dev.sh
```

Inside the container, configure the environment and start the real-robot controller:

```bash
source scripts/setup_env.sh
./deploy.sh real --hand-type dex1
```

Wait until the terminal shows:

```text
Init done
```

In the same terminal:

1. Press `#` to set the robot upright.
2. Press `]` after the robot has reached the required position. Robot will stand up when you press `]`.
3. Wait until you see `transitioning to CONTROL state`.
4. Press **Enter**.
5. Wait until you see `ZMQ STREAMING MODE: ENABLED`.

Keep this terminal running.

---

## 4. Workstation — Terminal 2: Ψ0 Policy Server

The policy server loads the trained Ψ0 checkpoint on the GPU, receives observations, and predicts actions.

Go to the Ψ0 repository:

```bash
cd ~/Psi0
```

Set the checkpoint directory and step. Use the checkpoint that matches the task:
For example
```bash
export CHECKPOINT_DIR="$HOME/Psi0/.runs/sonic/orange-box-pickup.real.flow1000.cosine.lr1.0e-04.b64.gpus1.2607262327"
export CHECKPOINT_STEP=50000
```



Start the policy server:

```bash
bash ./scripts/deploy/serve_psi0-rtc-sonic.sh
```

Wait until the policy server is ready before starting the communication client.

Keep this terminal running.

---

## 5. Workstation — Terminal 3: SONIC Communication Client

The communication client collects the G1 camera and robot state, sends observations to the policy server, receives predicted actions, and forwards those actions to the G1 controller.

Open a third workstation terminal:

```bash
cd ~/Psi0
bash ./real/scripts/deploy_psi0-sonic-rtc-client.sh
```

Keep this terminal running during inference.

---

## 6. Stop the Deployment

### Normal Stop

To stop the robot motion after inference, go to the **SONIC Communication Client** terminal and press:

```text
Ctrl+C
```

The SONIC Communication Client will stop, and the robot will stop moving.

### Emergency Stop

For an emergency stop, go to the **SONIC Controller** terminal and press:

```text
o
```
The robot will enter damping mode.

---

## 7. Deployment Checklist

Before starting inference, verify that all processes are running:

```text
Robot:
[ ] RealSense camera server
[ ] DEX1 gripper server

Workstation:
[ ] SONIC controller -> Init done
[ ] Ψ0 policy server
[ ] SONIC communication client
```















