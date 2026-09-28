# SONIC to Ψ0 Fine-Tuning Guide


Set the common paths used below:

```bash
cd /home/liyan/Psi0
export PSI_HOME=/home/liyan/Psi0
export task=merged_orange
```

## 1. Merge SONIC recording sessions

Activate the data-collection environment and run the processor from the SONIC repository:

```bash
cd "$PSI_HOME/third_party/GR00T-WholeBodyControl"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/process_dataset.py \
  --dataset-path \
    outputs/2026-07-24-16-59-38 \
    outputs/2026-07-25-15-12-42 \
    outputs/2026-07-25-16-22-27 \
    outputs/2026-07-24-17-08-05 \
    outputs/2026-07-25-15-25-17 \
    outputs/2026-07-25-16-34-58 \
    outputs/2026-07-25-14-52-14 \
    outputs/2026-07-25-14-52-14 \
    outputs/2026-07-25-15-42-38 \
  --output-path outputs/merged_orange

```

The processor removes discarded episodes and stale SMPL frames by default. It does not modify the source datasets.



## 2. Set the training prompt

Update the prompt in both source metadata files:

```text
third_party/GR00T-WholeBodyControl/outputs/merged_2026-07-15_to_2026-07-17/meta/tasks.jsonl
third_party/GR00T-WholeBodyControl/outputs/merged_2026-07-15_to_2026-07-17/meta/episodes.jsonl
```


```json
{"task_index": 0, "task": "Walk forward, pick up the box with both hands, step back, turn left, and place the box on the table."}
```


## 3. Convert SONIC data to the Ψ0 LeRobot schema

The merged SONIC dataset is already LeRobot v2.1, but Ψ0 requires different state, action, and camera fields. Convert the merged dataset once:

```bash
cd "$PSI_HOME"

python scripts/data/raw_sonic_to_psi_lerobot.py \
  --data-root="$PSI_HOME/third_party/GR00T-WholeBodyControl/outputs/$task" \
  --work-dir="$PSI_HOME/data/sonic/lerobot" \
  --repo-id="$task" \
  --robot-type=g1
```

Converted output:

```text
/home/liyan/Psi0/data/sonic/lerobot/merged_2026-07-15_to_2026-07-17
```

## 4. Calculate Ψ0 dataset statistics

```bash
python scripts/data/calc_modality_stats.py \
  --task-dir="$PSI_HOME/data/sonic/lerobot/$task"

cp "$PSI_HOME/data/sonic/lerobot/$task/meta/stats.json" \
   "$PSI_HOME/data/sonic/lerobot/$task/meta/stats_psi0.json"
```

The copy is required because the training configuration expects `meta/stats_psi0.json`.

## 5. Verify the converted dataset

```bash
DATASET="$PSI_HOME/data/sonic/lerobot/$task"

jq '{total_episodes, total_frames, total_tasks}' "$DATASET/meta/info.json"
jq -s 'map(.task) | unique' "$DATASET/meta/tasks.jsonl"
jq -s 'map(.instruction) | unique' "$DATASET/meta/episodes.jsonl"
ls -lh "$DATASET/meta/stats.json" "$DATASET/meta/stats_psi0.json"
```

Expected counts:

```text
total_episodes: 34
total_frames: 30816
total_tasks: 1
```

## 6. Configure Weights & Biases (W&B)(optional)

W&B is used to monitor training and validation loss. Configure it in the Ψ0 training environment on the new computer:

```bash
cd "$PSI_HOME"
source .venv-psi/bin/activate

pip install wandb
wandb login
```

When prompted, paste the API key from the W&B account settings page. Do not put the API key in this guide or commit it to Git.

Create a local `.env` file in the Ψ0 repository root, and make sure it is ignored by Git:

```bash
export WANDB_MODE=online
export WANDB_PROJECT=sonic-psi0
export WANDB_ENTITY=<your-wandb-username-or-team>
export WANDB_NAME=orange-box-pickup
```

Load the settings before starting training:

```bash
source .env
wandb login --verify
```

The training configuration must report metrics to W&B. Check the Ψ0 repository if necessary:

```bash
rg "wandb|report_to|WANDB" scripts configs
```

For a computer without network access, use offline mode:

```bash
export WANDB_MODE=offline
```

After the computer is connected to the internet, synchronize the offline run:

```bash
wandb sync <offline-run-directory>
```


## 7. Check the training prerequisites



## 8. Fine-tune Ψ0

Run from the Ψ0 repository root.

One GPU:

```bash
cd "$PSI_HOME"

CUDA_VISIBLE_DEVICES=0 \
bash scripts/train/psi0/finetune-real-sonic-psi0.sh \
  "$task" \
  orange-box-pickup
```


The first argument is the dataset directory name. The second is the experiment name; it does not change the language instruction.



If training runs out of GPU memory, reduce this setting in `scripts/train/psi0/finetune-real-sonic-psi0.sh`:

```bash
--train.train_batch_size=64
```

Try `8`, then `4` or `2` if necessary. Do not change gradient accumulation unless you intentionally want to change the effective batch size.

## 9. Monitor and select a checkpoint

Monitor training and validation loss in the terminal or W&B. Prefer a checkpoint with good validation loss rather than automatically assuming the final checkpoint is best.

The training log prints the exact run directory as:

```text
Accelerator runs in: <run-directory>
```

Record that directory and the selected checkpoint step for deployment.




