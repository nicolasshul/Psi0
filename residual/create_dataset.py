import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from psi.config.config import LaunchConfig
from psi.config.transform import ActionStateTransform, Psi0ModelTransform
from psi.models.psi0 import Psi0Model
from psi.utils import pad_to_len, parse_args_to_tyro_config, seed_everything


class DatasetChunkInferer:

    def __init__(
        self,
        run_dir: Path | str,
        ckpt_step: int | str = "latest",
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    ):
        self.run_dir = Path(run_dir)
        self.device = torch.device(device)

        # 1. Load run config & seeds
        config_: LaunchConfig = parse_args_to_tyro_config(self.run_dir / "argv.txt")  # type: ignore
        conf = (self.run_dir / "run_config.json").open("r").read()
        self.launch_cfg = config_.model_validate_json(conf)
        seed_everything(self.launch_cfg.seed or 42)

        # 2. Load policy
        self.model = Psi0Model.from_pretrained(
            self.run_dir, ckpt_step, self.launch_cfg, device=device
        )
        self.model.to(self.device)
        self.model.eval()

        # 3. Model transforms
        self.maxmin: ActionStateTransform = self.launch_cfg.data.transform.field  # type: ignore
        self.model_transform: Psi0ModelTransform = self.launch_cfg.data.transform.model  # type: ignore

    def _preprocess_image(self, bgr_frame: np.ndarray) -> list:
        from torchvision.transforms import v2

        # Convert OpenCV BGR frame to RGB PIL Image
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        transforms = [
            self.model_transform.resize(),
            self.model_transform.center_crop(),
        ]
        t = v2.Compose(transforms)
        return [t(pil_img)]

    def _preprocess_state(self, raw_state: list | np.ndarray) -> np.ndarray:
        states = np.asarray(raw_state, dtype=np.float32)
        if states.ndim == 2:
            states = states[-1]

        obs = states
        if (
            self.maxmin.pad_state_dim is not None
            and self.maxmin.pad_state_dim != len(obs)
        ):
            obs = pad_to_len(obs, self.maxmin.pad_state_dim, dim=0)[0]

        obs = self.maxmin.normalize_state_func(obs)
        return obs[np.newaxis, np.newaxis, :]  # Shape: (1, 1, Ds)

    @torch.no_grad()
    def predict_chunk(
        self,
        bgr_frame: np.ndarray,
        raw_state: list | np.ndarray,
        instruction: str = "",
        num_inference_steps: int = 8,
    ) -> np.ndarray:
        """Infers action chunk and applies inverse maxmin normalization."""
        processed_img = self._preprocess_image(bgr_frame)
        processed_obs = self._preprocess_state(raw_state)

        norm_actions = self.model.predict_action(
            observations=processed_img,
            states=torch.from_numpy(processed_obs).to(self.device),
            traj2ds=None,
            instructions=[instruction.lower()],
            num_inference_steps=num_inference_steps,
        )[0].float().detach().cpu().numpy()  # (H, Da)

        return self.maxmin.denormalize(norm_actions)

    def process_episode(
        self,
        video_path: Path | str,
        parquet_path: Path | str,
        output_dir: Path | str,
        instruction: str = "teleop demo",
        step_stride: int = 15,  # Stride between inferences (e.g. s_min / execution horizon)
        ctrl_hz: float = 30.0,  # Robot control frequency (1/CTRL_PERIOD_SEC)
    ):
        """
        Extracts synchronized video frames and parquet states, runs policy inference,
        and saves chunks indexed by episode timestamp and frame index.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load parquet metadata
        df = pd.read_parquet(parquet_path)
        df = df.sort_values(by="frame_index").reset_index(drop=True)

        # 2. Open MP4 video capture
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Failed to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_records = min(total_frames, len(df))

        dt = 1.0 / ctrl_hz
        print(f"Processing {num_records} aligned steps from {video_path}...")

        all_episode_chunks = []

        for frame_idx in range(num_records):
            ret, frame = cap.read()
            if not ret:
                break

            # Only run inference at step intervals (or every frame if step_stride=1)
            if frame_idx % step_stride != 0:
                continue

            row = df.iloc[frame_idx]
            base_timestamp = float(row["timestamp"])
            raw_state = row["observation.state"]

            # Predict action chunk of shape (H, Da)
            actions = self.predict_chunk(
                bgr_frame=frame,
                raw_state=raw_state,
                instruction=instruction,
            )
            horizon = actions.shape[0]

            # Calculate explicit timestamps for each predicted step in the chunk:
            # t_k = t_0 + k * dt
            chunk_timestamps = [base_timestamp + k * dt for k in range(horizon)]

            chunk_record = {
                "episode_index": int(row.get("episode_index", 0)),
                "frame_index": int(frame_idx),
                "query_timestamp": base_timestamp,
                "step_timestamps": chunk_timestamps,
                "actions": actions.tolist(),  # (H, Da)
            }
            all_episode_chunks.append(chunk_record)

            # Option A: Save individual chunk as compressed npz/json
            chunk_filename = output_dir / f"chunk_frame_{frame_idx:06d}.npz"
            np.savez_compressed(
                chunk_filename,
                actions=actions,
                query_timestamp=base_timestamp,
                step_timestamps=np.array(chunk_timestamps, dtype=np.float64),
                frame_index=frame_idx,
                episode_index=int(row.get("episode_index", 0)),
            )

        cap.release()

        # Option B: Save single consolidated episode index file
        summary_path = output_dir / "episode_chunks_manifest.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_episode_chunks, f, indent=2)

        print(f"Saved {len(all_episode_chunks)} action chunks to {output_dir}")


# -------------------------------------------------------------------------
# Usage
# -------------------------------------------------------------------------
if __name__ == "__main__":
    RUN_DIR = Path("/path/to/checkpoint_dir")
    VIDEO_PATH = Path("/path/to/episode_0.mp4")
    PARQUET_PATH = Path("/path/to/episode_0.parquet")
    OUTPUT_DIR = Path("./action_chunks/episode_0")

    inferer = DatasetChunkInferer(run_dir=RUN_DIR, ckpt_step="latest")
    inferer.process_episode(
        video_path=VIDEO_PATH,
        parquet_path=PARQUET_PATH,
        output_dir=OUTPUT_DIR,
        instruction="pick up the object",
        step_stride=5,  # Matches MIN_EXEC_HORIZON
        ctrl_hz=30.0,    # 30Hz control clock
    )