
#!/usr/bin/env python3
"""
Validate Psi0 predicted action chunks.

Checks:
  1. JSON manifest structure and chunk count
  2. Action dimensions, horizons, NaNs, and infinities
  3. Timestamp ordering and expected 30 Hz spacing
  4. JSON vs. NPZ consistency
  5. Query timestamp spacing and overlapping prediction consistency
  6. Action plots and a summary report

Usage:
    python3 validate_action_chunks.py
    python3 validate_action_chunks.py --input ./action_chunks/episode_0
    python3 validate_action_chunks.py --input ./action_chunks/episode_0 --fps 30
"""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def load_actions(chunk):
    """Support both plain action arrays and timestamped action objects."""
    items = chunk.get("actions", [])

    if not items:
        return np.empty((0, 0)), np.array([])

    if isinstance(items[0], dict):
        actions = np.asarray(
            [item["action"] for item in items], dtype=float
        )
        timestamps = np.asarray(
            [item["timestamp"] for item in items], dtype=float
        )
    else:
        actions = np.asarray(items, dtype=float)
        timestamps = np.asarray(
            chunk.get(
                "action_timestamps",
                chunk.get("step_timestamps", []),
            ),
            dtype=float,
        )

    return actions, timestamps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("./action_chunks/episode_0"),
        help="Directory containing the manifest and NPZ files",
    )
    parser.add_argument(
        "--fps", type=float, default=30.0,
        help="Expected action frequency in Hz",
    )
    parser.add_argument(
        "--plot-dims", type=int, default=12,
        help="Maximum action dimensions to plot",
    )
    parser.add_argument(
        "--overlap-tol", type=float, default=None,
        help="Optional maximum acceptable mean absolute overlap "
             "difference, in action units",
    )
    args = parser.parse_args()

    manifest = args.input / "episode_chunks_manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    with manifest.open() as f:
        chunks = json.load(f)

    errors = []
    warnings = []
    records = []
    print(f"\nValidating: {args.input}")
    print(f"Chunks in manifest: {len(chunks)}")

    # ---------- Validate individual chunks ----------
    for i, chunk in enumerate(chunks):
        frame = chunk.get("frame_index", i)
        query_t = float(chunk.get("query_timestamp", np.nan))
        actions, timestamps = load_actions(chunk)

        if actions.ndim != 2:
            errors.append(
                f"Chunk {i} (frame {frame}): expected 2-D actions, "
                f"got shape {actions.shape}"
            )
            continue

        horizon, action_dim = actions.shape

        if horizon == 0 or action_dim == 0:
            errors.append(f"Chunk {i}: empty actions")
            continue

        if not np.isfinite(query_t):
            errors.append(f"Chunk {i}: invalid query timestamp")

        if not np.isfinite(actions).all():
            errors.append(f"Chunk {i}: actions contain NaN or Inf")

        if timestamps.size != horizon:
            errors.append(
                f"Chunk {i}: {horizon} actions but "
                f"{timestamps.size} timestamps"
            )
        elif not np.isfinite(timestamps).all():
            errors.append(f"Chunk {i}: timestamps contain NaN or Inf")
        elif timestamps.size > 1:
            gaps = np.diff(timestamps)

            if np.any(gaps <= 0):
                errors.append(
                    f"Chunk {i}: action timestamps are not increasing"
                )

            if not np.allclose(
                gaps, 1.0 / args.fps, atol=1e-4, rtol=0
            ):
                warnings.append(
                    f"Chunk {i}: timestamp gaps differ from "
                    f"expected {1.0 / args.fps:.6f}s"
                )

            if abs(timestamps[0] - query_t) > 0.05:
                warnings.append(
                    f"Chunk {i}: first action timestamp "
                    f"({timestamps[0]:.3f}) differs from query "
                    f"timestamp ({query_t:.3f})"
                )

        # Compare JSON data with saved NPZ data.
        npz_path = args.input / f"chunk_frame_{int(frame):06d}.npz"

        if not npz_path.exists():
            warnings.append(f"Missing NPZ file: {npz_path.name}")
        else:
            try:
                with np.load(npz_path) as data:
                    if "actions" not in data:
                        errors.append(
                            f"{npz_path.name}: missing actions array"
                        )
                    else:
                        npz_actions = data["actions"]

                        if npz_actions.shape != actions.shape:
                            errors.append(
                                f"{npz_path.name}: JSON/NPZ shape "
                                f"mismatch: {actions.shape} vs "
                                f"{npz_actions.shape}"
                            )
                        elif not np.allclose(
                            npz_actions, actions,
                            rtol=1e-5, atol=1e-6,
                            equal_nan=True,
                        ):
                            errors.append(
                                f"{npz_path.name}: JSON/NPZ action "
                                f"values differ"
                            )

                    if timestamps.size and "step_timestamps" in data:
                        npz_times = data["step_timestamps"]
                        if (
                            npz_times.shape != timestamps.shape
                            or not np.allclose(
                                npz_times, timestamps,
                                rtol=1e-5, atol=1e-6,
                            )
                        ):
                            warnings.append(
                                f"{npz_path.name}: timestamps differ "
                                "between JSON and NPZ"
                            )
            except Exception as exc:
                errors.append(
                    f"Could not read {npz_path.name}: {exc}"
                )

        records.append({
            "index": i,
            "frame": int(frame),
            "query_t": query_t,
            "timestamps": timestamps,
            "actions": actions,
            "horizon": horizon,
            "action_dim": action_dim,
        })

    # ---------- Validate ordering across chunks ----------
    records.sort(key=lambda r: r["query_t"])

    query_times = np.asarray([r["query_t"] for r in records])

    if len(query_times) > 1:
        query_gaps = np.diff(query_times)

        if np.any(query_gaps <= 0):
            warnings.append(
                "Query timestamps are duplicated or not increasing"
            )

        print(
            f"Query time range: {query_times[0]:.3f}s - "
            f"{query_times[-1]:.3f}s"
        )
        print(
            f"Query gap min/median/max: "
            f"{query_gaps.min():.4f} / "
            f"{np.median(query_gaps):.4f} / "
            f"{query_gaps.max():.4f}s"
        )
    elif len(query_times) == 1:
        print(f"Only query timestamp: {query_times[0]:.3f}s")

    horizons = sorted(set(r["horizon"] for r in records))
    dimensions = sorted(set(r["action_dim"] for r in records))

    print(f"Unique action horizons: {horizons}")
    print(f"Unique action dimensions: {dimensions}")

    # ---------- Analyze overlapping predictions ----------
    # Compare actions assigned to the same absolute timestamp.
    # This assumes timestamps refer to the same action/control clock.
    overlap_diffs = []

    for i in range(len(records)):
        a = records[i]
        ta = a["timestamps"]
        xa = a["actions"]

        if len(ta) != len(xa) or len(ta) == 0:
            continue

        for j in range(i + 1, len(records)):
            b = records[j]
            tb = b["timestamps"]
            xb = b["actions"]

            if len(tb) != len(xb) or len(tb) == 0:
                continue

            if xa.shape[1] != xb.shape[1]:
                continue

            # Match timestamps within half a timestep.
            tol = 0.5 / args.fps
            matches = []

            for ia, t in enumerate(ta):
                ib = int(np.argmin(np.abs(tb - t)))
                if abs(tb[ib] - t) <= tol:
                    matches.append((ia, ib))

            if matches:
                diffs = [
                    np.mean(np.abs(xa[ia] - xb[ib]))
                    for ia, ib in matches
                ]
                overlap_diffs.extend(diffs)

    if overlap_diffs:
        print(
            "Overlapping predictions mean absolute difference: "
            f"mean={np.mean(overlap_diffs):.6g}, "
            f"median={np.median(overlap_diffs):.6g}, "
            f"max={np.max(overlap_diffs):.6g}"
        )

        if args.overlap_tol is not None:
            if np.mean(overlap_diffs) > args.overlap_tol:
                warnings.append(
                    "Mean overlapping prediction difference exceeds "
                    f"tolerance {args.overlap_tol}"
                )
    else:
        print(
            "No overlapping predictions compared. Check timestamp "
            "alignment and chunk formats."
        )

    # ---------- Action statistics ----------
    valid_actions = [
        r["actions"] for r in records
        if r["actions"].ndim == 2 and r["actions"].size
    ]

    if valid_actions:
        all_actions = np.concatenate(valid_actions, axis=0)

        print("\nAction statistics (predictions include overlap):")
        print(f"Total predicted action vectors: {len(all_actions)}")
        print(f"Action dimension: {all_actions.shape[1]}")

        for d in range(all_actions.shape[1]):
            values = all_actions[:, d]
            if np.isfinite(values).any():
                finite = values[np.isfinite(values)]
                print(
                    f"  dim {d:02d}: min={finite.min():.5g}, "
                    f"max={finite.max():.5g}, "
                    f"mean={finite.mean():.5g}, "
                    f"std={finite.std():.5g}"
                )

    # ---------- Plots ----------
    if plt is not None and records:
        plot_dir = args.input / "validation_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Plot the middle chunk.
        middle = records[len(records) // 2]
        actions = middle["actions"]
        timestamps = middle["timestamps"]

        if len(timestamps) == len(actions) and len(actions):
            n_dims = min(actions.shape[1], args.plot_dims)
            fig, axes = plt.subplots(
                n_dims, 1,
                figsize=(12, max(3, 2 * n_dims)),
                sharex=True, squeeze=False,
            )

            for d in range(n_dims):
                axes[d, 0].plot(timestamps, actions[:, d])
                axes[d, 0].set_ylabel(f"Dim {d}")
                axes[d, 0].grid(True, alpha=0.3)

            axes[-1, 0].set_xlabel("Timestamp (seconds)")
            fig.suptitle(
                f"Middle chunk: frame {middle['frame']}, "
                f"query t={middle['query_t']:.3f}s"
            )
            fig.tight_layout()
            fig.savefig(plot_dir / "middle_chunk.png", dpi=150)
            plt.close(fig)

        # Plot query times and their spacing.
        if len(query_times) > 1:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(np.arange(len(query_times)), query_times, ".-")
            ax.set_xlabel("Chunk index (sorted by query timestamp)")
            ax.set_ylabel("Query timestamp (seconds)")
            ax.set_title("Chunk query timestamps")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plot_dir / "query_timestamps.png", dpi=150)
            plt.close(fig)

        print(f"\nPlots saved to: {plot_dir}")
    elif plt is None:
        print("\nInstall matplotlib to generate plots: pip install matplotlib")

    # ---------- Final report ----------
    print("\n========== VALIDATION REPORT ==========")
    print(f"Chunks checked: {len(records)}")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")

    if errors:
        print("\nERRORS:")
        for msg in errors:
            print("  [ERROR]", msg)

    if warnings:
        print("\nWARNINGS (first 30):")
        for msg in warnings[:30]:
            print("  [WARN]", msg)
        if len(warnings) > 30:
            print(f"  ... and {len(warnings) - 30} more")

    if not errors and not warnings:
        print("All structural and timestamp checks passed.")
    elif not errors:
        print("No structural errors found, but review the warnings.")

    print(
        "\nNote: passing these checks does not establish that actions "
        "are physically valid or match the demonstration."
    )


if __name__ == "__main__":
    main()
