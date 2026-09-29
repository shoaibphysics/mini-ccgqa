"""Plot and compare the training results of the attention variants.

I specified the comparisons and used substantial AI assistance
to write the plotting code. I checked the metric definitions
and interpreted the results.
"""

import csv
import json
import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


# These prefixes identify our four completed runs.
RUNS = {
    "Full GQA": "2.72M-gqa-*",
    "Naive compression": "2.44M-gqa-*",
    "CCGQA": "2.46M-ccgqa-*",
    "Dynamic CCGQA": "2.47M-ccgqa_dynamic-*",
}
COLORS = ["tab:blue", "tab:orange", "teal", "tab:purple"]
STYLES = ["-", "--", "-.", "-"]


def save_figure(fig, title, filename, output, caption):
    fig.suptitle(title, fontsize=15)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="upper center",
            bbox_to_anchor=(0.5, 0.91), ncol=4, frameon=False
        )

    for ax in fig.axes:
        ax.grid(alpha=0.25)
        ax.set_xlabel("Optimizer step")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.xaxis.set_major_formatter("{x:,.0f}")

    fig.text(0.5, 0.025, caption, ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0.12, 1, 0.90))

    for extension in ("png", "svg"):
        fig.savefig(output / f"{filename}.{extension}", dpi=200)
    plt.close(fig)


def main(runs_root="runs/base", output_dir="assets/comparison", zoom_start=1000):
    histories, metadata = {}, {}

    for name, pattern in RUNS.items():
        folders = [p for p in Path(runs_root).glob(pattern) if p.is_dir()]
        if len(folders) != 1:
            raise ValueError(
                f"Expected one run for {name}; found {len(folders)}. Check RUNS."
            )

        folder = folders[0]
        metadata[name] = json.loads((folder / "metadata.json").read_text())
        rows = np.atleast_1d(np.genfromtxt(
            folder / "history.csv", delimiter=",", names=True
        ))
        h = {column: rows[column] for column in rows.dtype.names}
        h["train_ppl"] = np.exp(h["train_loss"])
        h["val_ppl"] = np.exp(h["val_loss"])
        histories[name] = h

    full = histories["Full GQA"]
    reference = metadata["Full GQA"]
    steps = full["step"]

    # Compare measurements from the same steps and training setup.
    for name, h in histories.items():
        np.testing.assert_array_equal(h["step"], steps)
        np.testing.assert_array_equal(h["tokens_seen"], full["tokens_seen"])
        if (metadata[name]["train"] != reference["train"]
                or metadata[name]["data"] != reference["data"]):
            raise ValueError("Use runs with the same training settings and data.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    m = reference["model"]
    compressed_dim = metadata["CCGQA"]["model"]["head_dim"]
    caption = "Parameters: " + " | ".join(
        f"{name}={metadata[name]['num_params']:,}" for name in RUNS
    )
    caption += (
        f"\nd_model={m['d_model']} | d_ff={m['d_ff']} | layers={m['layers']} | "
        f"Q/KV heads={m['q_heads']}/{m['kv_heads']} | "
        f"head_dim={m['head_dim']} (full), {compressed_dim} (compressed) | "
        f"vocab={m['vocab_size']} | dropout={m['dropout']:g}"
    )
    title = f"TinyStories | seed {reference['train']['seed']}"

    # 1–2. Loss and perplexity, including a later-training view.
    for metric, label, filename in [
        ("val_loss", "Validation loss (nats/token)", "01_validation_loss"),
        ("val_ppl", "Validation perplexity", "02_validation_perplexity"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

        for ax, start in zip(axes, [0, zoom_start]):
            mask = steps >= start
            for (name, h), color, style in zip(
                histories.items(), COLORS, STYLES
            ):
                ax.plot(
                    steps[mask], h[metric][mask], label=name,
                    color=color, linestyle=style, linewidth=2
                )
            ax.set_ylabel(label)
            ax.set_title(
                "Complete run" if start == 0 else f"From step {start:,}"
            )

        if metric == "val_ppl":
            axes[0].set_yscale("log")
            axes[0].set_ylabel("Validation perplexity (log scale)")

        save_figure(fig, f"{title} | {label}", filename, output, caption)

    # 3. Dynamic CCGQA compared with CCGQA.
    cc = histories["CCGQA"]
    dynamic = histories["Dynamic CCGQA"]

    differences = [
        (
            cc["val_loss"] - dynamic["val_loss"],
            "Loss reduction (nats/token)"
        ),
        (
            cc["val_ppl"] - dynamic["val_ppl"],
            "Perplexity reduction"
        ),
        (
            100 * (1 - dynamic["val_ppl"] / cc["val_ppl"]),
            "Perplexity reduction (%)"
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5))
    for ax, (values, label) in zip(axes, differences):
        ax.plot(steps, values, color="tab:purple", linewidth=2)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.fill_between(steps, 0, values, color="tab:purple", alpha=0.1)
        ax.set_ylabel(label)
        ax.set_title(f"Final: {values[-1]:.4f}")

    save_figure(
        fig, f"{title} | Dynamic vs CCGQA: positive is better",
        "03_dynamic_advantage", output, caption
    )

    # 4 and 6. Separate comparisons against full and naive GQA.
    for baseline, candidates, filename in [
        (
            "Full GQA",
            ["Naive compression", "CCGQA", "Dynamic CCGQA"],
            "04_relative_to_full_gqa"
        ),
        (
            "Naive compression",
            ["CCGQA", "Dynamic CCGQA"],
            "06_relative_to_naive_compression"
        ),
    ]:
        fig, ax = plt.subplots(figsize=(13, 5.5))
        baseline_ppl = histories[baseline]["val_ppl"]

        for (name, h), color, style in zip(
            histories.items(), COLORS, STYLES
        ):
            if name not in candidates:
                continue

            gain = 100 * (1 - h["val_ppl"] / baseline_ppl)
            ax.plot(
                steps, gain, color=color, linestyle=style, linewidth=2,
                label=f"{name}: {gain[-1]:+.2f}% at final step"
            )

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(f"Perplexity reduction relative to {baseline} (%)")

        save_figure(
            fig, f"{title} | Relative to {baseline}: positive is better",
            filename, output, caption
        )

    # 5. Training–validation gap.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for (name, h), color, style in zip(histories.items(), COLORS, STYLES):
        gap = h["val_loss"] - h["train_loss"]
        axes[0].plot(
            steps, gap, label=name, color=color,
            linestyle=style, linewidth=2
        )
        axes[1].plot(
            steps, np.exp(gap), color=color,
            linestyle=style, linewidth=2
        )

    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[1].axhline(1, color="black", linewidth=0.8)
    axes[0].set_ylabel("Validation loss − training loss (nats/token)")
    axes[1].set_ylabel("Validation perplexity / training perplexity")
    axes[0].set_title("Loss gap")
    axes[1].set_title("Equivalent perplexity ratio")

    save_figure(
        fig, f"{title} | Training–validation gap",
        "05_validation_gap", output, caption
    )

    # Final metrics, all at the same optimizer step.
    with (output / "final_metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "parameters", "step", "tokens_seen",
            "train_loss", "val_loss", "train_ppl", "val_ppl"
        ])

        for name, h in histories.items():
            writer.writerow([
                name, metadata[name]["num_params"], int(steps[-1]),
                int(h["tokens_seen"][-1]),
                h["train_loss"][-1], h["val_loss"][-1],
                h["train_ppl"][-1], h["val_ppl"][-1]
            ])

    pairs = [(name, "Full GQA") for name in RUNS if name != "Full GQA"]
    pairs += [
        ("CCGQA", "Naive compression"),
        ("Dynamic CCGQA", "Naive compression"),
        ("Dynamic CCGQA", "CCGQA"),
    ]

    with (output / "final_comparisons.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "reference", "step", "loss_reduction",
            "ppl_reduction", "relative_ppl_reduction",
            "ppl_reduction_percent"
        ])

        for name, baseline in pairs:
            h, b = histories[name], histories[baseline]
            relative = 1 - h["val_ppl"][-1] / b["val_ppl"][-1]
            writer.writerow([
                name, baseline, int(steps[-1]),
                b["val_loss"][-1] - h["val_loss"][-1],
                b["val_ppl"][-1] - h["val_ppl"][-1],
                relative, 100 * relative
            ])
            print(
                f"{name} vs {baseline}: "
                f"{100 * relative:+.2f}% perplexity reduction"
            )

    print(f"Saved plots and tables: {output.resolve()}")


if __name__ == "__main__":
    main()