"""Compare base and SFT models using validation metrics and generated text.

I used substantial AI assistance to write the evaluation and plotting code.
I worked through the evaluation logic, checked the comparison settings,
and ran and interpreted the final results.
"""


import csv
import hashlib
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from mini_ccgqa.generate import load_model, generate_text


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def evaluate(model, tokenizer, examples, batch_size=16):
    """Score story tokens, including EOS, with the true previous tokens."""
    model.eval()
    eos = tokenizer.token_to_id("<|endoftext|>")
    total_loss, total_tokens, total_correct = 0.0, 0, 0

    for start in range(0, len(examples), batch_size):
        inputs, labels = [], []

        for example in examples[start:start + batch_size]:
            prompt = tokenizer.encode(example["prompt"]).ids
            answer = tokenizer.encode(example["response"]).ids + [eos]
            padding = 256 - len(prompt) - len(answer)

            if padding < 0:
                raise ValueError("A validation example exceeds 256 tokens.")

            inputs.append((prompt + answer + [eos] * padding)[:-1])
            labels.append(([-100] * len(prompt) + answer + [-100] * padding)[1:])

        x = torch.tensor(inputs, device=model.device)
        y = torch.tensor(labels, device=model.device)
        logits = model(x).logits

        total_loss += F.cross_entropy(
            logits.transpose(1, 2),
            y,
            ignore_index=-100,
            reduction="sum").item()

        mask = y != -100
        total_tokens += mask.sum().item()
        total_correct += ((logits.argmax(-1) == y) & mask).sum().item()

    loss = total_loss / total_tokens

    return dict(
        loss=loss,
        perplexity=math.exp(loss),
        token_accuracy_pct=100 * total_correct / total_tokens,
        response_tokens=total_tokens)


def make_plots(history, results, metadata, output):
    cfg = metadata["model"]
    model_name = {
        "gqa": "GQA",
        "ccgqa": "CCGQA",
        "ccgqa_dynamic": "Dynamic CCGQA"}[cfg["attention_type"]]

    caption = (
        f"Parameters={metadata['num_params']:,} | "
        f"d_model={cfg['d_model']} | "
        f"d_ff={cfg['d_ff']} | layers={cfg['layers']}\n"
        f"Q/KV heads={cfg['q_heads']}/{cfg['kv_heads']} | "
        f"head_dim={cfg['head_dim']} | vocab={cfg['vocab_size']} | "
        f"dropout={cfg['dropout']:g}")

    def save(fig, filename):
        fig.text(0.5, 0.03, caption, ha="center", fontsize=9)
        fig.subplots_adjust(bottom=0.28, top=0.82, wspace=0.35)

        for extension in ("png", "svg"):
            fig.savefig(output / f"{filename}.{extension}", dpi=160)

        plt.close(fig)

    # Training history: response-only loss and perplexity.
    steps = [float(row["step"]) for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.suptitle(f"{model_name} | TinyStories SFT: learning curves")

    for ax, key, label in zip(axes, ("loss", "ppl"), ("Loss", "Perplexity")):
        for split in ("train", "val"):
            values = [float(row[f"{split}_{key}"]) for row in history]
            ax.plot(steps, values, label=split)

        ax.set(xlabel="Optimizer step", ylabel=label)
        ax.grid(alpha=0.2)
        ax.legend()

    save(fig, "01_sft_learning_curves")

    # Evaluate Base and SFT on identical validation examples.
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.8))
    fig.suptitle(f"{model_name} | TinyStories: Base versus SFT")

    metrics = (
        ("loss", "Response loss (lower is better)"),
        ("perplexity", "Perplexity (lower is better)"),
        ("token_accuracy_pct", "Next-token accuracy (%)"))

    for ax, (key, title) in zip(axes, metrics):
        values = [row[key] for row in results]
        bars = ax.bar(
            [row["model"] for row in results],
            values,
            color=["steelblue", "seagreen"])
        ax.bar_label(bars, fmt="%.3f", padding=4)
        ax.set(title=title, ylim=(0, max(values) * 1.2))

    save(fig, "02_base_vs_sft")


def save_samples(models, examples, output, num_prompts):
    selected = random.Random(42).sample(examples, num_prompts)

    with (
        (output / "samples.md").open("w") as report,
        (output / "outputs.jsonl").open("w") as records):
    

        for number, example in enumerate(selected, 1):
            for decoding, temperature in (
                ("greedy", 0.0),
                ("top_p", 0.8)):
                seed = None if temperature == 0 else 42 + number

                for stage, (model, tokenizer) in models.items():
                    sample_id = f"{number:02d}-{stage}-{decoding}"
                    prompt = example["prompt"]
                    available = 256 - len(tokenizer.encode(prompt).ids)

                    print(f"\nExample {sample_id}", flush=True)

                    text = generate_text(
                        model,
                        tokenizer,
                        prompt=prompt,
                        max_new_tokens=available,
                        temperature=temperature,
                        top_p=0.9,
                        top_k=0,
                        seed=seed)

                    records.write(json.dumps(dict(
                        id=sample_id,
                        model=stage,
                        decoding=decoding,
                        temperature=temperature,
                        top_p=0.9 if temperature else None,
                        top_k=0,
                        seed=seed,
                        max_new_tokens=available,
                        prompt=prompt,
                        text=text)) + "\n")

                    report.write(f"## {sample_id}\n\n{text}\n\n---\n\n")

                    for file in (report, records):
                        file.flush()


def main(num_prompts=0, project=None, output_root=None):
    project = (
        Path(project) if project
        else Path(__file__).resolve().parents[2]
    )
    name = (
        "2.47M-ccgqa_dynamic-tinystories-d192-ff512-l6-q8-kv2-hd12-"
        "v2048-drop0-seq256-seed42-d4c0d6d5d0f6"
    )

    runs = {
        "Base": project / "runs/base" / name,
        "SFT": project / "runs/sft" / f"{name}-summary",
    }

    metadata = json.loads(
        (runs["SFT"] / "metadata.json").read_text()
    )
    validation = project / "data/sft_summary_hf/valid.jsonl"

    # Match the data, tokenizer and starting checkpoint of this SFT run.
    for path, key in (
        (validation, "val_sha256"),
        (project / "data/tokenizer.json", "tokenizer_sha256"),
        (runs["Base"] / "best_model.pt", "base_sha256"),
    ):
        if fingerprint(path) != metadata["train"][key]:
            raise ValueError(
                f"File does not match this SFT run: {path}"
            )

    examples = [
        json.loads(line)
        for line in validation.read_text().splitlines()
    ]

    if not 0 <= num_prompts <= len(examples):
        raise ValueError(
            "num_prompts must be between 0 and the validation size."
        )

    root = (
        Path(output_root) if output_root
        else runs["SFT"] / "evaluation"
    )
    output = root / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=False)

    print(
        f"Evaluating {len(examples)} validation stories. "
        f"Saving to {output}",
        flush=True,
    )

    models, results = {}, []

    for stage, path in runs.items():
        models[stage] = load_model(path)
        print(f"Scoring {stage}...", flush=True)
        results.append(dict(
            model=stage,
            **evaluate(*models[stage], examples),
        ))

    base, sft = results

    for row in results:
        row["ppl_reduction_pct"] = 100 * (
            1 - row["perplexity"] / base["perplexity"]
        )

    with (output / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(base))
        writer.writeheader()
        writer.writerows(results)

    table = (
        "| Model | Response loss | Perplexity | Token accuracy | "
        "PPL reduction vs Base |\n"
        "|---|---:|---:|---:|---:|\n"
    )

    for row in results:
        table += (
            f"| {row['model']} | {row['loss']:.4f} | "
            f"{row['perplexity']:.3f} | "
            f"{row['token_accuracy_pct']:.2f}% | "
            f"{row['ppl_reduction_pct']:.2f}% |\n"
        )

    table += (
        f"\nLoss reduction: {base['loss'] - sft['loss']:.4f} "
        "nats/token.\n"
        f"Token-accuracy change: "
        f"{sft['token_accuracy_pct'] - base['token_accuracy_pct']:+.2f} "
        "percentage points.\n"
        "\nMetrics use the true preceding story tokens; prompts and "
        "padding are excluded. They measure next-token prediction, "
        "not generated-story quality or summary adherence.\n"
    )

    print("\n" + table, flush=True)
    (output / "summary.md").write_text(table)

    # Record exactly which files were evaluated.
    (output / "evaluation.json").write_text(json.dumps(dict(
        examples=len(examples),
        validation=str(validation),
        val_sha256=fingerprint(validation),
        num_generation_prompts=num_prompts,
        tokenizer_sha256=metadata["train"]["tokenizer_sha256"],
        checkpoints={
            stage: dict(
                path=str(path / "best_model.pt"),
                sha256=fingerprint(path / "best_model.pt"),
            )
            for stage, path in runs.items()
        },
    ), indent=2))

    with (runs["SFT"] / "history.csv").open() as file:
        history = list(csv.DictReader(file))

    make_plots(history, results, metadata, output)

    if num_prompts:
        save_samples(models, examples, output, num_prompts)

    print("Comparison saved:", output, flush=True)


if __name__ == "__main__":
    main(num_prompts=20)  # Use 20 to also generate 80 paired stories.