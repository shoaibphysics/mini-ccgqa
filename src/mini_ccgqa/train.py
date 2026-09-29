"""Training structure adapted from shoaibphysics/vision-transformer.

I used substantial AI assistance to adapt and write the training code.
I worked through the training logic, selected the experimental settings,
and conducted and interpreted the final benchmarks.

stop_at is the TOTAL optimizer step to reach, not additional steps.
Keep both configs unchanged when resuming; extend training with stop_at.
Beyond total_steps, the learning rate stays at min_lr.
resume=True starts fresh if this configuration has no checkpoint yet.
"""

import csv
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tokenizers import Tokenizer

from mini_ccgqa.config import ModelConfig, TrainConfig
from mini_ccgqa.data import (
    DATA_DIR,
    DATASET,
    REVISION,
    get_batch,
    load_tokens,
)
from mini_ccgqa.model import TransformerLM

def get_lr(step, total_steps, cfg):
    warmup = int(cfg.warmup_fraction * total_steps)

    if warmup > 0 and step <= warmup:
        return cfg.max_lr * step / warmup

    progress = min(
        1.0,
        (step - warmup) / (total_steps - warmup),
    )

    return cfg.min_lr + 0.5 * (cfg.max_lr - cfg.min_lr) * (
        1 + math.cos(math.pi * progress)
    )


def train_step(model, tokens, optimizer, cfg, device, generator):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for _ in range(cfg.accumulation_steps):
        x, y = get_batch(
            tokens, cfg.batch_size, cfg.seq_len, device, generator
        )
        _, loss = model(x, y)
        (loss / cfg.accumulation_steps).backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0, error_if_nonfinite=True
    )
    optimizer.step()


@torch.inference_mode()
def evaluate(model, tokens, cfg, device, seed):
    model.eval()

    # Reset a private generator: evaluate the same windows every time.
    generator = torch.Generator().manual_seed(seed)
    total = 0.0

    for _ in range(cfg.eval_batches):
        x, y = get_batch(
            tokens, cfg.batch_size, cfg.seq_len, device, generator
        )
        _, loss = model(x, y)
        total += loss.item()

    # Every batch contains the same number of target tokens.
    loss = total / cfg.eval_batches
    return loss, math.exp(loss)


def save(state, path):
    # Replace the checkpoint only after writing succeeds.
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def write_history(history, output):
    path = output / "history.csv"
    temporary = path.with_suffix(".tmp")

    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    temporary.replace(path)

    # Use this run's saved model settings.
    metadata = json.loads((output / "metadata.json").read_text())
    model = metadata["model"]

    caption = (
        f"Parameters={metadata['num_params']:,} | "
        f"d_model={model['d_model']} | d_ff={model['d_ff']} | layers={model['layers']}\n"
        f"Q/KV heads={model['q_heads']}/{model['kv_heads']} | "
        f"head_dim={model['head_dim']} | vocab={model['vocab_size']} | dropout={model['dropout']:g}"
    )

    steps = [row["step"] for row in history]

    for metric, label in (("loss", "Cross-entropy loss"), ("ppl", "Perplexity")):
        fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")

        for split, name in (("train", "Training subset"), ("val", "Validation subset")):
            ax.plot(
                steps,
                [row[f"{split}_{metric}"] for row in history],
                "o-",
                markersize=3,
                label=name,
            )

        ax.set(
            title=f"{model['attention_type'].upper()} | TinyStories",
            xlabel="Optimizer step",
            ylabel=label,
            xlim=(0, max(1, steps[-1])),
        )

        if metric == "ppl":
            ax.set_yscale("log")

        ax.legend()
        ax.grid(alpha=0.25)
        fig.supxlabel(caption, fontsize=9)
        fig.savefig(output / f"{metric}.png", dpi=150)
        plt.close(fig)

def main(
    resume=True,
    stop_at=None,
    runs_root=None,
    model_config=None,
    train_config=None,
):
    if train_config is None:
        cfg = TrainConfig()
    else:
        cfg = train_config

    assert min(
        cfg.total_steps,
        cfg.batch_size,
        cfg.seq_len,
        cfg.accumulation_steps,
        cfg.eval_batches,
    ) > 0
    assert cfg.eval_every > 0 and cfg.save_every > 0
    assert cfg.save_every % cfg.eval_every == 0
    assert 0 <= cfg.warmup_fraction < 1
    assert 0 <= cfg.min_lr <= cfg.max_lr

    torch.manual_seed(cfg.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    accelerator = (
        torch.cuda if device.type == "cuda"
        else torch.mps if device.type == "mps"
        else None
    )

    train_tokens = load_tokens("train")
    val_tokens = load_tokens("validation")

    tokenizer_path = DATA_DIR / "tokenizer.json"
    vocab_size = Tokenizer.from_file(
        str(tokenizer_path)
    ).get_vocab_size()

    config = model_config or ModelConfig(
        vocab_size=vocab_size,
        attention_type="ccgqa",
    )
    assert config.vocab_size == vocab_size
    assert min(len(train_tokens), len(val_tokens)) > cfg.seq_len

    model = TransformerLM(config).to(device)

    # Decay matrices/convolution weights.
    # Exclude biases, normalization scales and temperatures.
    decay, no_decay = [], []

    for parameter in model.parameters():
        group = decay if parameter.ndim >= 2 else no_decay
        group.append(parameter)

    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=0.0)

    end_step = cfg.total_steps if stop_at is None else stop_at
    assert end_step > 0

    tokens_per_step = (
        cfg.batch_size * cfg.seq_len * cfg.accumulation_steps
    )
    num_params = sum(p.numel() for p in model.parameters())

    settings = dict(
        version=1,
        model=asdict(config),
        train=asdict(cfg),
        data=dict(
            dataset=DATASET,
            revision=REVISION,
            train_tokens=len(train_tokens),
            validation_tokens=len(val_tokens),
            tokenizer_sha256=hashlib.sha256(
                tokenizer_path.read_bytes()
            ).hexdigest(),
        ),
    )

    # Same settings produce the same run folder, as in your ViT trainer.
    config_hash = hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode()
    ).hexdigest()[:12]

    run_name = (
        f"{num_params / 1e6:.2f}M-{config.attention_type}-tinystories"
        f"-d{config.d_model}-ff{config.d_ff}-l{config.layers}"
        f"-q{config.q_heads}-kv{config.kv_heads}-hd{config.head_dim}"
        f"-v{config.vocab_size}-drop{config.dropout:g}"
        f"-seq{cfg.seq_len}-seed{cfg.seed}-{config_hash}"
    )

    output = Path(runs_root or "runs/base") / run_name
    output.mkdir(parents=True, exist_ok=True)

    last_path = output / "last.pt"
    best_path = output / "best_model.pt"

    resume = resume and last_path.is_file()

    if not resume and (last_path.exists() or best_path.exists()):
        raise FileExistsError(
            "Saved run exists. Use resume=True or a different runs_root."
        )

    generator = torch.Generator().manual_seed(cfg.seed)
    step, history, best_loss = 0, [], float("inf")

    if resume:
        saved = torch.load(
            last_path, map_location="cpu", weights_only=True
        )
        assert saved["settings"] == settings, "Run settings changed."

        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])

        step = saved["step"]
        history = saved["history"]
        best_loss = saved["best_loss"]

        generator.set_state(saved["batch_rng"])
        torch.set_rng_state(saved["torch_rng"])

        if accelerator is not None and saved["device_type"] == device.type:
            accelerator.set_rng_state(saved["device_rng"])

        # A best-model save may be newer than the last full checkpoint.
        if best_path.exists():
            best_loss = min(
                best_loss,
                torch.load(
                    best_path, map_location="cpu", weights_only=True
                )["val_loss"],
            )

        # Restore the history corresponding to the loaded checkpoint.
        write_history(history, output)

    else:
        metadata = dict(
            **settings,
            num_params=num_params,
            tokens_per_step=tokens_per_step,
            torch_version=str(torch.__version__),
        )
        (output / "metadata.json").write_text(
            json.dumps(metadata, indent=2)
        )

    print(
        f"{num_params:,} parameters | {device} | {output}",
        flush=True,
    )
    print(
        f"Starting step: {step} | target: {end_step} | "
        f"tokens/step: {tokens_per_step:,}",
        flush=True,
    )

    while True:
        measure = (
            step == 0
            or step % cfg.eval_every == 0
            or step == end_step
        )

        if measure and (not history or history[-1]["step"] != step):
            train_loss, train_ppl = evaluate(
                model, train_tokens, cfg, device, cfg.seed + 1
            )
            val_loss, val_ppl = evaluate(
                model, val_tokens, cfg, device, cfg.seed + 2
            )

            history.append(dict(
                step=step,
                tokens_seen=step * tokens_per_step,
                lr=optimizer.param_groups[0]["lr"],
                train_loss=train_loss,
                train_ppl=train_ppl,
                val_loss=val_loss,
                val_ppl=val_ppl,
            ))

            print(
                f"Step {step:5d} | "
                f"train loss {train_loss:.4f}, ppl {train_ppl:.3e} | "
                f"val loss {val_loss:.4f}, ppl {val_ppl:.3e}",
                flush=True,
            )

            if val_loss < best_loss:
                best_loss = val_loss

                save(dict(
                    model=model.state_dict(),
                    config=asdict(config),
                    data_info=settings["data"],
                    step=step,
                    val_loss=val_loss,
                ), best_path)

            if step % cfg.save_every == 0 or step == end_step:
                save(dict(
                    settings=settings,
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    step=step,
                    history=history,
                    best_loss=best_loss,
                    batch_rng=generator.get_state(),
                    torch_rng=torch.get_rng_state(),
                    device_type=device.type,
                    device_rng=(
                        accelerator.get_rng_state()
                        if accelerator is not None else None
                    ),
                ), last_path)

            write_history(history, output)

        if step >= end_step:
            break

        for group in optimizer.param_groups:
            group["lr"] = get_lr(step + 1, cfg.total_steps, cfg)

        train_step(
            model, train_tokens, optimizer, cfg, device, generator
        )
        step += 1

    print("Results:", output)
    return output


if __name__ == "__main__":
    # One training setup shared by all four models.
    train_cfg = TrainConfig(
        total_steps=12_000,
        batch_size=16,
        seq_len=256,
        accumulation_steps=1,
        seed=42,
        eval_every=250,
        save_every=250,
        eval_batches=16,
        max_lr=3e-4,
        min_lr=3e-5,
        warmup_fraction=0.02,
        weight_decay=0.05)

    # Name, attention implementation, features per head.
    experiments = [
        ("Full GQA", "gqa", 24),
        ("Naively compressed GQA", "gqa", 12),
        ("CCGQA", "ccgqa", 12),
        ("Dynamic CCGQA", "ccgqa_dynamic", 12)]

    # Train sequentially: finish one model before starting the next.
    for name, attention_type, head_dim in experiments:
        model_cfg = ModelConfig(
            vocab_size=2048,
            d_model=192,
            d_ff=512,
            layers=6,
            attention_type=attention_type,
            q_heads=8,
            kv_heads=2,
            head_dim=head_dim,
            rope_theta=10_000.0,
            norm_eps=1e-5,
            dropout=0.0)

        print(f"\nExperiment: {name}", flush=True)

        main(
            model_config=model_cfg,
            train_config=train_cfg,
            runs_root="runs/base",
            resume=True,
            stop_at=None,  # Reach train_cfg.total_steps.
        )