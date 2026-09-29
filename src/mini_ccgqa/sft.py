"""Fine-tune a pretrained model for summary-to-story generation.

I used substantial AI assistance to write the fine-tuning code.
I studied the training objective, selected the experimental settings,
and conducted and evaluated the final training run.
"""

import hashlib
import json
import math
import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import torch
from tokenizers import Tokenizer

from mini_ccgqa.config import ModelConfig, TrainConfig
from mini_ccgqa.model import TransformerLM
from mini_ccgqa.sft_data import PROJECT, REPO, REVISION, prepare_data
from mini_ccgqa.train import get_lr, save, write_history


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def evaluate(model, data, batch_size, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0

    for start in range(0, len(data[0]), batch_size):
        x, y = [t[start:start + batch_size].to(device) for t in data]
        _, loss = model(x, y)

        # Average over response tokens, excluding prompt and padding.
        tokens = (y != -100).sum().item()
        total_loss += loss.item() * tokens
        total_tokens += tokens

    return total_loss / total_tokens


def main(
    base_dir,
    epochs=2,
    batch_size=16,
    lr=3e-5,
    train_size=10_000,
    val_size=500,
    stop_at=None,
):
    base_dir = Path(base_dir).resolve()
    source = base_dir / "best_model.pt"
    base = torch.load(source, map_location="cpu", weights_only=True)

    tokenizer_path = PROJECT / "data/tokenizer.json"
    tokenizer_hash = fingerprint(tokenizer_path)
    if tokenizer_hash != base["data_info"]["tokenizer_sha256"]:
        raise ValueError("Use the base model's original tokenizer.")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    data, data_dir = prepare_data(tokenizer, train_size, val_size)

    batches = math.ceil(train_size / batch_size)
    total_steps = epochs * batches
    end_step = total_steps if stop_at is None else min(stop_at, total_steps)
    schedule = TrainConfig(
        max_lr=lr,
        min_lr=lr / 10,
        warmup_fraction=0.04,
    )

    settings = dict(
        base_sha256=fingerprint(source),
        tokenizer_sha256=tokenizer_hash,
        dataset=REPO,
        revision=REVISION,
        train_sha256=fingerprint(data_dir / "train.jsonl"),
        val_sha256=fingerprint(data_dir / "valid.jsonl"),
        train_size=train_size,
        val_size=val_size,
        epochs=epochs,
        batch_size=batch_size,
        total_steps=total_steps,
        max_lr=lr,
        min_lr=lr / 10,
        warmup_fraction=0.04,
        weight_decay=0.01,
        seed=42,
        eval_every=125,
        max_length=256,
        method="full_sft",
        task="summary_to_story",
    )

    output = PROJECT / "runs/sft" / f"{base_dir.name}-summary"
    output.mkdir(parents=True, exist_ok=True)
    last_path = output / "last.pt"
    best_path = output / "best_model.pt"

    if best_path.exists() and not last_path.exists():
        raise FileExistsError(
            "Resume checkpoint missing; use a new SFT folder."
        )

    torch.manual_seed(42)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    accelerator = (
        torch.cuda if device.type == "cuda" else
        torch.mps if device.type == "mps" else None
    )

    model = TransformerLM(ModelConfig(**base["config"])).to(device)
    model.load_state_dict(base["model"])

    optimizer = torch.optim.AdamW([
        {
            "params": [p for p in model.parameters() if p.ndim >= 2],
            "weight_decay": 0.01,
        },
        {
            "params": [p for p in model.parameters() if p.ndim < 2],
            "weight_decay": 0.0,
        },
    ], lr=0.0)

    num_params = sum(p.numel() for p in model.parameters())
    step, history, best_loss, response_tokens = 0, [], float("inf"), 0

    # Automatically resume an existing SFT run.
    if last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        if state["settings"] != settings:
            raise ValueError(
                "Settings changed. Resume with the original settings."
            )

        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step, history = state["step"], state["history"]
        best_loss = state["best_loss"]
        response_tokens = state["response_tokens"]

        if best_path.exists():
            best = torch.load(
                best_path, map_location="cpu", weights_only=True
            )
            best_loss = min(best_loss, best["val_loss"])

        torch.set_rng_state(state["torch_rng"])
        if accelerator is not None and state["device"] == device.type:
            accelerator.set_rng_state(state["device_rng"])

        write_history(history, output)
    else:
        metadata = dict(
            model=base["config"],
            train=settings,
            num_params=num_params,
            base_run=str(base_dir),
            base_step=base["step"],
            data_dir=str(data_dir),
            torch_version=str(torch.__version__),
        )
        (output / "metadata.json").write_text(
            json.dumps(metadata, indent=2)
        )
        (output / "tokenizer.json").write_bytes(
            tokenizer_path.read_bytes()
        )

    # Fixed training subset for monitoring; evaluate all validation examples.
    train_eval = tuple(t[:256] for t in data["train"])
    print(f"SFT | {num_params:,} parameters | {device}", flush=True)
    print(f"Starting step: {step} | target: {end_step}", flush=True)

    while True:
        measure = step == 0 or step % 125 == 0 or step == end_step

        if measure and (not history or history[-1]["step"] != step):
            train_loss = evaluate(model, train_eval, batch_size, device)
            val_loss = evaluate(model, data["valid"], batch_size, device)

            history.append(dict(
                step=step,
                response_tokens=response_tokens,
                lr=optimizer.param_groups[0]["lr"],
                train_loss=train_loss,
                val_loss=val_loss,
                train_ppl=math.exp(train_loss),
                val_ppl=math.exp(val_loss),
            ))

            print(
                f"Step {step:5d} | train loss {train_loss:.4f}, "
                f"ppl {math.exp(train_loss):.2f} | "
                f"val loss {val_loss:.4f}, ppl {math.exp(val_loss):.2f}",
                flush=True,
            )

            if val_loss < best_loss:
                best_loss = val_loss
                save(dict(
                    model=model.state_dict(),
                    config=base["config"],
                    step=step,
                    val_loss=val_loss,
                    data_info=dict(
                        tokenizer_sha256=tokenizer_hash,
                        dataset=REPO,
                        revision=REVISION,
                    ),
                ), best_path)

            save(dict(
                settings=settings,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                step=step,
                history=history,
                best_loss=best_loss,
                response_tokens=response_tokens,
                torch_rng=torch.get_rng_state(),
                device=device.type,
                device_rng=(
                    accelerator.get_rng_state() if accelerator else None
                ),
            ), last_path)

            write_history(history, output)

        if step >= end_step:
            break

        # A reproducible shuffle for each epoch, including after resuming.
        epoch, batch = divmod(step, batches)
        order = torch.randperm(
            train_size,
            generator=torch.Generator().manual_seed(42 + epoch),
        )
        indices = order[batch * batch_size:(batch + 1) * batch_size]
        x, y = [t[indices].to(device) for t in data["train"]]

        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )

        for group in optimizer.param_groups:
            group["lr"] = get_lr(step + 1, total_steps, schedule)

        optimizer.step()
        response_tokens += (y != -100).sum().item()
        step += 1

    print("SFT saved:", output, flush=True)
    return output


if __name__ == "__main__":
    base_dir = PROJECT / (
        "runs/base/2.47M-ccgqa_dynamic-tinystories-d192-ff512-l6-q8-kv2-hd12-"
        "v2048-drop0-seq256-seed42-d4c0d6d5d0f6"
    )

    main(base_dir, epochs=4, batch_size=16, lr=3e-4)