"""Prepare summary-to-story examples for supervised fine-tuning.

I used substantial AI assistance to write the data preparation code.
I worked through the selection, tokenization, and masking logic
and checked that it matched the intended training task.
"""

import json
from pathlib import Path

import torch
from datasets import load_dataset

PROJECT = Path(__file__).resolve().parents[2]
REPO = "aditya-6122/tiny-stories-instruct"
REVISION = "5eecccd2954de4a3f1ac5ac2b0fb13ee9a2d7ae9"


def make_prompt(summary):
    return f"Summary: {summary.strip()}\nStory:\n"


def prepare_data(tokenizer, train_size=10_000, val_size=500):
    folder = PROJECT / "data/sft_summary_hf"
    folder.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        REPO,
        revision=REVISION,
        cache_dir=str(PROJECT / "data/hf_sft_cache"),
    )

    eos = tokenizer.token_to_id("<|endoftext|>")
    seen, result = set(), {}

    for name, source, count in (
        ("train", "train", train_size),
        ("valid", "validation", val_size),
    ):
        rows, inputs, labels = [], [], []

        for row in dataset[source].shuffle(seed=42):
            summary = row["summary"].strip()
            story = row["story"].strip()

            if not summary or not story or story in seen:
                continue

            prompt = make_prompt(summary)
            p = tokenizer.encode(prompt).ids
            a = tokenizer.encode(story).ids + [eos]
            padding = 256 - len(p) - len(a)

            # Keep complete stories that fit our context.
            if padding < 0:
                continue

            seen.add(story)
            rows.append({"prompt": prompt, "response": story})

            inputs.append((p + a + [eos] * padding)[:-1])
            labels.append(
                ([-100] * len(p) + a + [-100] * padding)[1:]
            )

            if len(rows) == count:
                break

        if len(rows) != count:
            raise ValueError(
                f"Only {len(rows)} suitable {name} examples."
            )

        (folder / f"{name}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )

        result[name] = (
            torch.tensor(inputs),
            torch.tensor(labels),
        )
        print(f"SFT {name}: {len(rows):,} examples", flush=True)

    return result, folder