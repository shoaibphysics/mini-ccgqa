from array import array
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
import hashlib

DATA_DIR = Path("data")
DATASET = "roneneldan/TinyStories"
REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
VOCAB_SIZE = 2048
EOS = "<|endoftext|>"


def stories(split):
    """Yield nonempty story texts."""
    for row in split:
        text = row["text"]
        if text and text.strip():
            yield text


def prepare_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer_path = DATA_DIR / "tokenizer.json"
    train_path = DATA_DIR / "train.bin"
    validation_path = DATA_DIR / "validation.bin"

    # Reuse the completed preparation on later runs.
    if all(path.exists() for path in (
        tokenizer_path, train_path, validation_path
    )):
        print("Prepared data already exists.")
        return

    # Download once, then read the locally cached dataset.
    dataset = load_dataset(
        DATASET,
        revision=REVISION,
        streaming=False,
    )

    # Train BPE on 50,000 training stories only.
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False
    )
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[EOS],
        show_progress=True,
    )

    print("Training tokenizer...")
    tokenizer.train_from_iterator(
        islice(stories(dataset["train"]), 50_000),
        trainer=trainer,
        length=50_000,
    )
    tokenizer.save(str(tokenizer_path))
    eos_id = tokenizer.token_to_id(EOS)

    # Keep one copy of each validation story, preserving its order.
    validation = {}
    for text in stories(dataset["validation"]):
        key = hashlib.sha256(text.strip().encode("utf-8")).digest()
        validation.setdefault(key, text)

    validation_before = len(validation)

    # Encode training first, removing validation overlaps as we go.
    for split_name in ("train", "validation"):
        if split_name == "train":
            batches = dataset["train"].iter(batch_size=1000)
        else:
            retained = list(validation.values())
            print("Validation overlaps removed:", validation_before - len(retained))
            print("Retained validation stories:", len(retained))
            batches = (
                {"text": retained[start:start + 1000]}
                for start in range(0, len(retained), 1000)
            )

        output_path = DATA_DIR / f"{split_name}.bin"
        temporary_path = DATA_DIR / f"{split_name}.bin.partial"
        token_count = 0

        print(f"Encoding {split_name}...", flush=True)

        with temporary_path.open("wb") as file:
            for batch in batches:
                texts = [text for text in batch["text"] if text and text.strip()]

                if split_name == "train":
                    for text in texts:
                        key = hashlib.sha256(text.strip().encode("utf-8")).digest()
                        validation.pop(key, None)

                encoded = tokenizer.encode_batch(texts, add_special_tokens=False)
                ids = array("H")
                for story in encoded:
                    ids.extend(story.ids)
                    ids.append(eos_id)

                np.asarray(ids, dtype="<u2").tofile(file)
                token_count += len(ids)

        temporary_path.replace(output_path)
        print(f"Saved {split_name}: {token_count:,} tokens")

    print("Data preparation finished.")

def load_tokens(split: str):
    """Read saved token IDs without loading the whole file into RAM."""
    return np.memmap(
        DATA_DIR / f"{split}.bin",
        dtype="<u2",
        mode="r",
    )


def get_batch(tokens, batch_size: int, seq_len: int, device="cpu", generator=None):
    """Sample input tokens and their next-token targets."""
    starts = torch.randint(len(tokens) - seq_len,(batch_size,),generator=generator)
    windows = np.stack([
        tokens[start:start + seq_len + 1]
        for start in starts.tolist()
    ])

    batch = torch.from_numpy(windows.astype(np.int64))
    x = batch[:, :-1].to(device)  # (B, T)
    y = batch[:, 1:].to(device)   # (B, T)
    return x, y


if __name__ == "__main__":
    prepare_data()

    train_tokens = load_tokens("train")
    validation_tokens = load_tokens("validation")

    x, y = get_batch(train_tokens, batch_size=2, seq_len=8)

    print("Training tokens:", len(train_tokens))
    print("Validation tokens:", len(validation_tokens))
    print("Input shape:", x.shape)
    print("Target shape:", y.shape)