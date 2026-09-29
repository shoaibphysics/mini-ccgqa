"""Model and training configuration definitions.

I independently wrote and checked these configuration definitions.
No AI code-generation assistance was used here.
I selected the hyperparameters for the final experiments based on
multiple training runs.
"""

from dataclasses import dataclass

@dataclass
class ModelConfig:
    # Overall model
    vocab_size: int = 2048
    d_model: int = 128
    layers: int = 3
    d_ff: int = 384

    # Attention settings
    attention_type: str = "gqa"  # "gqa" or "ccgqa"
    q_heads: int = 4
    kv_heads: int = 2
    head_dim: int = 16

    # Position encoding and normalization
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5

    # Regularization
    dropout: float = 0.0


@dataclass
class TrainConfig:
    # Small defaults for local checks.
    total_steps: int = 100
    batch_size: int = 2
    seq_len: int = 32
    accumulation_steps: int = 1
    seed: int = 42

    # Evaluation and checkpointing.
    eval_every: int = 5
    save_every: int = 5
    eval_batches: int = 4

    # Optimizer and learning-rate schedule.
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_fraction: float = 0.1
    weight_decay: float = 0.05