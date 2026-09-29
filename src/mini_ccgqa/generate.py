"""Generate text from saved model checkpoints.

I used substantial AI assistance to write the generation utilities.
I studied the decoding methods, selected the sampling settings,
and assessed the generated outputs.
"""

from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import GenerationMixin, PreTrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from mini_ccgqa.config import ModelConfig
from mini_ccgqa.model import TransformerLM


class TextGenerator(PreTrainedModel, GenerationMixin):
    """Connect our model to Hugging Face's generation loop."""

    def __init__(self, model, eos_id):
        config = PreTrainedConfig(
            vocab_size=model.token_embedding.num_embeddings,
            eos_token_id=eos_id,
            pad_token_id=eos_id,
            use_cache=False)
        super().__init__(config)
        self.model = model

    def forward(self, input_ids, attention_mask=None, **kwargs):
        logits, _ = self.model(input_ids)
        return CausalLMOutput(logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # Keep the complete prefix for CCGQA's convolution history.
        return {"input_ids": input_ids}


def load_model(run_dir):
    """Load a trained checkpoint and its existing tokenizer."""
    project = Path(__file__).resolve().parents[2]
    tokenizer = Tokenizer.from_file(str(project / "data/tokenizer.json"))

    saved = torch.load(
        Path(run_dir) / "best_model.pt",
        map_location="cpu",
        weights_only=True )
    core = TransformerLM(ModelConfig(**saved["config"]))
    core.load_state_dict(saved["model"])

    model = TextGenerator( core, tokenizer.token_to_id("<|endoftext|>") )

    model.model_name = {
        "gqa": "GQA",
        "ccgqa": "CCGQA",
        "ccgqa_dynamic": "Dynamic CCGQA"}[saved["config"]["attention_type"]]
    stage = Path(run_dir).parent.name  
    model.stage = "Base" if stage == "base" else stage.upper()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device).eval()
    return model, tokenizer


def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens=128,
    temperature=0.8,
    top_p=0.9,
    top_k=0,
    seed=None):
    """Return the prompt followed by its generated continuation.

    temperature=0: greedy decoding.
    top_k=0: disable top-k filtering.
    top_p=1.0: disable top-p filtering.
    """
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids

    # Our current models were trained with a context length of 256.
    if not ids or not 1 <= max_new_tokens <= 256 - len(ids):
        raise ValueError("Keep a nonempty prompt + new tokens within 256 tokens.")
    if temperature < 0:
        raise ValueError("Temperature must be nonnegative.")

    if seed is not None:
        torch.manual_seed(seed)

    input_ids = torch.tensor([ids], device=model.device)

    settings = {"do_sample": temperature > 0}
    if temperature > 0:
        settings.update(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k)

    num_params = sum(p.numel() for p in model.parameters())

    if temperature == 0:
        method = "Greedy decoding"
    else:
        method = (
            f"Sampling | temperature={temperature:g}"
            f" | top_p={top_p:g} | top_k={top_k}")

    print("\n" + "=" * 80)
    print(f"{model.stage} | {model.model_name}"
          f" | {num_params / 1e6:.2f}M parameters")
    print(f"{method} | max_new_tokens={max_new_tokens}"
          f" | sampling_seed={seed}")
    print("-" * 80)

    output = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        use_cache=False,
        **settings)

    story = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    print(story)
    print("=" * 80)
    return story


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[2]
    run_dir = project / (
        "runs/base/2.47M-ccgqa_dynamic-tinystories-d192-ff512-l6-q8-kv2-hd12-"
        "v2048-drop0-seq256-seed42-d4c0d6d5d0f6"
    )

    model, tokenizer = load_model(run_dir)

    # Example: top-p sampling.
    story = generate_text(
        model,
        tokenizer,
        prompt="Once upon a time, a little rabbit",
        max_new_tokens=128,
        temperature=0.8,
        top_p=0.9,
        top_k=0,
        seed=42,
    )


    # Second example: CCGQA with top-k sampling.
    run_dir = project / (
        "runs/base/2.46M-ccgqa-tinystories-d192-ff512-l6-q8-kv2-hd12-"
        "v2048-drop0-seq256-seed42-ba0ecc023377"
    )

    model, tokenizer = load_model(run_dir)

    story = generate_text(
        model,
        tokenizer,
        prompt="Once upon a time, a little rabbit",
        max_new_tokens=128,
        temperature=0.8,
        top_p=1.0,
        top_k=40,
        seed=42,
    )