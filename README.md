# mini-CCGQA

**Compressed attention and an input-dependent extension.**

Attention connects information across tokens in a sequence, but computing these
relationships can be expensive. Using smaller query, key and value projections can
reduce parameters and attention arithmetic, but prediction quality
can suffer. This project explores whether better use of local context
can recover some of that loss.

I implement compressed convolutional grouped-query attention (CCGQA)
used in [ZAYA1-8B](https://arxiv.org/pdf/2605.05365v1#page=3) and described
in the [CCA paper](https://arxiv.org/pdf/2510.04476v2#page=4).
I also propose a small input-dependent extension, **Dynamic CCGQA**,
and compare both with full-width and naively compressed GQA on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).

## What changes between the models?

- **Full GQA** is the reference model, with a head dimension of 24.
- **Naive compression** halves that dimension to 12, leaving the residual
  stream, feedforward network and number of layers unchanged.
- **CCGQA** uses the same compressed widths but adds two causal Q/K
  convolutions: one filters each channel across time, and the other mixes
  time and channels within each head. Shared Q/K features and a one-token
  delay on half the value channels add further local information.
- **Dynamic CCGQA** changes only the first convolution. Small
  input-dependent gates adjust the contributions from the previous and
  current positions, allowing the local mixture to adapt to context. This is our proposed extension to improve CCGQA, explained in
  [Section 5 of the theory notes](docs/theory.md#5-dynamic-ccgqa).

All four models are small dense decoder Transformers with SwiGLU
feedforward layers. The focus is on ZAYA's attention construction;
its mixture-of-experts layers, router and learned residual scaling
are left for future work.

The [theory notes](docs/theory.md) explain the mathematics and
implementation in detail.

## Experiment and results

All four models are trained from scratch with the same tokenizer,
training-window stream and evaluation windows.

| Shared setting | Value |
|---|---|
| Vocabulary | 2,048-token byte-level BPE |
| Residual width / feedforward width | 192 / 512 |
| Layers | 6 |
| Query heads / KV heads | 8 / 2 |
| Batch size / sequence length | 16 / 256 |
| Optimizer updates | 12,000 |
| Training-token exposures | 49,152,000 per model |
| Optimizer | AdamW |
| Learning rate | 240-step warm-up to `3e-4`, then cosine decay to `3e-5` |
| Seed | 42 |
| Validation | Fixed 65,536-token sample, evaluated every 250 updates |

The tokenizer is trained on training stories only. Exact matches with
training stories and duplicate validation stories are removed from the
validation split before tokenization.

At the final checkpoint:

| Model | Parameters | Validation loss (nats/token) | Perplexity |
|---|---:|---:|---:|
| Full GQA | 2,718,156 | 2.3322 | 10.300 |
| Naive compressed GQA | 2,441,676 | 2.5326 | 12.586 |
| CCGQA | 2,461,836 | 2.4810 | 11.954 |
| Dynamic CCGQA | 2,471,076 | 2.4684 | 11.804 |

![Validation loss over the complete run and during later training](assets/comparison/01_validation_loss.png)

The CCGQA variants achieve lower validation loss during much of early
training, but full GQA overtakes both and remains ahead from step 4,500
onward. Among the compressed models, CCGQA improves on naive compression,
and Dynamic CCGQA provides a further small gain. The corresponding
[perplexity curves](../assets/comparison/02_validation_perplexity.png)
show similar trends.

### What CCGQA recovers

CCGQA finishes with **5.03% lower perplexity than naive compressed GQA**.
Dynamic CCGQA increases that reduction to **6.22%**. Relative to the naive
model, these additions increase the parameter count by **0.83%** and
**1.20%**, respectively.

![Perplexity reduction relative to naive compressed GQA](assets/comparison/06_relative_to_naive_compression.png)

Both variants recover part of the quality lost through dimensional
compression. Dynamic CCGQA maintains a small advantage over CCGQA at
every recorded evaluation after initialization. 

At this small scale, the dynamic extension therefore provides a modest
gain that persists across the observed training run. This motivates
longer runs and larger-model experiments to test whether the advantage
continues under those settings.

### What the dynamic gates add

Dynamic CCGQA achieves **1.26% lower perplexity than CCGQA**, by adding just
9,240 parameters, i.e. approximately **0.38%** of CCGQA's parameter count.

![Loss and perplexity improvements of Dynamic CCGQA over CCGQA](assets/comparison/03_dynamic_advantage.png)

Its validation loss is lower at every recorded evaluation after
initialization. The improvement is modest, but persists throughout the
observed training run.

These experiments match training tokens and the surrounding architecture,
not total parameters or FLOPs. 

The results cover one seed, model scale and training schedule. Additional
seeds, larger models and longer training would test whether the gains
persist.

All six figures, including
[perplexity curves](assets/comparison/02_validation_perplexity.png),
[comparisons against full GQA](assets/comparison/04_relative_to_full_gqa.png)
and the [training–validation gap](assets/comparison/05_validation_gap.png),
are available with the numerical tables in [assets/comparison](assets/comparison/).

## Supervised fine-tuning

I also fine-tune Dynamic CCGQA for summary-to-story generation using
[TinyStories instruction pairs](https://huggingface.co/datasets/aditya-6122/tiny-stories-instruct).

Full-weight SFT uses 10,000 training examples for four epochs, giving
2,500 optimizer updates. The model receives a summary followed by
`Story:`, and learns to predict the corresponding story. Only story
tokens and EOS contribute to the loss; prompt and padding positions
are excluded. The base checkpoint is preserved.

Both checkpoints are evaluated on the same 500 SFT validation examples:

| Model | Response loss | Perplexity | Next-token accuracy |
|---|---:|---:|---:|
| Base Dynamic CCGQA | 2.4653 | 11.767 | 46.49% |
| After SFT | 2.1525 | 8.606 | 51.03% |

SFT reduces response perplexity by **26.86%** and increases next-token
accuracy by **4.54 percentage points**. Accuracy counts how often the
highest-probability token matches the reference next token.

These measurements use teacher forcing: each prediction receives the
true preceding story tokens. They demonstrate improved conditional
token prediction, while freely generated stories still show repetition,
inconsistent characters and topic drift. The SFT validation set differs
from the pretraining evaluation set.

### Generation before and after SFT

The following example uses the same prompt and greedy decoding for both
checkpoints, with a limit of 212 new tokens. The generated text is
reproduced without corrections.

**Prompt**

```text
Summary: Dave goes on a mission to his neighbor's garden to find delicious vegetables and declares it the best lunch ever.
Story:
```

**Before SFT: base Dynamic CCGQA**

```text
The moral of the story is that it's important to be careful and listen to your friends.
```

**After SFT: Dynamic CCGQA**

```text
Once upon a time, there was a little boy named Jack. Jack was very excited because he had a special toy. He was very excited because he had never seen anything like it before.
One day, Jack decided to go to the store. He saw a big box in the store and he said, "I want to see what's inside." Jack was excited and asked, "What is that?"
Jack was excited and said, "I'm going to get some more. It's a special treat. It's a special treat. It's a special treat for you."
Jack was so excited and he ran to the store to buy some food. He was so happy to have a new friend. He was so happy to have a new friend and he was so happy.
```

In this example, SFT produces a story rather than a generic moral.
Although the response remains repetitive and does not follow the summary
closely, it shows some adaptation to the expected narrative structure.

Our preliminary attempts at further post-training did not produce a
clear additional improvement. The model's small capacity may be one
limiting factor. Testing larger models and revisiting preference-based
training, including DPO, are directions for future work to investigate
whether this emerging structure can develop into more coherent stories
that better follow the prompt.

The full set of paired outputs is available in
[the generation samples](runs/sft/2.47M-ccgqa_dynamic-tinystories-d192-ff512-l6-q8-kv2-hd12-v2048-drop0-seq256-seed42-d4c0d6d5d0f6-summary/evaluation/20260927-190809-220801/samples.md).


## Running the project

Use Python 3.12 and run the following commands from the repository root.
Install the locked environment with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
```

### Use the saved models

The repository includes the completed runs, their original tokenizer
and the SFT validation examples.

Generate the two examples defined in `generate.py`:

```bash
uv run python -m mini_ccgqa.generate
```

The module also exposes `load_model()` and `generate_text()` for use from
Python. It supports greedy decoding (`temperature=0`), top-k and top-p
sampling through Hugging Face's generation utilities.

The supplied examples use different sampling settings to demonstrate
these options. For model comparisons, use matching prompts and decoding
settings.

Rebuild the six pretraining comparison figures and numerical tables:

```bash
uv run python -m mini_ccgqa.plot_comparison
```

Outputs are saved in `assets/comparison/` as PNG, SVG and CSV files.

Compare the saved base and SFT checkpoints:

```bash
uv run python -m mini_ccgqa.compare_sft
```

This saves validation metrics, two figures and 80 continuations from
20 shared prompts under the SFT run's `evaluation/` directory.

### Train a new experiment

For a fresh experiment, use a separate working copy with empty `data/`
and `runs/` directories. Keep the supplied checkpoints and their tokenizer
intact in the original copy.

Prepare the pretraining data:

```bash
uv run python -m mini_ccgqa.data
```

This downloads TinyStories, trains the BPE tokenizer, filters the
validation split and writes `train.bin` and `validation.bin`.

Train the four models:

```bash
uv run python -m mini_ccgqa.train
```

The configurations at the bottom of `train.py` run sequentially.

Each run saves:

- `metadata.json`: model, data and training settings.
- `history.csv`: recorded training and validation metrics.
- `loss.png` and `ppl.png`: learning curves.
- `best_model.pt`: the checkpoint with the lowest validation loss.
- `last.pt`: model, optimizer and random-generator states for resuming.

Re-running resumes an existing run. A completed run does not automatically
receive additional training updates.

For summary-to-story SFT:

```bash
uv run python -m mini_ccgqa.sft
```

The SFT module prepares the instruction data and trains from the selected
base checkpoint. Its outputs are stored separately under `runs/sft`;
base checkpoints remain unchanged under `runs/base`.

The generation and SFT examples refer to the supplied run names.
Update those paths if a new experiment produces different names.
Full token arrays and download caches are generated locally and are
excluded from the repository.

## Code layout

| Module | Purpose |
|---|---|
| `model.py`, `config.py` | Transformer components, attention variants and configuration definitions |
| `data.py`, `train.py` | Tokenizer, pretraining data and four-model training experiment |
| `plot_comparison.py` | Comparison figures and final metric tables |
| `generate.py` | Checkpoint loading and autoregressive generation |
| `sft_data.py`, `sft.py` | Summary-to-story data preparation and full-weight SFT |
| `compare_sft.py` | Base/SFT evaluation, plots and paired generation |

See [the source](src/mini_ccgqa/) for the implementation and
[the theory notes](docs/theory.md) for the mathematical explanation,
experimental details and interpretation of the results.

## AI assistance declaration

I used generative AI mostly with visualization utilities, including plotting, figure layout and saving outputs. AI also helped refine the wording of this README and the theory notes.

*Human Work:* I planned the study, worked through the underlying mathematics, and independently implemented and checked the final model and configuration modules. I designed the experiments and visual comparisons, selected the final hyperparameters through multiple training runs, and analyzed
and interpreted the results.