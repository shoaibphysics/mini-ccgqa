| Model | Response loss | Perplexity | Token accuracy | PPL reduction vs Base |
|---|---:|---:|---:|---:|
| Base | 2.4653 | 11.767 | 46.49% | 0.00% |
| SFT | 2.1525 | 8.606 | 51.03% | 26.86% |

Loss reduction: 0.3129 nats/token.
Token-accuracy change: +4.54 percentage points.

Metrics use the true preceding story tokens; prompts and padding are excluded. They measure next-token prediction, not generated-story quality or summary adherence.
