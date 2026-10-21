# Model and runtime notices

[HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
is Apache-2.0 licensed. The download is pinned to revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`. Weights and tokenizer assets are not
included in the repository or release. The download preserves upstream README
and available license files. Raw reports hash all downloaded top-level files.

The reference uses FP32 CPU execution, greedy decoding, eager attention and the
Transformers 4.46.3 streamer API. See the [upstream generation documentation](https://huggingface.co/docs/transformers/v4.46.3/en/internal/generation_utils).
SmolLM2 uses the built-in Llama implementation; custom model code is disabled.
PyTorch 2.2.2 is used because it supplies an Intel macOS wheel for this host.
These are reproducibility pins, not a claim that these versions are current or
suitable for an exposed service. Use trusted local files only.
