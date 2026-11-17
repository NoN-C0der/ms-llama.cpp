# MS (Mad Shift) quantization

MS (Mad Shift) is a GGUF weight format for llama.cpp created by **Matthew Priester**
([GitHub](https://github.com/devvexus), [ORCID 0009-0000-0032-5292](https://orcid.org/0009-0000-0032-5292)). At the same file
size, MS files stay much closer to the original model than standard K-/I-quants and unsloth's dynamic quants: on Qwen3.5-2B,
4.5-9.4x lower KL divergence to the BF16 model than unsloth's smallest file of the same or larger size; on
Qwen3.8-27B, 1.61-2.31x lower than unsloth's UD3 file of about the same size (revision 4ca72078, August 2026).

Models: [huggingface.co/devvexus](https://huggingface.co/devvexus) ("MS Quants" collection).

## Format

- **Types:** `MS_C1` ... `MS_C6` (k = 1..6 bits per weight; type ids 43..48 on this fork's master), file type 42.
- **Block:** 512 weights along a row: fp16 scale `d`, 16-bit field holding the 14-bit start state `s0`, then 511 k-bit symbols
  (MSB first, k zero pad bits): k + 0.0625 bits per weight.
- **Decode** (the bitshift-trellis construction and the computed "1MAD" decoding function described in QTIP): for each weight, `s = ((s << k) | symbol) & 0x3FFF`
  (the first weight uses `s0`); `x = 34038481 * s + 76625530 (mod 2^32)`; `w = (byte0(x) + byte1(x) + byte2(x) + byte3(x) - 510) * d`.
  Every weight's state is the 14-bit window of the block's bit stream that ends at its symbol, so weights decode independently.
- **Rotation:** each MS tensor stores the codes of `W R`, where `R` is a random Hadamard rotation given by `prism.hadamard.*`
  metadata (block sizes, sign vectors); the graph rotates the input instead (`(W R)(R^T x) = W x`).
- **Split tables / heads:** an embedding table or output head may be stored as two tensors (`token_embd` + `token_embd_lo`,
  `output` + `output_lo`): the most frequent tokens at a higher k, the rest lower (the stored-row order is in
  `msffn.emb_split_order`); logits come out in vocabulary order.
- **Attribution keys** in released files: `general.author`, `general.url`, `general.description`.

## Run

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release   # or without -DGGML_CUDA=ON for CPU only
cmake --build build --config Release -j
./build/bin/llama-cli -m <MS file>.gguf -ngl 99
```

On ARM CPUs the default fast path rounds activations to 8 bits, as llama.cpp does for its own quantized types, which raises the KL divergence slightly; `--no-repack` selects the exact path, which matched the CUDA build in our test.

## Credit

MS uses the bitshift-trellis construction and the 1MAD decoding function described in QTIP: A. Tseng, Q. Sun, D. Hou, C. De Sa, *QTIP: Quantization with Trellises and Incoherence
Processing*, NeurIPS 2024, [arXiv:2406.11235](https://arxiv.org/abs/2406.11235). The MS format, its llama.cpp integration and
the MS files are by Matthew Priester; see `NOTICE`. The Qwen3.8-27B MS files also contain some tensors taken unchanged from
unsloth's GGUFs (see their model card).
