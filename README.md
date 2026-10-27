# Ornith-1.5-122B on one 24 GB GPU and 32 GB of RAM

A 122B-parameter mixture-of-experts coding model (Ornith-1.5-122B, MIT) running on a single consumer GPU with
24 GB of VRAM and 32 GB of system RAM, at about 22 tokens/s on real coding output, with a patched
[Strata](https://github.com/Niko1221/Strata) engine and [unsloth's](https://huggingface.co/unsloth/Ornith-1.5-122B-GGUF)
UD-IQ2_XXS quant.

The first version that ran at all did 0.4 tokens/s on the same machine. Nothing in the hardware changed; every step below is a
software fix, measured on the same coding prompt.

## The machine

| Part | |
|---|---|
| GPU | AMD Radeon RX 7900 XTX, 24 GB (ROCm 7.2.3) |
| CPU | Intel i7-7700K, 4 cores |
| RAM | 32 GB DDR4, running at 2133 MT/s (XMP off) |
| Storage | NVMe SSD |
| PCIe | 3.0 x16 |

An old CPU, slow RAM and PCIe 3.0: most of what limits the speed here is on the host side, not the GPU.

## The model file (42.0 GB)

unsloth's `Ornith-1.5-122B-UD-IQ2_XXS.gguf`, used as is. Its experts get more bits in the later layers
(IQ1_S in the first layers, up to IQ3_XXS in the last ones). Stock Strata could not load it; the third patch here
adds those expert types.

Quality against the BF16 model (KL divergence on held-out text, 2048-token windows, same text for every file):

| File | Size | KLD vs BF16 | Same top token |
|---|---|---|---|
| **unsloth UD-IQ2_XXS (this setup)** | **42.0 GB** | **0.1225** | **88.19%** |
| AtomicChat AD-IQ3_XXS-IQ2_S | 47.6 GB | 0.1361 | 87.76% |

Correctness of the new kernels: greedy decoding with Strata and with llama.cpp on the same file gives the same 100 of
100 tokens on a coding prompt. On two other prompts they split at tokens 90 and 66, both at near-ties; llama.cpp
splits from itself at token 36 when only the GPU/CPU placement of experts changes.

## Speed

All numbers: real coding prompts, 600 generated tokens, greedy, no speculative decoding.

| Step | Decode tokens/s |
|---|---|
| Stock engine, experts copied into RAM | no token in 22 minutes (RAM full, swapping) |
| Experts read from a repacked file on the SSD instead of an in-RAM copy | 0.25-0.46 |
| Host-to-GPU copies through a pinned bounce buffer (see below) | 12.5 |
| Expert profile + RAM tier + VRAM slots for the most used experts | 15.7 |
| Dense weights in less VRAM, embeddings in RAM, profile from coding tasks | 16.7 |
| **unsloth UD-IQ2_XXS (smaller experts, more of them on the GPU)** | **21-25, about 22 typical** |

The steps up to 16.7 were measured on AtomicChat's file; the last row is the setup in this repository. Run-to-run
noise on this box is about 15%.

Long prompts:

| | Prompt (8,886 tokens) | Decode at 9K context |
|---|---|---|
| **Long-prompt mode** (streamed layers, borrowed staging) | **407 tokens/s** (22 s) | 12.2 tokens/s |

What did not help: letting the CPU workers sleep sooner (slower), moving experts between GPU and RAM during
generation (more copies than gain), 4096-token prompt chunks (crash in the hybrid path).

MTP (the model's own draft head) gives about 24 tokens/s with 1 draft token, but on long prompts the greedy output is
not yet reproducible between runs with it on, so it is off here until that is understood.

## What was wrong, and the fixes

1. **Pinned file pages.** On ROCm, a `hipMemcpy` from a memory-mapped file pins the source pages and they never
   leave RAM. Loading the dense weights and the GPU-resident experts straight from mapped files kept about 20 GB of
   file pages in a 32 GB machine, so the CPU's experts went to swap. Every host-to-GPU copy from a mapping now goes
   through a small pinned bounce buffer: resident file pages dropped from 20.9 GB to 0.4 GB.
2. **One copy of each expert, in the right place.** The experts are repacked once into a file on the SSD (the
   layout the engine reads). A profile says which experts are used most when writing code: those get the VRAM slots,
   the next ones an 18 GB RAM tier the page cache cannot evict, the rest are read from the SSD.
3. **The profile matters more than the slot count.** A profile from general calibration text covered 58% of the
   expert calls on code; one recorded on 30 coding tasks covers 78%. With this file, about 78% of expert calls run
   on the GPU on short prompts.
4. **Long prompts.** Big prompt chunks stream each layer's experts to the GPU; the staging buffers borrow the VRAM
   of a few layers' slots during the prompt and give it back afterwards, and the CPU gathers experts on 4 threads.
5. **New expert types.** IQ1_S gate/up and IQ1_S/IQ2_XXS/IQ2_XS/IQ2_S down kernels for decode and prompt, two loader
   fixes, with parity tests against ggml on real rows from the model.

Run the engine with `vm.swappiness=1` on a 32 GB machine, otherwise the kernel swaps the RAM tier out.

## Reproduce

1. Build the patched Strata: the three patches in this repo apply on top of Strata `4b9e05a2`
   (also on the branch [`ornith-122b-24gb`](https://github.com/lapadatbogdan/Strata/tree/ornith-122b-24gb) of our fork), HIP build as in Strata's docs.
2. Get the model: `scripts/get_model.sh <dir>` (42.0 GB from unsloth's repository).
3. Run: `scripts/run.sh <build dir> <model> profiles/ornith-1.5-122b-code.q36profile`, or with `LONG=1` for long
   prompts (agents). The first run writes a 33 GB `.q36pack` next to the model, so plan for about 75 GB of disk.
   Record your own profile with `STRATA_Q36_PROFILE_OUT`.

## Not measured (yet)

- NVIDIA GPUs. Expected to be faster: the host side dominates here, and a modern CPU with DDR5 has 2-3x the
  memory bandwidth of this box. **Estimate, not a measurement:** roughly 2x on an RTX 4090 with DDR5.
  Numbers from other machines are very welcome.

## Credits

Ornith-1.5-122B by [ornith-ai](https://huggingface.co/ornith-ai/Ornith-1.5-122B) (MIT). Quantization by
[unsloth](https://huggingface.co/unsloth/Ornith-1.5-122B-GGUF); [AtomicChat's](https://huggingface.co/AtomicChat/Ornith-1.5-122B-GGUF)
quant was the first file we got running. Engine: [Strata](https://github.com/Niko1221/Strata) (MIT); a first
Ornith patch was posted by kirinriki7777-sys in Strata issue #3574.

License of this repository: MIT.
