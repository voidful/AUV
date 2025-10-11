# AUV

> Teaching **A**udio **U**niversal **V**ector Quantization with Single Nested Codebook

[![python](https://img.shields.io/badge/Python-3.8-brightgreen?logo=Python&style=for-the-badge)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/Paper-2509.21968-b31b1b.svg?logo=arXiv&style=for-the-badge)](https://arxiv.org/abs/2509.21968)
[![demo](https://img.shields.io/badge/Demo-Samples-orange.svg?logo=Github&style=for-the-badge)](https://swivid.github.io/AUV/)

## Setup
```bash
pip install auv
wget https://huggingface.co/SWivid/AUV/resolve/main/auv.pt
```

## Inference
Command line usage, reconstruct all `.wav` files under the `input-dir` and write to the `output-dir`:
```bash
auv-infer --input-dir INPUT_WAV_DIR --output-dir OUTPUT_WAV_DIR --ckpt CKPT_PATH
# if torch.bfloat16 inference: --bf16
# if need to assign gpu: --device cuda:0
```

Python script usage see [`src/auv/infer.py`](src/auv/infer.py).
