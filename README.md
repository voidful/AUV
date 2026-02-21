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

### Batch Inference (Python API)
AUV natively supports variable-length batch inference for efficient encoding pipelines. 
If your Dataloader pads multiple audio samples together (e.g. `[Batch, Time]`), you can pass their original valid lengths to `auv.encode` to automatically mask padded regions and receive the downsampled output lengths:

```python
from auv.model import AUV
import torch

auv = AUV()
auv.from_pretrained("auv.pt", device="cuda:0")

# Packed batch input: [Batch, Time]
batch_wav_tensor = torch.randn(32, 160000).cuda()
# Original lengths of each sample before padding
batch_lengths = torch.tensor([80000, 160000, ...]).cuda()

data = {
    "sample": batch_wav_tensor,
    "sample_rate": 16000,
    "lengths": batch_lengths # Optional: Triggers Padding Masking
}

with torch.no_grad():
    res = auv.encode(data)

# Extract robust 2D Token Matrix: [Batch, Max_Output_Time]
tokens_2d = res["tokens"] 

# Automatically calculated downsampled valid lengths: [Batch]
out_lengths = res["lengths"]

# Example: Slice valid tokens out safely
valid_tokens = [t[:l] for t, l in zip(tokens_2d, out_lengths)]
```
