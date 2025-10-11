import os
from argparse import ArgumentParser
from glob import glob
from os.path import basename, join

import torch
import torchaudio
from tqdm import tqdm

from auv.model import AUV


parser = ArgumentParser()
parser.add_argument("--input-dir", type=str, default=".")
parser.add_argument("--output-dir", type=str, default="outputs")
parser.add_argument("--ckpt", type=str, default="auv.pt")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--bf16", action="store_true")
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir
ckpt = args.ckpt
device = args.device
bf16 = args.bf16


def main():
    print(f"Load codec ckpt from {ckpt}")
    print(f"Use bf16: {bf16}")
    auv = AUV()
    auv.from_pretrained(ckpt)
    auv = auv.to(device)
    auv.eval()
    target_sr = auv.tokenizer.sample_rate

    wav_paths = glob(join(input_dir, "*.wav"))

    os.makedirs(output_dir, exist_ok=True)
    for wav_path in tqdm(wav_paths):
        target_wav_path = join(output_dir, basename(wav_path))
        wav, sr = torchaudio.load(wav_path)
        wav = wav.to(device)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        data = {
            "sample": wav,
            "sample_rate": sr,
            # "tokens": vq_code,  # will skip tokenization, directly use the given indices to decode
        }

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=bf16,
        ):
            enc_res = auv.encode(data)
            vq_post_emb = enc_res["quantized"]
            # vq_code = enc_res["tokens"]
            # vq_pre_emb = enc_res["before_quantize"]

            recon = auv.decode(vq_post_emb)
            # recon = auv.decode_tokens(vq_code.transpose(1, 2))

        torchaudio.save(target_wav_path, recon[0].cpu(), target_sr)


if __name__ == "__main__":
    main()
