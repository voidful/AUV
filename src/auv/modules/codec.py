import sys
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torchaudio.functional import resample

from auv.modules.conformer import ConformerEncoder
from auv.modules.residual_vq import ResidualVQ
from auv.modules.stft import ISTFTHead, STFTHead


class AUVCodecEncWithVQ(nn.Module):
    def __init__(
        self,
        *,
        enc_conf: Dict[str, Any],
        vq_conf: Optional[Dict[str, Any]] = None,
        init_model: Optional[str] = None,
        sample_rate: int = 24000,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.dim = enc_conf["hidden_size"]
        self.hop_length = enc_conf["hop_length"]
        self.stft_head = STFTHead(dim=self.dim, n_fft=self.hop_length * 4, hop_length=self.hop_length)
        self.conformer = ConformerEncoder(**enc_conf)

        # VQ
        if vq_conf is None:
            vq_conf = {
                "vq_type": "fvq",
                "num_quantizers": 1,
                "dim": 512,
                "codebook_size": 8192,
                "codebook_dim": 8,
                "commitment": 0.25,
                "codebook_partition_maps": None,
            }
        self.codebook_partition_maps = vq_conf.pop("codebook_partition_maps")
        if self.codebook_partition_maps is not None:
            assert "all" not in self.codebook_partition_maps
            self.codebook_partition_maps["all"] = [0, sys.maxsize]
        self.vq = ResidualVQ(**vq_conf)

        # init model
        if init_model is not None:
            model_dict = torch.load(init_model, map_location="cpu")
            self.load_state_dict(model_dict)

    def train(self, mode=True):
        super().train(mode)

    def get_hidden_feat(self, x, input_sample_rate=None):
        assert x.dim() == 2, "wav input for auv encoder should with dim==2"
        # resample
        if input_sample_rate is not None and input_sample_rate != self.sample_rate:
            x = resample(x, orig_freq=input_sample_rate, new_freq=self.sample_rate)

        feat = self.stft_head(x)
        (feat,) = self.conformer(feat)
        feat = feat.transpose(1, 2)

        return feat

    def forward(self, wav_input, wav_types=None, input_sample_rate=None, **kwargs):
        hidden_feats = self.get_hidden_feat(wav_input, input_sample_rate=input_sample_rate)

        codebook_partitions = (
            [self.codebook_partition_maps[wav_t] for wav_t in wav_types]
            if self.codebook_partition_maps is not None and wav_types is not None
            else None
        )
        quantized, indices, vq_loss = self.vq(hidden_feats, codebook_partitions=codebook_partitions)

        vq_loss_items = {"vq_loss": vq_loss.sum().item()}
        res = {
            "quantized": quantized.transpose(1, 2),
            "tokens": indices,
            "vq_loss": vq_loss.sum(),
            "vq_loss_items": vq_loss_items,
            "before_quantize": hidden_feats.transpose(1, 2),
        }
        return res

    def vq2emb(self, token):
        x = self.vq.vq2emb(token)
        return x.transpose(1, 2)


class AUVCodecDec(nn.Module):
    def __init__(
        self,
        *,
        dec_conf: Dict[str, Any],
        distill_depth: int = -1,
    ):
        super().__init__()

        self.dim = dec_conf["hidden_size"]
        self.hop_length = dec_conf["hop_length"]
        self.conformer = ConformerEncoder(**dec_conf)
        self.istft_head = ISTFTHead(dim=self.dim, n_fft=self.hop_length * 4, hop_length=self.hop_length)

        self.distill_depth = distill_depth

    def forward(self, x, **kwargs):
        if kwargs.get("return_hidden_feat", False):
            x, all_hidden_feats = self.conformer(x, output_hidden_states=True)
            hidden_feat = all_hidden_feats[self.distill_depth]  # [B, T, C]
        else:
            (x,) = self.conformer(x)

        x = self.istft_head(x).unsqueeze(1)

        if kwargs.get("return_hidden_feat", False):
            return x, hidden_feat

        return x
