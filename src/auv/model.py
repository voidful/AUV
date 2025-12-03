import torch
import torch.nn as nn

from auv.modules.codec import AUVCodecDec, AUVCodecEncWithVQ
from auv.modules.activations import Snake


class AUV(nn.Module):
    def __init__(self):
        super().__init__()

    def _init_snake_from_gelu(self):
        """
        Initialize Snake activation alpha parameters to approximate GELU behavior.
        This helps when loading from GELU pretrained weights.
        
        Snake: x + (1/alpha) * sin^2(alpha * x)
        When alpha=1.0, Snake provides a smooth non-linear activation that
        can serve as a good starting point for fine-tuning from GELU weights.
        """
        for module in self.modules():
            if isinstance(module, Snake):
                with torch.no_grad():
                    # Initialize alpha to 1.0 for smooth activation behavior
                    module.alpha.fill_(1.0)

    def from_pretrained(self, ckpt_path, device="cuda", init_snake_from_gelu=True):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.tokenizer = AUVCodecEncWithVQ(
            sample_rate=16000,
            enc_conf={
                "hop_length": 320,
                "num_layers": 8,
                "hidden_size": 512,
                "ffn_mult": 4,
                "max_position_embeddings": 4096,
            },
            vq_conf={
                "vq_type": "fvq",
                "num_quantizers": 1,
                "dim": 512,
                "codebook_size": 20480,
                "codebook_dim": 8,
                "commitment": 0.25,
                "codebook_partition_maps": {
                    "speech": [0, 8192],
                    "vocal": [0, 12288],
                    "music": [0, 20480],
                    "audio": [0, 20480],
                    "other": [12288, 20480],
                },
            },
        )
        self.token2wav = AUVCodecDec(
            dec_conf={
                "hop_length": 320,
                "num_layers": 12,
                "hidden_size": 512,
                "ffn_mult": 4,
                "max_position_embeddings": 4096,
            },
        )
        incompatible_keys = self.load_state_dict(ckpt, strict=False)
        if incompatible_keys.missing_keys:
            print(f"[AUV] Missing keys in checkpoint (using default values): {len(incompatible_keys.missing_keys)} keys")
            # Check if missing keys are Snake alpha parameters (from GELU pretrained weights)
            snake_alpha_missing = any('.alpha' in k for k in incompatible_keys.missing_keys)
            if snake_alpha_missing and init_snake_from_gelu:
                print("[AUV] Detected GELU->Snake conversion, initializing Snake alpha parameters...")
                self._init_snake_from_gelu()
        if incompatible_keys.unexpected_keys:
            print(f"[AUV] Unexpected keys in checkpoint (ignored): {len(incompatible_keys.unexpected_keys)} keys")
        del ckpt

    def forward(self, feature):
        raise NotImplementedError

    @torch.no_grad()
    def encode(self, data):
        wav_input = data["sample"]
        sample_rate = data["sample_rate"]
        assert wav_input.size(0) == 1, "Only support batch_size == 1 when inference"

        tokenizer_out = self.tokenizer(wav_input, input_sample_rate=sample_rate)
        quantized = tokenizer_out["quantized"]
        tokens = tokenizer_out["tokens"]
        before_quantize = tokenizer_out["before_quantize"]

        res = {
            "quantized": quantized,
            "tokens": tokens,
            "before_quantize": before_quantize,
        }

        return res

    @torch.no_grad()
    def decode(self, emb, **kwargs):
        audio = self.token2wav(emb)
        return audio

    @torch.no_grad()
    def decode_tokens(self, tokens, **kwargs):
        quantized = self.tokenizer.vq2emb(tokens)
        audio = self.token2wav(quantized)
        return audio
