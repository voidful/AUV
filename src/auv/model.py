import torch
import torch.nn as nn

from auv.modules.codec import AUVCodecDec, AUVCodecEncWithVQ


class AUV(nn.Module):
    def __init__(self):
        super().__init__()

    def from_pretrained(self, ckpt_path, device="cuda"):
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
        self.load_state_dict(ckpt)
        del ckpt

    def forward(self, feature):
        raise NotImplementedError

    @torch.no_grad()
    def encode(self, data):
        wav_input = data["sample"]
        
        # Squeeze channel dimension if someone accidentally passes [B, 1, T]
        if wav_input.dim() == 3 and wav_input.size(1) == 1:
            wav_input = wav_input.squeeze(1)
            
        sample_rate = data["sample_rate"]
        lengths = data.get("lengths", None)

        tokenizer_out = self.tokenizer(wav_input, lengths=lengths, input_sample_rate=sample_rate)
        quantized = tokenizer_out["quantized"]
        tokens = tokenizer_out["tokens"]
        before_quantize = tokenizer_out["before_quantize"]

        codebook_size = self.tokenizer.vq.num_quantizers

        if tokens.dim() == 3:
            # tokens usually comes out as [num_quantizers, Batch, Time] from ResidualVQ stack
            # or could be [Batch, num_quantizers, Time]
            # Since AUV has num_quantizers = 1 usually, we squeeze that dimension out
            if tokens.size(0) == 1:
                tokens = tokens.squeeze(0)
            elif tokens.size(1) == 1:
                tokens = tokens.squeeze(1)

        assert tokens.dim() == 2, f"Expected 2D [Batch, Time] for tokens_2d, got {tokens.dim()}D: {tokens.shape}"
        assert quantized.dim() == 3, f"Expected 3D [Batch, Time, Dim] for quantized, got {quantized.dim()}D: {quantized.shape}"
        assert before_quantize.dim() == 3, f"Expected 3D [Batch, Time, Dim] for before_quantize, got {before_quantize.dim()}D: {before_quantize.shape}"
        
        res = {
            "quantized": quantized,
            "tokens": tokens,
            "before_quantize": before_quantize,
        }

        if lengths is not None:
            # calculate lengths after STFT and Conformer downsampling
            # STFT uses hop_length
            feat_lengths = torch.ceil(lengths / self.tokenizer.hop_length).long()
            res["lengths"] = feat_lengths

        return res

    @torch.no_grad()
    def decode(self, emb, **kwargs):
        assert emb.dim() == 3, f"Expected 3D [Batch, Time, Dim] for decode input, got {emb.dim()}D: {emb.shape}"
        audio = self.token2wav(emb)
        return audio

    @torch.no_grad()
    def decode_tokens(self, tokens, **kwargs):
        if tokens.dim() == 3:
            # If the user passes raw output from a codebook stack [num_quantizers, Batch, Time] 
            # we should dynamically squeeze it if it's singular.
            if tokens.size(0) == 1:
                tokens = tokens.squeeze(0)
            elif tokens.size(1) == 1:
                tokens = tokens.squeeze(1)
                
        assert tokens.dim() == 2, f"Expected 2D [Batch, Time] for decode_tokens input, got {tokens.dim()}D: {tokens.shape}"
        
        # vq2emb expects [Batch, Time, num_quantizers]
        tokens_for_vq = tokens.unsqueeze(-1)
        quantized = self.tokenizer.vq2emb(tokens_for_vq)
        audio = self.token2wav(quantized)
        return audio
