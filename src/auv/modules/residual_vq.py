import torch
from torch import nn

from auv.modules.factorized_vector_quantize import FactorizedVectorQuantize


REGISTERED_VQ_TYPE = {
    "fvq": FactorizedVectorQuantize,
}


class ResidualVQ(nn.Module):
    def __init__(
        self,
        *,
        vq_type,
        num_quantizers,
        codebook_size,
        **kwargs,
    ):
        super().__init__()
        try:
            VQ = REGISTERED_VQ_TYPE[vq_type]
        except KeyError:
            raise NotImplementedError(f"{vq_type} is not implemented.")

        if isinstance(codebook_size, int):
            codebook_size = [codebook_size] * num_quantizers
        self.quantizers = nn.ModuleList([VQ(codebook_size=size, **kwargs) for size in codebook_size])

        self.num_quantizers = num_quantizers

    def forward(self, x, codebook_partitions=None):
        quantized_out = 0.0
        residual = x

        all_losses = []
        all_indices = []

        for idx, quantizer in enumerate(self.quantizers):
            z_q_i, indices_i, vq_loss_i = quantizer(
                residual,
                codebook_partitions=codebook_partitions,
            )
            quantized_out = quantized_out + z_q_i
            residual = residual - z_q_i

            vq_loss_i = vq_loss_i.mean()

            all_indices.append(indices_i)
            all_losses.append(vq_loss_i)

        all_losses, all_indices = map(torch.stack, (all_losses, all_indices))
        return quantized_out, all_indices, all_losses

    def vq2emb(self, vq, proj=True):
        # [B, T, num_quantizers]
        quantized_out = 0.0
        for idx, quantizer in enumerate(self.quantizers):
            z_q_i = quantizer.vq2emb(vq[:, :, idx], proj=proj)
            quantized_out = quantized_out + z_q_i
        return quantized_out

    def get_emb(self):
        embs = []
        for idx, quantizer in enumerate(self.quantizers):
            embs.append(quantizer.get_emb())
        return embs
