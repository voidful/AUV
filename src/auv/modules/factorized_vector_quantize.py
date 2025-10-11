import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils.parametrizations import weight_norm


class FactorizedVectorQuantize(nn.Module):
    def __init__(self, dim, codebook_size, codebook_dim, commitment, **kwargs):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment = commitment

        if dim != self.codebook_dim:
            self.in_proj = weight_norm(nn.Conv1d(dim, self.codebook_dim, kernel_size=1))
            self.out_proj = weight_norm(nn.Conv1d(self.codebook_dim, dim, kernel_size=1))
        else:
            self.in_proj = nn.Identity()
            self.out_proj = nn.Identity()

        self._codebook = nn.Embedding(codebook_size, self.codebook_dim)

    @property
    def codebook(self):
        return self._codebook

    def forward(self, z, codebook_partitions=None):
        """Quantized the input tensor using a fixed codebook and returns
        the corresponding codebook vectors

        Parameters
        ----------
        z: torch.Tensor[B x D x T]
        codebook_partitions: List[List[int]], optional
            List of codebook indices partitions, same for each quantizer
            (ex: [[0, 1024], [1024, 2048]] to allocate for z[0] and z[1])

        Returns
        -------
        z_q: torch.Tensor[B x D x T]
            Quantized continuous representation of input
        vq_loss: Tensor[B]
            Commitment loss to train encoder to predict vectors closer to codebook entries
            plus Codebook loss to update the codebook
        indices: torch.Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        z_e: torch.Tensor[B x D x T]
            Projected latents (continuous representation of input before quantization)
        """

        # Factorized codes project input into low-dimensional space
        z_e = self.in_proj(z)
        z_q, indices = self.decode_latents(z_e, codebook_partitions=codebook_partitions)

        # Compute commitment loss and codebook loss
        if self.training:
            commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2]) * self.commitment
            codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])
            commit_loss = commitment_loss + codebook_loss
        else:
            commit_loss = torch.zeros(z.shape[0], device=z.device)

        z_q = z_e + (z_q - z_e).detach()

        z_q = self.out_proj(z_q)

        return z_q, indices, commit_loss

    def vq2emb(self, indices, proj=True):
        z_q = self.decode_code(indices)
        if proj:
            z_q = self.out_proj(z_q)
        return z_q

    def get_emb(self):
        return self.codebook.weight

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents, codebook_partitions=None):
        B, D, T = latents.shape
        codebook = self.codebook.weight  # codebook: (N x D)
        encodings = rearrange(latents, "b d t -> (b t) d")

        # L2 normalize encodings and codebook
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance with codebook. If l2-normalized, equal to cosine distance
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )

        if codebook_partitions is not None:
            for i, (start, end) in enumerate(codebook_partitions):
                dist[i * T : (i + 1) * T, :start] = float("inf")
                dist[i * T : (i + 1) * T, end:] = float("inf")

        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=B)
        z_q = self.decode_code(indices)

        return z_q, indices
