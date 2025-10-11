import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    @torch.amp.autocast("cuda", enabled=False)
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @torch.amp.autocast("cuda", enabled=False)
    def apply_rotary_pos_emb(self, qk, cos, sin, position_ids):
        cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
        sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        qk_embed = (qk * cos) + (self.rotate_half(qk) * sin)
        return qk_embed


class MultiheadRMSNorm(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, p=2, dim=-1) * self.gamma


class Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 16,
        head_dim: int = 64,
        dropout: float = 0.0,
        qk_norm: bool = False,
        qk_norm_scale: float = 10.0,
        qkv_bias: bool = False,
        max_position_embeddings: int = 2048,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        dim_inner = num_heads * head_dim
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = MultiheadRMSNorm(head_dim, heads=num_heads)
            self.k_norm = MultiheadRMSNorm(head_dim, heads=num_heads)
        self.scale = qk_norm_scale if qk_norm else 1.0 / math.sqrt(head_dim)
        self.dropout = dropout

        self.to_qkv = nn.Linear(hidden_size, 3 * dim_inner, bias=qkv_bias)
        self.to_out = nn.Linear(dim_inner, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.rotary_embed = RotaryEmbedding(self.head_dim, max_position_embeddings=max_position_embeddings)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        batch_size, q_len, _ = hidden_states.size()
        q, k, v = self.to_qkv(hidden_states).chunk(3, dim=-1)
        q, k, v = map(lambda x: rearrange(x, "b n (h d) -> b h n d", h=self.num_heads), (q, k, v))
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        kv_seq_len = k.shape[-2]
        if past_key_values is not None:
            kv_seq_len += past_key_values[0].shape[-2]

        # position embeddings
        cos, sin = self.rotary_embed(k, seq_len=kv_seq_len)
        if position_ids is None:
            position_ids = torch.arange(q.shape[-2], dtype=torch.long, device=q.device).unsqueeze(0)
        q, k = map(lambda x: self.rotary_embed.apply_rotary_pos_emb(x, cos, sin, position_ids), (q, k))

        # deal history
        if past_key_values is not None:
            k = torch.cat([past_key_values[0], k], dim=2)
            v = torch.cat([past_key_values[1], v], dim=2)
        past_key_values = (k, v) if use_cache else None

        # attention mask
        if mask is not None:  # 0 for padding position
            if mask.size() != (batch_size, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, q_len, kv_seq_len)}, got {mask.size()}"
                )
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale,
        )

        attn_output = rearrange(attn_output, "b h n d -> b n (h d)")
        attn_output = self.to_out(attn_output)

        return attn_output, past_key_values


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.up_proj = nn.Linear(hidden_size, ffn_hidden_size)
        self.down_proj = nn.Linear(ffn_hidden_size, hidden_size)
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x):
        x = self.act_fn(self.up_proj(x))
        x = self.down_proj(x)
        return x


class ConformerConvolutionModule(nn.Module):
    def __init__(self, hidden_size, conv_kernel_size=31, causal=False):
        super().__init__()
        self.causal = causal
        self.conv_kernel_size = conv_kernel_size
        # norm before conv
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.pointwise_conv1 = nn.Conv1d(hidden_size, 2 * hidden_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.glu = nn.GLU(dim=1)
        # depth-wise conv
        if self.causal:
            padding = 0
            self.lorder = self.conv_kernel_size - 1
        else:
            assert (self.conv_kernel_size % 2) == 1, "conformer conv_kernel_size should be odd for SAME padding"
            padding = (self.conv_kernel_size - 1) // 2
            self.lorder = 0
        self.depthwise_conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=self.conv_kernel_size,
            stride=1,
            padding=padding,
            groups=hidden_size,
            bias=False,
        )
        self.conv_norm = nn.BatchNorm1d(hidden_size)
        self.activation = nn.GELU(approximate="tanh")
        # pointwise conv 2
        self.pointwise_conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, hidden_states, mask: Optional[torch.Tensor] = None, cache: Optional[torch.Tensor] = None):
        """
        hidden_states: [B, T, F]
        mask: [B, T]
        cache: [B, F, h], for depthwise-conv
        """

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)

        # pointwise-conv 1
        hidden_states = self.pointwise_conv1(hidden_states)
        hidden_states = self.glu(hidden_states)

        # mask pad & cache
        if mask is not None:
            mask = mask.unsqueeze(1)
            hidden_states = hidden_states.masked_fill(~mask, 0.0)
        prepend_len = 0
        if cache is not None:
            prepend_len = cache.shape[2]
            hidden_states = torch.cat([cache, hidden_states], dim=2)
        cache_len = self.lorder if self.lorder > 0 else (self.conv_kernel_size - 1) // 2
        new_cache = hidden_states[:, :, -cache_len:]

        # depthwise conv
        if self.lorder > 0:
            hidden_states = F.pad(hidden_states, (self.lorder, 0), "constant", 0.0)
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = hidden_states[:, :, prepend_len:]

        # batch norm & activation
        hidden_states = self.conv_norm(hidden_states)
        hidden_states = self.activation(hidden_states)

        # pointwise conv 2
        hidden_states = self.pointwise_conv2(hidden_states)
        if mask is not None:
            hidden_states = hidden_states.masked_fill(~mask, 0.0)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states, new_cache


class ConformerEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size=1024,
        ffn_mult=4,
        head_dim=64,
        num_heads=None,
        qk_norm=False,
        qk_norm_scale=10.0,
        conv_kernel_size=31,
        max_position_embeddings=4096,
        causal=False,
    ):
        super().__init__()
        embed_dim = hidden_size
        if num_heads is None:
            num_heads = embed_dim // head_dim

        # ffn 1
        self.ffn1_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn1 = FeedForward(embed_dim, hidden_size * ffn_mult)

        # self attention
        self.attn_layer_norm = nn.LayerNorm(embed_dim)
        self.attn = Attention(
            embed_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            qk_norm=qk_norm,
            qk_norm_scale=qk_norm_scale,
            max_position_embeddings=max_position_embeddings,
        )

        # conformer convolution
        self.conv_module = ConformerConvolutionModule(
            hidden_size=hidden_size, conv_kernel_size=conv_kernel_size, causal=causal
        )

        # ffn 2
        self.ffn2_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn2 = FeedForward(embed_dim, hidden_size * ffn_mult)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        mask_pad: Optional[torch.Tensor] = None,  # for conv module
        position_ids: Optional[torch.LongTensor] = None,
        caches: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        if caches is not None:  # should be tuple of (past_keys, past_values, cnn_caches)
            past_key_values, cnn_caches = caches[0:2], caches[2]
        else:
            past_key_values, cnn_caches = None, None

        # ffn1
        residual = hidden_states
        hidden_states = self.ffn1_layer_norm(hidden_states)
        hidden_states = self.ffn1(hidden_states)
        hidden_states = residual + hidden_states * 0.5

        # self attn
        residual = hidden_states
        hidden_states = self.attn_layer_norm(hidden_states)
        hidden_states, past_key_values = self.attn(
            hidden_states,
            mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # convolution  module
        residual = hidden_states
        hidden_states, cnn_caches = self.conv_module(hidden_states, mask=mask_pad, cache=cnn_caches)
        hidden_states = residual + hidden_states

        #  ffn 2
        residual = hidden_states
        hidden_states = self.ffn2_layer_norm(hidden_states)
        hidden_states = residual + 0.5 * self.ffn2(hidden_states)

        # out
        hidden_states = self.final_layer_norm(hidden_states)
        if use_cache:
            new_cache = past_key_values + (cnn_caches,)
        else:
            new_cache = None
        return hidden_states, new_cache


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers=12,
        hidden_size=1024,
        ffn_mult=4,
        head_dim=64,
        num_heads=None,
        qk_norm=False,
        qk_norm_scale=10.0,
        conv_kernel_size=31,
        max_position_embeddings=4096,
        causal=False,
        **kwargs,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # layers
        self.layers = nn.ModuleList(
            [
                ConformerEncoderLayer(
                    hidden_size=hidden_size,
                    ffn_mult=ffn_mult,
                    head_dim=head_dim,
                    num_heads=num_heads,
                    qk_norm=qk_norm,
                    qk_norm_scale=qk_norm_scale,
                    conv_kernel_size=conv_kernel_size,
                    max_position_embeddings=max_position_embeddings,
                    causal=causal,
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mask_pad: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        caches: Optional[List[Tuple[torch.Tensor]]] = None,
        output_hidden_states: bool = False,
        use_cache: bool = False,
    ):
        all_hidden_states = () if output_hidden_states else None
        new_caches = () if use_cache else None
        batch_size, seq_len, _, device = *hidden_states.shape, hidden_states.device
        seq_len_with_past, past_len = seq_len, 0
        if caches is not None:
            past_len = caches[0][0].shape[-2]
            seq_len_with_past += past_len

        # position id
        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)
        else:
            position_ids = position_ids.view(-1, seq_len).long()

        # attention mask
        if attention_mask is not None:
            if attention_mask.size() != (batch_size, seq_len, seq_len):
                raise ValueError(
                    f"given attn mask should be of size {(batch_size, seq_len, seq_len)}, got {attention_mask.size()}"
                )
            if past_len > 0:
                extra = torch.ones(batch_size, seq_len, past_len, dtype=attention_mask.dtype, device=device)
                attention_mask = torch.cat([extra, attention_mask], dim=2)
        if mask_pad is not None:
            if mask_pad.size() != (batch_size, seq_len):
                raise ValueError(f"given mask_pad should be of size {(batch_size, seq_len)}, got {mask_pad.size()}")

        # layer forward
        for i, layer in enumerate(self.layers):
            layer_cache = caches[i] if caches is not None else None
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states, cache = layer(
                hidden_states,
                attention_mask=attention_mask,
                mask_pad=mask_pad,
                position_ids=position_ids,
                caches=layer_cache,
                use_cache=use_cache,
            )
            if use_cache:
                new_caches = new_caches + (cache,)
        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple([v for v in [hidden_states, all_hidden_states, new_caches] if v is not None])
