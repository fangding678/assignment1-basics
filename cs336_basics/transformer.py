import math

import torch
from torch import Tensor
from jaxtyping import Bool, Float, Int
from einops import einsum, rearrange


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, weights: Float[Tensor, " d_out d_in"] | None = None, device=None, dtype=None):
        super().__init__()
        if weights is None:
            std1 = math.sqrt(2. / (in_features + out_features))
            weights = torch.nn.Parameter(torch.empty((in_features, out_features), device=device, dtype=dtype))
            self.weights = torch.nn.init.trunc_normal_(weights, mean=0, std=std1, a=-3*std1, b=3*std1)
        else:
            self.weights = torch.nn.Parameter(weights)
        self.device = device
        self.dtype = dtype
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = einsum(x, self.weights, "... d_in, d_out d_in -> ... d_out")
        return y


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, weights: Float[Tensor, " d_out d_in"] | None = None, device=None, dtype=None):
        super().__init__()
        if weights is None:
            weights = torch.nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
            self.weights = torch.nn.init.trunc_normal_(weights, mean=0, std=1, a=-3, b=3)
        else:
            self.weights = torch.nn.Parameter(weights)
        pass

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        r, c = token_ids.shape
        vocab_size, emb_dim = self.weights.shape
        embedding_weight = self.weights.index_select(0, token_ids.view(r * c)).view(r, c, emb_dim)
        return embedding_weight


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, weights: Float[Tensor, "d_out"] | None = None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        if weights is None:
            self.g = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        else:
            self.g = torch.nn.Parameter(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        result = x / torch.sqrt(torch.unsqueeze(torch.square(x).sum(-1) / self.d_model, -1) + self.eps) * self.g
        return result.to(in_dtype)


def silu(x):
    return x * torch.sigmoid(x)


class GLU(torch.nn.Module):
    def __init__(self, w1_weight: Float[Tensor, " d_ff d_model"], w2_weight: Float[Tensor, " d_ff d_model"]):
        super().__init__()
        self.W1 = w1_weight
        self.W2 = w2_weight

    def forward(self, x):
        x1 = einsum(x, self.W1, "... d_model, d_ff d_model -> ... d_ff")
        x2 = einsum(x, self.W2, "... d_model, d_ff d_model -> ... d_ff")
        return silu(x1) * x2


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, d_ff, w1_weight: Float[Tensor, " d_ff d_model"],
                 w2_weight: Float[Tensor, " d_model d_ff"], w3_weight: Float[Tensor, " d_ff d_model"]):
        super().__init__()
        self.W1 = w1_weight
        self.W2 = w2_weight
        self.W3 = w3_weight

    def forward(self, x):
        x1 = einsum(x, self.W1, "... d_model, d_ff d_model -> ... d_ff")
        x3 = einsum(x, self.W3, "... d_model, d_ff d_model -> ... d_ff")
        x2 = silu(x1) * x3
        result = einsum(x2, self.W2, "... d_ff, d_model d_ff -> ... d_model")
        return result


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.rope: Float[Tensor, "max_seq_len d_k d_k"] = torch.zeros((max_seq_len, d_k, d_k), device=device)
        for seq in range(max_seq_len):
            for dim in range(d_k // 2):
                t = seq / (theta ** (2 * dim / d_k))
                self.rope[seq, 2 * dim, 2 * dim] = math.cos(t)
                self.rope[seq, 2 * dim, 2 * dim + 1] = math.sin(t)
                self.rope[seq, 2 * dim + 1, 2 * dim] = -math.sin(t)
                self.rope[seq, 2 * dim + 1, 2 * dim + 1] = math.cos(t)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        # result = x.transpose(-2, -3) @ self.rope[:seq_len]
        # result = torch.matmul(x.transpose(-2, -3), self.rope[:seq_len])
        result = einsum(x.transpose(-2, -3), self.rope[:seq_len], "... s b d, s d e -> ... s b e")
        result = result.transpose(-2, -3)
        # position_result = result.index_select(1, token_positions)
        position_result = result[..., token_positions, :]
        return position_result


def softmax(x, dim):
    mx = torch.max(x, dim, keepdim=True).values
    xt = torch.exp(x - mx)
    result = xt / torch.sum(xt, dim, keepdim=True)
    return result


def scaled_dot_product_attention(Q: Float[Tensor, " ... q_dim d_k"], K: Float[Tensor, " ... k_dim d_k"],
                                 V: Float[Tensor, " ... k_dim d_v"],
                                 mask: Bool[Tensor, " ... q_dim k_dim"] | None = None) -> torch.Tensor:
    dim = Q.shape[-1]
    dot = einsum(Q, K, "... q_dim d_k,  ... k_dim d_k -> ... q_dim k_dim") / math.sqrt(dim)
    if mask is not None:
        dot = dot.masked_fill(~mask, -1e9)
    dot = softmax(dot, -1)
    result = einsum(dot, V, "... q_dim k_dim, ... k_dim d_v -> ... q_dim d_v")

    return result

'''
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
'''


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model, num_head, q_weight, k_weight, v_weight, o_weight, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.q_weight = q_weight
        self.k_weight = k_weight
        self.v_weight = v_weight
        self.o_weight = o_weight
        self.causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)

    def forward(self, x):
        seq_len = x.shape[1]
        x_q = einsum(x, self.q_weight, "... seq_len d_in, d_k d_in -> ... seq_len d_k")
        x_k = einsum(x, self.k_weight, "... seq_len d_in, d_k d_in -> ... seq_len d_k")
        x_v = einsum(x, self.q_weight, "... seq_len d_in, d_v d_in -> ... seq_len d_v")
        x_q = rearrange(x_q, " ... seq_len (num_head d_dim) -> ... num_head seq_len d_dim", num_head=self.num_head)
        x_k = rearrange(x_k, " ... seq_len (num_head d_dim) -> ... num_head seq_len d_dim", num_head=self.num_head)
        x_v = rearrange(x_v, " ... seq_len (num_head d_dim) -> ... num_head seq_len d_dim", num_head=self.num_head)
        mask = self.causal_mask[:seq_len, :seq_len]
        mha = scaled_dot_product_attention(x_q, x_k, x_v, mask)
        mha = rearrange(mha, "... num_head q_dim d_dim -> ... q_dim (num_head d_dim)")
        print('\n\nmha.shape', mha.shape)
        print('self.o_weight.shape', self.o_weight.shape)
        result = einsum(mha, self.o_weight, "... q_dim d_v, d_model d_v -> ... q_dim d_model")
        return result


class MultiHeadSelfAttentionRope(torch.nn.Module):
    def __init__(self, d_model, num_head, max_seq_len, theta, q_weight, k_weight, v_weight, o_weight):
        super().__init__()
        self.num_head = num_head
        self.q_weight = q_weight
        self.k_weight = k_weight
        self.v_weight = v_weight
        self.o_weight = o_weight
        self.rope = RotaryPositionalEmbedding(theta, d_model, max_seq_len)

    def forward(self, x, token_positions):
        bs, seq_len = x.shape[0], x.shape[1]
        x_q = einsum(x, self.q_weight, "... seq_len d_in, d_k d_in -> ... seq_len d_k")
        x_k = einsum(x, self.k_weight, "... seq_len d_in, d_k d_in -> ... seq_len d_k")
        x_v = einsum(x, self.q_weight, "... seq_len d_in, d_v d_in -> ... seq_len d_v")
        x_q = self.rope(x_q, token_positions)
        # x_k = self.rope(x_k, token_positions)
        x_q = rearrange(x_q, " ... seq_len (num_head d_dim) -> ... num_head seq_len d_dim", num_head=self.num_head)
        x_k = rearrange(x_k, " ... seq_len (num_head d_dim) -> ... num_head seq_len d_dim", num_head=self.num_head)
        x_v = rearrange(x_v, " ... seq_len (num_head d_dim) -> ... num_head seq_len d_dim", num_head=self.num_head)
        mask = torch.triu(torch.ones(bs, seq_len, seq_len, dtype=torch.bool), diagonal=1)
        mha = scaled_dot_product_attention(x_q, x_k, x_v, mask)
        mha = rearrange(mha, "... num_head q_dim d_dim -> ... q_dim (num_head d_dim)")
        print('\n\nmha.shape', mha.shape)
        print('self.o_weight.shape', self.o_weight.shape)
        result = einsum(mha, self.o_weight, "... q_dim d_v, d_model d_v -> ... q_dim d_model")
        return result





