import math

import numpy as np
import torch
from torch import Tensor
from jaxtyping import Bool, Float, Int
from einops import einsum, rearrange
import einx


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
        # r, c = token_ids.shape
        # vocab_size, emb_dim = self.weights.shape
        # embedding_weight = self.weights.index_select(0, token_ids.view(r * c)).view(r, c, emb_dim)
        embedding_weight = self.weights[token_ids]
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
        rope: Float[Tensor, "max_seq_len d_k d_k"] = torch.zeros((max_seq_len, d_k, d_k), device=device)
        for seq in range(max_seq_len):
            for dim in range(d_k // 2):
                t = seq / (theta ** (2 * dim / d_k))
                rope[seq, 2 * dim, 2 * dim] = math.cos(t)
                rope[seq, 2 * dim, 2 * dim + 1] = math.sin(t)
                rope[seq, 2 * dim + 1, 2 * dim] = -math.sin(t)
                rope[seq, 2 * dim + 1, 2 * dim + 1] = math.cos(t)
        self.register_buffer('rope', rope)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # print('token_positions.shape is very important', token_positions.shape)
        seq_len = x.shape[-2]
        x = rearrange(x, '... b s d -> ... s b d')
        result = einsum(x, self.rope[:seq_len], "... s b d, s d e -> ... s b e")
        result = rearrange(result, '... s b e -> ... b s e')
        position_result = result[..., token_positions, :]
        return position_result


def softmax(x, dim=-1):
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
        if dot.ndim > mask.ndim:
            mask = mask.view((1, ) * (dot.ndim - mask.ndim) + mask.shape)
        dot = dot.masked_fill(mask, -1e9)
    dot = softmax(dot, -1)
    result = einsum(dot, V, "... q_dim k_dim, ... k_dim d_v -> ... q_dim d_v")

    return result


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model, num_head, q_weight, k_weight, v_weight, o_weight, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.q_weight = q_weight
        self.k_weight = k_weight
        self.v_weight = v_weight
        self.o_weight = o_weight
        causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer('causal_mask', causal_mask)

    def forward(self, x):
        seq_len = x.shape[1]
        x_q = einsum(x, self.q_weight, "... seq_len d_in, d_k d_in -> ... seq_len d_k")
        x_k = einsum(x, self.k_weight, "... seq_len d_in, d_k d_in -> ... seq_len d_k")
        x_v = einsum(x, self.v_weight, "... seq_len d_in, d_v d_in -> ... seq_len d_v")
        x_q = rearrange(x_q, " ... seq_len (num_head d_dim) -> ... num_head seq_len d_dim", num_head=self.num_head)
        x_k = rearrange(x_k, " ... seq_len (num_head d_dim) -> ... num_head seq_len d_dim", num_head=self.num_head)
        x_v = rearrange(x_v, " ... seq_len (num_head d_dim) -> ... num_head seq_len d_dim", num_head=self.num_head)
        mask = self.causal_mask[:seq_len, :seq_len]
        mha = scaled_dot_product_attention(x_q, x_k, x_v, mask)
        mha = rearrange(mha, "... num_head q_dim d_dim -> ... q_dim (num_head d_dim)")
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
        self.rope = RotaryPositionalEmbedding(theta, d_model // num_head, max_seq_len)
        causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer('causal_mask', causal_mask)

    def forward(self, x, token_positions):
        bs, seq_len = x.shape[0], x.shape[1]
        x_q = einsum(x, self.q_weight, "... seq_len d_in, d_k d_in -> ... seq_len d_k")
        x_k = einsum(x, self.k_weight, "... seq_len d_in, d_k d_in -> ... seq_len d_k")
        x_v = einsum(x, self.v_weight, "... seq_len d_in, d_v d_in -> ... seq_len d_v")
        x_q = rearrange(x_q, " ... seq_len (num_head d_dim) -> ... num_head seq_len d_dim", num_head=self.num_head)
        x_k = rearrange(x_k, " ... seq_len (num_head d_dim) -> ... num_head seq_len d_dim", num_head=self.num_head)
        x_v = rearrange(x_v, " ... seq_len (num_head d_dim) -> ... num_head seq_len d_dim", num_head=self.num_head)
        x_q = self.rope(x_q, torch.squeeze(token_positions))
        x_k = self.rope(x_k, torch.squeeze(token_positions))
        mask = self.causal_mask[:seq_len, :seq_len]
        mha = scaled_dot_product_attention(x_q, x_k, x_v, mask)
        mha = rearrange(mha, "... num_head q_dim d_dim -> ... q_dim (num_head d_dim)")
        result = einsum(mha, self.o_weight, "... q_dim d_v, d_model d_v -> ... q_dim d_model")
        return result


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, weights):
        super().__init__()
        q_weight, k_weight, v_weight = [weights[f'attn.{s}_proj.weight'] for s in 'qkv']
        o_weight = weights['attn.output_proj.weight']
        W1, W2, W3 = [weights[f'ffn.w{i}.weight'] for i in [1, 2, 3]]
        lw1, lw2 = weights['ln1.weight'], weights['ln2.weight']
        self.mha_rope = MultiHeadSelfAttentionRope(d_model, num_heads, max_seq_len, theta, q_weight, k_weight, v_weight, o_weight)
        self.rms_norm1 = RMSNorm(d_model, weights=lw1)
        self.rms_norm2 = RMSNorm(d_model, weights=lw2)
        self.swi_glu = SwiGLU(d_model, d_ff, W1, W2, W3)

    def forward(self, x):
        token_pos = torch.arange(0, x.shape[-2], dtype=torch.int)
        x = x + self.mha_rope(self.rms_norm1(x), token_pos)
        return x + self.swi_glu(self.rms_norm2(x))


class TransformerLM(torch.nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, weights):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, weights['token_embeddings.weight'])
        self.final_norm = RMSNorm(d_model, weights=weights['ln_final.weight'])
        self.output_linear = Linear(vocab_size, d_model, weights['lm_head.weight'])
        block_list = []
        for num_layer in range(num_layers):
            tmp_weights = {}
            for s in 'qkv':
                tmp_weights[f'attn.{s}_proj.weight'] = weights[f'layers.{num_layer}.attn.{s}_proj.weight']
            tmp_weights['attn.output_proj.weight'] = weights[f'layers.{num_layer}.attn.output_proj.weight']
            for i in [1, 2, 3]:
                tmp_weights[f'ffn.w{i}.weight'] = weights[f'layers.{num_layer}.ffn.w{i}.weight']
            tmp_weights['ln1.weight'] = weights[f'layers.{num_layer}.ln1.weight']
            tmp_weights['ln2.weight'] = weights[f'layers.{num_layer}.ln2.weight']
            block_list.append(TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, tmp_weights))
        self.blocks = torch.nn.ModuleList(block_list)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        x = self.output_linear(x)
        # x = softmax(x)
        return x


def resource_accounting(vocab_size=50257, context_length=1024, num_layers=48, d_model=1600, num_heads=25, d_ff=6400):
    vocab_param = vocab_size * d_model
    block_param = 4 * d_model * d_model + 2 * d_model + 3 * d_model * d_ff
    trainable_parameter = vocab_param + num_layers * block_param + d_model + vocab_param
    print('trainable_parameter', trainable_parameter)

    in_project_flops = 2 * 3 * context_length * d_model * d_model
    rope_flops = 2 * 2 * num_heads * context_length * (d_model // num_heads) * (d_model // num_heads)
    att_flops = 2 * 2 * num_heads * context_length * context_length * (d_model // num_heads)
    out_project_flops = 2 * context_length * d_model * d_model
    norm_flops = 2 * context_length * d_model
    ffn_flops = 2 * 3 * context_length * d_model * d_ff

    total_flops = num_layers * (in_project_flops + rope_flops + att_flops + out_project_flops + norm_flops + ffn_flops)
    print(f'total_flops = {total_flops:.3E}')
    print(f'per_layer_flops = {total_flops/num_layers:.3E}')
    print(f'in_project_flops = {in_project_flops:.3E}')
    print(f'rope_flops = {rope_flops:.3E}', )
    print(f'att_flops = {att_flops:.3E}')
    print(f'out_project_flops = {out_project_flops:.3E}')
    print(f'norm_flops = {norm_flops:.3E}')
    print(f'ffn_flops = {ffn_flops:.3E}')


def test_resource_accounting():
    """
    vocab_size = 50257
    context_length = 1024
    num_layers = 48
    d_model = 1600
    num_heads = 25
    d_ff = 6400
    """
    print('-' * 20, 'small GPT2', '-' * 20)
    resource_accounting(num_layers=12, d_model=768, num_heads=12, d_ff=768*4)
    print('-' * 20, 'medium GPT2', '-' * 20)
    resource_accounting(num_layers=24, d_model=1024, num_heads=16, d_ff=1024*4)
    print('-' * 20, 'large GPT2', '-' * 20)
    resource_accounting(num_layers=36, d_model=1280, num_heads=20, d_ff=1280*4)
    print('-' * 20, 'large GPT2 XL', '-' * 20)
    resource_accounting()
    print('-' * 20, 'large GPT2 XL long_seq', '-' * 20)
    resource_accounting(context_length=16384)


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(inputs, targets) -> Float[Tensor, ""]:
        m_inputs = inputs - torch.max(inputs, -1, keepdim=True).values
        m_targets = m_inputs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        # p = torch.exp(m_targets) / torch.sum(torch.exp(m_inputs), -1)
        # result = torch.mean(-torch.log(p))
        result = torch.mean(torch.log(torch.sum(torch.exp(m_inputs), -1)) - m_targets)
        return result







