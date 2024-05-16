from random import random
from functools import partial, cache

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList

from beartype import beartype
from beartype.typing import Union, List, Optional, Callable, Tuple, Dict, Any

from einops import pack, unpack, repeat, reduce, rearrange
from einops.layers.torch import Rearrange, Reduce

from src.attend import Attend

# helpers

def exists(val):
    return val is not None

def xnor(x, y):
    """ (True, True) or (False, False) -> True """
    return not (x ^ y)

def divisible_by(num, den):
    return (num % den) == 0

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# tensor helpers

def l2norm(t, dim = -1):
    return F.normalize(t, dim = dim)

def pack_one(x, pattern):
    return pack([x], pattern)

def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]

def rotate_half(x):
    x1, x2 = rearrange(x, '... (d c) -> ... d c', c = 2).unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d c -> ... (d c)')

@autocast(enabled = False)
def apply_rotary_pos_emb(pos, t):
    return t * pos.cos() + rotate_half(t) * pos.sin()

# sync batchnorm

@cache
def get_is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1

def MaybeSyncBatchnorm2d(is_distributed = None):
    is_distributed = default(is_distributed, get_is_distributed())
    return nn.SyncBatchNorm if is_distributed else nn.BatchNorm2d

# channel rmsnorm

class RMSNorm(Module):
    def __init__(self, dim, affine = True):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if affine else 1.

    def forward(self, x):
        return l2norm(x) * self.gamma * self.scale

# sinusoidal positions

def posemb_sincos_1d(seq, dim, temperature = 10000, device = None, dtype = torch.float32):
    n = torch.arange(seq, device = device)
    omega = torch.arange(dim // 2, device = device) / (dim // 2 - 1)
    omega = 1. / (temperature ** omega)

    n = n[:, None] * omega[None, :]
    pos_emb = torch.cat((n.sin(), n.cos()), dim = 1)
    return pos_emb.type(dtype)

# helper classes

class Residual(Module):
    @beartype
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class FeedForward(Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.,
        adaptive_ln = False
    ):
        super().__init__()
        self.adaptive_ln = adaptive_ln

        inner_dim = int(dim * mult)
        self.norm = RMSNorm(dim, affine = not adaptive_ln)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
    ):
        x = self.norm(x)
        return self.net(x)


class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 32,
        dropout = 0.,
        window_size = 7,
        num_mem_kv = 4,
        flash = True
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dim_inner = dim_head * heads

        self.norm = RMSNorm(dim)
        self.heads = heads

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)

        self.to_v_gates = nn.Sequential(
            nn.Linear(dim, self.heads),
            nn.Sigmoid(),
            Rearrange('b n h -> b h n 1')
        )

        self.attend = Attend(
            causal = False,
            dropout = dropout,
            flash = flash
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        rotary_emb = None
    ):
        batch, height, width, window_height, window_width, _, device, h = *x.shape, x.device, self.heads

        x = self.norm(x)

        # flatten

        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')

        # project for queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        g = self.to_v_gates(x)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # 2d rotary

        if exists(rotary_emb):
            q = apply_rotary_pos_emb(rotary_emb, q)
            k = apply_rotary_pos_emb(rotary_emb, k)

        # attention

        out = self.attend(q, k, v)

        # gate values per head, allow for attending to nothing

        out = out * g

        # merge heads

        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1 = window_height, w2 = window_width)

        # combine heads out

        out = self.to_out(out)
        return rearrange(out, '(b x y) ... -> b x y ...', x = height, y = width)

class TransformerAttention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        dim_context = None,
        heads = 8,
        num_mem_kv = 4,
        norm_context = False,
        adaptive_ln = False,
        dropout = 0.1,
        flash = True,
        causal = False
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.heads = heads
        inner_dim = dim_head * heads

        dim_context = default(dim_context, dim)

        self.adaptive_ln = adaptive_ln
        self.norm = RMSNorm(dim, affine = not adaptive_ln)

        self.context_norm = RMSNorm(dim_context) if norm_context else None

        self.attn_dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim_context, inner_dim * 2, bias = False)

        self.num_mem_kv = num_mem_kv
        self.mem_kv = None
        if num_mem_kv > 0:
            self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))

        self.attend = Attend(
            dropout = dropout,
            flash = flash,
            causal = causal
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        context = None,
        mask = None,
        attn_mask = None,
        cache: Optional[Tensor] = None,
        return_cache = False
    ):
        b = x.shape[0]

        assert xnor(exists(context), exists(self.context_norm))

        if exists(context):
            context = self.context_norm(context)

        kv_input = default(context, x)

        x = self.norm(x)

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        if exists(cache):
            ck, cv = cache
            k = torch.cat((ck, k), dim = -2)
            v = torch.cat((cv, v), dim = -2)

        new_kv_cache = torch.stack((k, v))

        if exists(self.mem_kv):
            mk, mv = map(lambda t: repeat(t, '... -> b ...', b = b), self.mem_kv)

            k = torch.cat((mk, k), dim = -2)
            v = torch.cat((mv, v), dim = -2)

            if exists(mask):
                mask = F.pad(mask, (self.num_mem_kv, 0), value = True)

            if exists(attn_mask):
                attn_mask = F.pad(attn_mask, (self.num_mem_kv, 0), value = True)

        out = self.attend(q, k, v, mask = mask, attn_mask = attn_mask)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        if not return_cache:
            return out

        return out, new_kv_cache

class Transformer(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        depth = 6,
        attn_dropout = 0.,
        ff_dropout = 0.,
        adaptive_ln = False,
        flash_attn = True,
        cross_attend = False,
        causal = False,
        final_norm = True
    ):
        super().__init__()
        self.layers = ModuleList([])

        attn_kwargs = dict(
            dim = dim,
            heads = heads,
            dim_head = dim_head,
            dropout = attn_dropout,
            flash = flash_attn
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for _ in range(depth):
            self.layers.append(ModuleList([
                TransformerAttention(**attn_kwargs, causal = causal, adaptive_ln = adaptive_ln, norm_context = False),
                TransformerAttention(**attn_kwargs, norm_context = True) if cross_attend else None,
                FeedForward(dim = dim, dropout = ff_dropout, adaptive_ln = adaptive_ln)
            ]))

        self.norm = RMSNorm(dim) if final_norm else nn.Identity()

    @beartype
    def forward(
        self,
        x,
        attn_mask = None,
        context: Optional[Tensor] = None,
        cache: Optional[Tensor] = None,
        return_cache = False
    ):
        has_cache = exists(cache)

        if has_cache:
            x_prev, x = x[..., :-1, :], x[..., -1:, :]

        cache = iter(default(cache, []))

        new_caches = []

        for attn, maybe_cross_attn, ff in self.layers:
            attn_out, new_cache = attn(
                x,
                attn_mask = attn_mask,
                return_cache = True,
                cache = next(cache, None)
            )

            new_caches.append(new_cache)

            x = x + attn_out

            if exists(maybe_cross_attn):
                assert exists(context)
                x = maybe_cross_attn(x, context = context) + x

            x = ff(x) + x

        new_caches = torch.stack(new_caches)

        if has_cache:
            x = torch.cat((x_prev, x), dim = -2)

        out = self.norm(x)

        if not return_cache:
            return out

        return out, new_caches

# token learner module

class TokenLearner(Module):
    """
    https://arxiv.org/abs/2106.11297
    using the 1.1 version with the MLP (2 dense layers with gelu) for generating attention map
    """

    def __init__(
        self,
        *,
        dim,
        ff_mult = 2,
        num_output_tokens = 8,
        num_layers = 2
    ):
        super().__init__()
        inner_dim = dim * ff_mult * num_output_tokens
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_output_tokens = num_output_tokens
        self.net = nn.Sequential(
            nn.Conv2d(dim * num_output_tokens, inner_dim, 1, groups = num_output_tokens),
            nn.GELU(),
            nn.Conv2d(inner_dim, num_output_tokens, 1, groups = num_output_tokens),
        )

    def forward(self, x):
        x, ps = pack_one(x, '* c h w')
        x = repeat(x, 'b c h w -> b (g c) h w', g = self.num_output_tokens)
        attn = self.net(x)

        attn = rearrange(attn, 'b g h w -> b 1 g h w')
        x = rearrange(x, 'b (g c) h w -> b c g h w', g = self.num_output_tokens)

        x = reduce(x * attn, 'b c g h w -> b c g', 'mean')
        x = unpack_one(x, ps, '* c n')
        return x

# Dueling heads for Q value

class DuelingHead(Module):
    def __init__(
        self,
        dim,
        expansion_factor = 2,
        action_bins = 256
    ):
        super().__init__()
        dim_hidden = dim * expansion_factor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.stem = nn.Sequential(
            nn.Linear(dim, dim_hidden),
            nn.SiLU()
        )

        self.to_values = nn.Sequential(
            nn.Linear(dim_hidden, 1)
        )

        self.to_advantages = nn.Sequential(
            nn.Linear(dim_hidden, action_bins)
        )

    def forward(self, x):
        x = self.stem(x)

        advantages = self.to_advantages(x)
        advantages = advantages - reduce(advantages, '... a -> ... 1', 'mean')

        values = self.to_values(x)

        q_values = values + advantages
        return q_values.sigmoid()

# Q head modules, for either single or multiple actions



class QHeadMultipleActions(Module):
    def __init__(
        self,
        dim,
        *,
        num_actions = 4,
        action_bins = 256,
        attn_depth = 2,
        attn_dim_head = 32,
        attn_heads = 8,
        dueling = False,
        weight_tie_action_bin_embed = False
    ):
        super().__init__()
        self.num_actions = num_actions
        self.action_bins = action_bins

        self.action_bin_embeddings = nn.Parameter(torch.zeros(num_actions, action_bins, dim))
        nn.init.normal_(self.action_bin_embeddings, std = 0.02)

        self.to_q_values = None
        if not weight_tie_action_bin_embed:
            self.to_q_values = nn.Linear(dim, action_bins)

        self.transformer = Transformer(
            dim = dim,
            depth = attn_depth,
            dim_head = attn_dim_head,
            heads = attn_heads,
            cross_attend = True,
            adaptive_ln = False,
            causal = True,
            final_norm = True
        )

        self.final_norm = RMSNorm(dim)

        self.dueling = dueling
        if dueling:
            self.to_values = nn.Parameter(torch.zeros(num_actions, dim))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

#    @property
#    def device(self):
#        print(self.action_bin_embeddings.device)
#        return self.action_bin_embeddings.device

    def maybe_append_actions(self, sos_tokens, actions: Optional[Tensor] = None):
        if not exists(actions):
            return sos_tokens

        batch, num_actions = actions.shape
        action_embeddings = self.action_bin_embeddings[:num_actions]

        action_embeddings = repeat(action_embeddings, 'n a d -> b n a d', b = batch)
        past_action_bins = repeat(actions, 'b n -> b n 1 d', d = action_embeddings.shape[-1])
        past_action_bins = past_action_bins.type(torch.int64)
        bin_embeddings = action_embeddings.gather(-2, past_action_bins)
        bin_embeddings = rearrange(bin_embeddings, 'b n 1 d -> b n d')

        tokens, _ = pack((sos_tokens, bin_embeddings), 'b * d')
        tokens = tokens[:, :self.num_actions] # last action bin not needed for the proposed q-learning
        return tokens

    def get_q_values(self, embed):
        num_actions = embed.shape[-2]

        if exists(self.to_q_values):
            logits = self.to_q_values(embed)
        else:
            # each token predicts next action bin
            action_bin_embeddings = self.action_bin_embeddings[:num_actions]
            action_bin_embeddings = torch.roll(action_bin_embeddings, shifts = -1, dims = 1)
            logits = einsum('b n d, n a d -> b n a', embed, action_bin_embeddings)

        if self.dueling:
            advantages = logits
            values = einsum('b n d, n d -> b n', embed, self.to_values[:num_actions])
            values = rearrange(values, 'b n -> b n 1')

            q_values = values + (advantages - reduce(advantages, '... a -> ... 1', 'mean'))
        else:
            q_values = logits

        return q_values.sigmoid()

    @torch.no_grad()
    def get_optimal_actions(
        self,
        encoded_state,
        return_q_values = False,
        actions: Optional[Tensor] = None,
        prob_random_action: float = 0.5,
        **kwargs
    ):
        assert 0. <= prob_random_action <= 1.
        batch = encoded_state.shape[0]

        sos_token = reduce(encoded_state, 'b ... d -> b 1 d', 'mean')
        tokens = self.maybe_append_actions(sos_token, actions = actions)

        action_bins = []
        cache = None

        for action_idx in range(self.num_actions):

            embed, cache = self.transformer(
                tokens,
                context = encoded_state,
                cache = cache,
                return_cache = True
            )

            last_embed = embed[:, action_idx]
            bin_embeddings = self.action_bin_embeddings[action_idx]

            q_values = einsum('b d, a d -> b a', last_embed, bin_embeddings)

            selected_action_bins = q_values.argmax(dim = -1)

            next_action_embed = bin_embeddings[selected_action_bins]

            tokens, _ = pack((tokens, next_action_embed), 'b * d')

            action_bins.append(selected_action_bins)

        action_bins = torch.stack(action_bins, dim = -1)

        if not return_q_values:
            return action_bins

        all_q_values = self.get_q_values(embed)
        return action_bins, all_q_values

    def forward(
        self,
        encoded_state: Tensor,
        actions: Optional[Tensor] = None
    ):
        """
        einops
        b - batch
        n - number of actions
        a - action bins
        d - dimension
        """

        # this is the scheme many hierarchical transformer papers do

        sos_token = reduce(encoded_state, 'b ... d -> b 1 d', 'mean')

        tokens = self.maybe_append_actions(sos_token, actions = actions)

        embed = self.transformer(tokens, context = encoded_state)

        return self.get_q_values(embed)

# Robotic Transformer

class MLPwithEmbedding(nn.Module):
    def __init__(self, state_dim, attend_dim, hidden_dim):
        super().__init__()
        # MLP layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)  # First fully connected layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, attend_dim)  # Output layer with the same dimension as input

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Activation function after first fully connected layer
        x = self.fc2(x)          # Output layer, could add activation if needed
        return x


class QRoboticTransformer(Module):

    @beartype
    def __init__(
        self,
        *,
        state_dim = 39,
        num_actions = 8,
        action_bins = 256,
        depth = 6,
        heads = 8,
        dim_head = 64,
        token_learner_ff_mult = 2,
        token_learner_num_layers = 2,
        token_learner_num_output_tokens = 8,
        dueling = False,                       # https://arxiv.org/abs/1511.06581
        flash_attn = True,
        q_head_attn_kwargs: dict = dict(
            attn_heads = 8,
            attn_dim_head = 64,
            attn_depth = 2
        ),
        weight_tie_action_bin_embed = True      # when projecting to action bin Q values, whether to weight tie to original embeddings
    ):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        # q-transformer related action embeddings

        assert num_actions >= 1

        self.num_actions = num_actions
        self.is_single_action = num_actions == 1
        self.action_bins = action_bins
        attend_dim = action_bins*2
        self.num_learned_tokens = token_learner_num_output_tokens

        # Embedding
        self.embedding_layer = MLPwithEmbedding(state_dim, attend_dim, 512)
        # Transformer
        self.transformer_depth = depth
        self.transformer = Transformer(
            dim = attend_dim,
            dim_head = dim_head,
            heads = heads,
            depth = depth,
            flash_attn = flash_attn,
            adaptive_ln = False,
            final_norm = True
        )

        self.q_head = QHeadMultipleActions(
            attend_dim,
            action_bins = action_bins,
            dueling = dueling,
            weight_tie_action_bin_embed = weight_tie_action_bin_embed,
            **q_head_attn_kwargs
        )


    @torch.no_grad()
    def get_optimal_actions(
        self,
        *args,
        return_q_values = False,
        actions: Optional[Tensor] = None,
        **kwargs
    ):
        encoded_state = self.encode_state(*args, **kwargs)
        return self.q_head.get_optimal_actions(encoded_state, return_q_values = return_q_values, actions = actions)

    def get_actions(
        self,
        state,
        *args,
        prob_random_action = 0.,  # otherwise known as epsilon in RL
        **kwargs,
    ):
        batch_size = state.shape[0]
        assert 0. <= prob_random_action <= 1.

        #if random() < prob_random_action:
        #    return self.get_random_actions(batch_size = batch_size)

        return self.get_optimal_actions(state, *args, **kwargs)

    def encode_state(
        self,
        state: Tensor,
    ):
        if state.ndim < 3:
            state = state.unsqueeze(0)

        batch_size, num_states, state_shape = state.shape
        device = state.device
        
        # Apply MLP to input state
        learned_tokens = self.embedding_layer(state)  # shape: (batch_size, hidden_dim)
        
        # Positional embedding
        pos_emb = posemb_sincos_1d(num_states, learned_tokens.shape[-1], device=device, dtype=learned_tokens.dtype)
        #learned_tokens = repeat(learned_tokens.squeeze(), 'b d -> b n d', n=self.num_learned_tokens) + pos_emb
        learned_tokens = learned_tokens + repeat(pos_emb, 'n d -> (n r) d', r = self.num_learned_tokens)
        
        # Causal attention mask
        attn_mask = ~torch.ones((num_states, num_states), dtype=torch.bool, device=device).triu(1)
        attn_mask = repeat(attn_mask, 'i j -> (i r1) (j r2)', r1 = self.num_learned_tokens, r2 = self.num_learned_tokens)
        
        # Transformer encoding
        # learned_tokens = rearrange(learned_tokens, 'b n d -> n b d')
        attended_tokens = self.transformer(learned_tokens, attn_mask=attn_mask)
        
        return attended_tokens

    def forward(
        self,
        state: Tensor,
        actions: Optional[Tensor] = None,
    ):

        # just auto-move inputs to the same device as robotic transformer

        state = state.to(self.device)

        if exists(actions):
            actions = actions.to(self.device)

        # encoding state

        encoded_state = self.encode_state(
            state = state
        )

        # head that returns the q values
        # supporting both single and multiple actions

        if self.is_single_action:
            assert not exists(actions), 'actions should not be passed in for single action robotic transformer'

            q_values = self.q_head(encoded_state)
        else:
            q_values = self.q_head(encoded_state, actions = actions)

        return q_values
