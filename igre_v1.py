# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from modeling_jamba import JambaModel
from configuration_jamba import JambaConfig
import mmfreelm
from mmfreelm.ops.fusedbitnet import FusedBitLinear as BitLinear

# %%
from mamba_ssm.models.mixer_seq_simple import MixerModel, create_block, _init_weights

# %% [markdown]
# # IGRE
# <ins>I</ins>nter<ins>g</ins>alactic <ins>R</ins>easoning <ins>E</ins>ngine

USE_BITLINEAR = False
DEEP_COPY_DECODER = False
import copy

# %%
class iEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_vocab: int,
    ):
        super(iEncoder, self).__init__()
        n_layer = 16
        self.d_model = d_model
        self.mixer = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            vocab_size=n_vocab + 2,
        )

        initializer_cfg = None

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

        self.embedding_weight = self.mixer.embedding.weight

    def forward(self, x, output_pooled_state=False):
        x = self.mixer(x)
        if output_pooled_state:
            s = x.mean(dim=1)
            return x, s
        return x




class IEncoderPretrainedHGRN(nn.Module):
    def __init__(self, hgrn_model):
        super(IEncoderPretrainedHGRN, self).__init__()
        self.vocab_size = hgrn_model.embeddings.weight.shape[0]
        self.d_model = hgrn_model.embeddings.weight.shape[1]
        self.hgrn_model = hgrn_model
        self.embedding_weight = self.hgrn_model.embeddings.weight

    def forward(self, x, output_pooled_state=False):
        x = self.hgrn_model(x, output_hidden_states=True).last_hidden_state
        if output_pooled_state:
            s = x.mean(dim=1)
            return x, s
        return x



class IEncoderBased(nn.Module):
    def __init__(self, based_model):
        super(IEncoderBased, self).__init__()
        self.vocab_size = based_model.embeddings.word_embeddings.weight.shape[0]
        self.d_model = based_model.embeddings.word_embeddings.weight.shape[1]
        self.based_model = based_model
        self.embedding_weight = self.based_model.embeddings.word_embeddings.weight

    def forward(self, x, output_pooled_state=False):
        x = self.based_model(x)
        if output_pooled_state:
            s = x.mean(dim=1)
            return x, s
        return x


# %%
class iOptionGenProbablyWrong(nn.Module):
    def __init__(
        self,
        d_model: int,
    ):
        super(iOptionGenProbablyWrong, self).__init__()
        c_channels = 5
        self.c_embedding = nn.Embedding(c_channels, d_model)
        self.fc_mu = (
            nn.Linear(d_model * 2, d_model)
            if not USE_BITLINEAR
            else BitLinear(d_model * 2, d_model)
        )
        self.fc_logvar = (
            nn.Linear(d_model * 2, d_model)
            if not USE_BITLINEAR
            else BitLinear(d_model * 2, d_model)
        )

    def forward(self, x, c):
        """c is a tensor of shape (batch_size)"""
        c = self.c_embedding(c)
        x = torch.cat([x, c], dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


# %%
from typing import Callable


# %%
def inv_softplus(bias: float | torch.Tensor) -> float | torch.Tensor:
    """Inverse softplus function.

    Args:
        bias (float or tensor): the value to be softplus-inverted.
    """
    is_tensor = True
    if not isinstance(bias, torch.Tensor):
        is_tensor = False
        bias = torch.tensor(bias)
    out = bias.expm1().clamp_min(1e-6).log()
    if not is_tensor and out.numel() == 1:
        return out.item()
    return out


class biased_softplus(nn.Module):
    """A biased softplus module.

    The bias indicates the value that is to be returned when a zero-tensor is
    passed through the transform.

    Args:
        bias (scalar): 'bias' of the softplus transform. If bias=1.0, then a _bias shift will be computed such that
            softplus(0.0 + _bias) = bias.
        min_val (scalar): minimum value of the transform.
            default: 0.1
    """

    def __init__(self, bias: float, min_val: float = 0.01) -> None:
        super().__init__()
        self.bias = inv_softplus(bias - min_val)
        self.min_val = min_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(x + self.bias) + self.min_val


# %%
def mappings(key: str) -> Callable:
    """Given an input string, returns a surjective function f(x): R -> R^+.

    Args:
        key (str): one of "softplus", "exp", "relu", "expln",
            or "biased_softplus". If the key beggins with "biased_softplus",
            then it needs to take the following form:
            ```"biased_softplus_{bias}"``` where ```bias``` can be converted to a floating point number that will be used to bias the softplus function.
            Alternatively, the ```"biased_softplus_{bias}_{min_val}"``` syntax can be used. In that case, the additional ```min_val``` term is a floating point
            number that will be used to encode the minimum value of the softplus transform.
            In practice, the equation used is softplus(x + bias) + min_val, where bias and min_val are values computed such that the conditions above are met.

    Returns:
         a Callable

    """
    _mappings: dict[str, Callable] = {
        "softplus": torch.nn.functional.softplus,
        "exp": torch.exp,
        "relu": torch.relu,
        "biased_softplus": biased_softplus(1.0),
    }
    if key in _mappings:
        return _mappings[key]
    elif key.startswith("biased_softplus"):
        stripped_key = key.split("_")
        if len(stripped_key) == 3:
            return biased_softplus(float(stripped_key[-1]))
        elif len(stripped_key) == 4:
            return biased_softplus(
                float(stripped_key[-2]), min_val=float(stripped_key[-1])
            )
        else:
            raise ValueError(f"Invalid number of args in  {key}")

    else:
        raise NotImplementedError(f"Unknown mapping {key}")


# %%
class NormalParamExtractor(nn.Module):
    """A non-parametric nn.Module that splits its input into loc and scale parameters.

    The scale parameters are mapped onto positive values using the specified ``scale_mapping``.

    Args:
        scale_mapping (str, optional): positive mapping function to be used with the std.
            default = "biased_softplus_1.0" (i.e. softplus map with bias such that fn(0.0) = 1.0)
            choices: "softplus", "exp", "relu", "biased_softplus_1";
        scale_lb (Number, optional): The minimum value that the variance can take. Default is 1e-4.

    Examples:
        >>> import torch
        >>> from tensordict.nn.distributions import NormalParamExtractor
        >>> from torch import nn
        >>> module = nn.Linear(3, 4)
        >>> normal_params = NormalParamExtractor()
        >>> tensor = torch.randn(3)
        >>> loc, scale = normal_params(module(tensor))
        >>> print(loc.shape, scale.shape)
        torch.Size([2]) torch.Size([2])
        >>> assert (scale > 0).all()
        >>> # with modules that return more than one tensor
        >>> module = nn.LSTM(3, 4)
        >>> tensor = torch.randn(4, 2, 3)
        >>> loc, scale, others = normal_params(*module(tensor))
        >>> print(loc.shape, scale.shape)
        torch.Size([4, 2, 2]) torch.Size([4, 2, 2])
        >>> assert (scale > 0).all()

    """

    def __init__(
        self,
        scale_mapping: str = "biased_softplus_1.0",
        scale_lb=1e-4,
    ) -> None:
        super().__init__()
        self.scale_mapping = scale_mapping
        self.scale_lb = scale_lb

    def forward(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        tensor, *others = tensors
        loc, scale = tensor.chunk(2, -1)
        scale = mappings(self.scale_mapping)(scale).clamp_min(self.scale_lb)
        return (loc, scale, *others)


# %%
class iOptionGen(nn.Module):
    def __init__(
        self,
        d_model: int,
    ):
        super(iOptionGen, self).__init__()
        c_channels = 5
        self.c_embedding = nn.Embedding(c_channels, d_model)

        self.net = nn.Sequential(
            # nn.Linear(d_model * 2, d_model * 2),
            (
                nn.Linear(d_model, d_model)
                if not USE_BITLINEAR
                else BitLinear(d_model, d_model)
            ),
            nn.SiLU(),
            # nn.Linear(d_model * 2, d_model),
            (
                nn.Linear(d_model, d_model)
                if not USE_BITLINEAR
                else BitLinear(d_model, d_model)
            ),
        )

        self.freeze_c = False

    def forward(self, s, c):
        if self.freeze_c:
          with torch.no_grad():
            c = self.c_embedding(c)
        else:
          c = self.c_embedding(c)
        z = s + c
        z = self.net(z)
        return z


# %%
class iDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_vocab: int,
    ):
        super(iDecoder, self).__init__()
        n_layer = 16
        ssm_cfg = None
        norm_epsilon = 1e-5
        rms_norm = False
        residual_in_fp32 = False
        fused_add_norm = False
        device = None
        dtype = None
        initializer_cfg = None
        factory_kwargs = {"device": device, "dtype": dtype}
        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = nn.LayerNorm(d_model, eps=norm_epsilon)

        self.lm_head = nn.Linear(d_model, n_vocab + 2, bias=True)

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def forward(self, x, num_last_tokens=0):
        hidden_states = x
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=None
            )

        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))

        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens, :]
        lm_logits = self.lm_head(hidden_states)
        return lm_logits


class iDecoderJamba(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_vocab: int,
    ):
        super(iDecoderJamba, self).__init__()
        n_layer = 6
        heads = 8
        self.lm_head = nn.Linear(d_model, n_vocab + 2, bias=True)
        config = JambaConfig(
            vocab_size=n_vocab + 2,
            tie_word_embeddings=False,
            hidden_size=d_model,
            num_experts=1,
            num_experts_per_tok=1,
            num_hidden_layers=n_layer,
            intermediate_size=d_model * 4,
            num_attention_heads=heads,
            num_key_value_heads=heads,
        )
        self.mixer = JambaModel(config)

    def forward(self, x, num_last_tokens=0):
        outputs = self.mixer(inputs_embeds=x, output_hidden_states=True)
        hidden_states = outputs[0]
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens, :]
        lm_logits = self.lm_head(hidden_states)
        return lm_logits


class iDecoderPretrainedHGRN(nn.Module):
    def __init__(self, hgrn_model, lm_head):
        super(iDecoderPretrainedHGRN, self).__init__()
        self.d_model = hgrn_model.embeddings.weight.shape[1]
        self.vocab_size = hgrn_model.embeddings.weight.shape[0]
        self.hgrn_model = hgrn_model if not DEEP_COPY_DECODER else copy.deepcopy(hgrn_model)
        # self.lm_head = lm_head
        self.lm_head = nn.Linear(self.d_model, self.vocab_size + 2, bias=True)


    def forward(self, x, num_last_tokens=0):
        # Assume x are input embeds, like in the case of Jamba
        hidden_states = self.hgrn_model(
            inputs_embeds=x, output_hidden_states=True
        ).last_hidden_state
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens, :]
        lm_logits = self.lm_head(hidden_states)
        return lm_logits


class iDecoderBased(nn.Module):
    def __init__(self, based_model, lm_head):
        super(iDecoderBased, self).__init__()
        self.d_model = based_model.embeddings.word_embeddings.weight.shape[1]
        self.vocab_size = based_model.embeddings.word_embeddings.weight.shape[0]
        self.based_model = based_model if not DEEP_COPY_DECODER else copy.deepcopy(based_model)
        self.lm_head = lm_head
        # self.lm_head = nn.Linear(self.d_model, self.vocab_size + 2, bias=True)


    def forward(self, x, num_last_tokens=0):
        # Assume x are input embeds, like in the case of Jamba
        hidden_states = self.based_model.forward_with_hidden_states(x)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens, :]
        lm_logits = self.lm_head(hidden_states)
        return lm_logits



# %%
class iActor(nn.Module):
    def __init__(
        self,
        based_model,
        lm_head,
        **kwargs,
    ):
        super(iActor, self).__init__()
        self.decoder = iDecoderBased(based_model, lm_head)

    def forward(
        self,
        x,
        z=None,
        num_last_tokens=0,
    ):
        assert z is not None
        z = z.view(-1, 1, z.shape[-1])
        x = x + z
        lm_logits = self.decoder(x, num_last_tokens=num_last_tokens)
        return lm_logits


class iCritic(nn.Module):
    def __init__(
        self,
        d_model: int,
    ):
        super(iCritic, self).__init__()
        self.fc = nn.Linear(d_model, 1) if not USE_BITLINEAR else BitLinear(d_model, 1)

    def forward(self, s):
        return self.fc(s)


class iOptionAppraiserFast(nn.Module):
    def __init__(
        self,
        d_model: int,
    ):
        super(iOptionAppraiserFast, self).__init__()
        self.fc = nn.Linear(d_model, 1) if not USE_BITLINEAR else BitLinear(d_model, 1)

    def forward(self, s, z):
        x = s + z
        return self.fc(x)


class iSys1(nn.Module):
    def __init__(self, d_model: int, n_vocab: int, hgrn_model, lm_head, **kwargs):
        super(iSys1, self).__init__()
        self.option_gen = iOptionGen(d_model)
        self.actor = iActor(hgrn_model, lm_head)
        self.critic = iCritic(d_model)
        self.appraiserfast = iOptionAppraiserFast(d_model)
        # TODO: Add appraiserfast and appraiserdeep

    def generate_option(self, s, c):
        z = self.option_gen(s, c)
        # return s # This works for sure
        return z

    def appraise_option(self, s, z):
        return self.appraiserfast(s, z)

    def decode_option(self, x, z, num_last_tokens=0, bias_with_critic=False, s=None):
        if bias_with_critic:
            v = self.critic(s)
            logits = self.actor(x, z, num_last_tokens=num_last_tokens)
            if len(logits.shape) == 3:
                v = v.unsqueeze(1)
            logits = logits + v
            return logits
        return self.actor(x, z, num_last_tokens=num_last_tokens)

    def forward(
        self,
        x,
        c=None,
        z=None,
        mu=None,
        logvar=None,
    ):
        raise NotImplementedError


# %%
class iSys2(nn.Module):
    def __init__(
        self,
        d_model: int,
    ):
        super(iSys2, self).__init__()
        self.pos_embedding = nn.Embedding(9, d_model)
        self.fc = nn.Linear(d_model, 1) if not USE_BITLINEAR else BitLinear(d_model, 1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x, mask):
        x = x + self.pos_embedding(torch.arange(x.size(1)).to(x.device)).unsqueeze(0)
        x = self.transformer_encoder(x, mask)
        return x, self.fc(x)


# %%
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: Optional[torch.Tensor] = None

    def reset(self, max_seqlen, max_batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()


# %%
def modify_logits_for_top_p_filtering(logits, top_p):
    """Set the logits for none top-p values to -inf. Done in-place."""
    if top_p <= 0.0 or top_p >= 1.0:
        return
    # First sort and calculate cumulative sum of probabilities.
    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    logits.masked_fill_(indices_to_remove, float("-inf"))


def sample(logits, top_k=1, top_p=0.0, temperature=1.0):
    """Sample from top-k logits.
    Arguments:
        logits: Tensor of shape (batch_size, vocab_size)
    """
    if top_k == 1:  # Short-circuit for greedy decoding
        return logits.argmax(dim=-1)
    else:
        if top_p > 0.0:
            assert top_p <= 1.0, "top-p should be in (0, 1]."
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))  # Safety check
            logits_top, indices = torch.topk(logits, top_k, dim=-1)
            if temperature != 1.0:
                logits_top /= temperature
            modify_logits_for_top_p_filtering(logits_top, top_p)
            return indices[
                torch.arange(indices.shape[0], device=indices.device),
                torch.multinomial(
                    torch.softmax(logits_top, dim=-1), num_samples=1
                ).squeeze(dim=-1),
            ]
        else:
            # Clone so that when we modify for top_p we don't change the original logits
            logits_top = logits / temperature if temperature != 1.0 else logits.clone()
            modify_logits_for_top_p_filtering(logits_top, top_p)
            return torch.multinomial(
                torch.softmax(logits_top, dim=-1), num_samples=1
            ).squeeze(dim=-1)


BIAS_WITH_CRITIC = False


class Igre(nn.Module):
    def __init__(self, based_model, lm_head, tokenizer):
        super(Igre, self).__init__()
        self.encoder = IEncoderBased(based_model)
        d_model = based_model.embeddings.word_embeddings.weight.shape[1]
        n_vocab = based_model.embeddings.word_embeddings.weight.shape[0]
        self.sys1 = iSys1(d_model, n_vocab, based_model, lm_head)
        self.sys2 = iSys2(d_model)
        self.tokenizer = tokenizer
        self.tie_weights()
        # self.freeze_c_embedding()

    def tie_weights(self):
        self.sys1.actor.decoder.lm_head.weight = self.encoder.embedding_weight

    def freeze_c_embedding(self):
        self.sys1.option_gen.freeze_c = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def encode(self, x, output_pooled_state=False):
        return self.encoder(x, output_pooled_state=output_pooled_state)

    def _gen1_decode(self, input_ids, z, **kwargs):
        batch_size, seqlen_og = input_ids.shape
        inference_params = InferenceParams(max_seqlen=2, max_batch_size=batch_size)
        input_ids_len = input_ids.shape[1]
        inference_params.seqlen_offset = input_ids_len

        scores, seqs = [], [input_ids]
        allowed_tokens = kwargs.get("allowed_tokens", [])

        def get_logits(input_ids, z):
            return self.sys1_forward_with_option(
                input_ids, z, num_last_tokens=1, allowed_tokens=allowed_tokens
            )

        def sample_tokens(logits, inference_params):
            token = sample(logits)
            # token = sample(logits, top_k=0)
            print("DEBUG SAMPLE TOKENS")
            for x in allowed_tokens:
                print(self.tokenizer.decode(x), x)
                print(logits[:, x])
                print("----")
            print("token", token)
            print("logit at token", logits[:, token.item()])
            print("max logit", logits.max())
            return token.unsqueeze(1)

        def should_stop(current_token, inference_params):
            max_length = input_ids_len + 10
            if inference_params.seqlen_offset == 0:
                return False
            if (
                self.tokenizer.eos_token_id is not None
                and (current_token == self.tokenizer.eos_token_id).all()
            ):
                return True
            if inference_params.seqlen_offset >= max_length - 1:
                return True
            return False

        seq_cat = input_ids
        # while not should_stop(seqs[-1], inference_params):
        while not should_stop(seqs[-1], inference_params):
            print("DEBUG GENERATE")
            print("seq_cat", seq_cat)
            print(self.tokenizer.decode(seq_cat[0].tolist()))
            logits = get_logits(seq_cat, z)
            scores.append(logits)
            sampled_tokens = sample_tokens(logits, inference_params)
            seqs.append(sampled_tokens)
            seq_cat = torch.cat(seqs, dim=1)
            inference_params.seqlen_offset += seqs[-1].shape[1]
        final_seq = torch.cat(seqs, dim=1)
        print("--------- DEBUG GENERATE FINAL -------- ")
        print("final_seq", final_seq)
        print(self.tokenizer.decode(final_seq[0].tolist()))
        return torch.cat(seqs, dim=1), scores

    def sys1_forward_with_option(
        self, input_ids, z, num_last_tokens=0, allowed_tokens=None
    ):
        x, s = self.encode(input_ids, output_pooled_state=True)
        logits = None
        if BIAS_WITH_CRITIC:
            logits = self.sys1.decode_option(
                x, z, bias_with_critic=True, s=s, num_last_tokens=num_last_tokens
            )
        else:
            logits = self.sys1.decode_option(x, z, num_last_tokens=num_last_tokens)
        if allowed_tokens is not None and len(allowed_tokens) > 0:
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[:, allowed_tokens] = False
            logits.masked_fill_(mask, float("-inf"))
        return logits

    def sys1_logits_and_critic_value_with_option(
        self, input_ids, z, allowed_tokens=None
    ):
        x, s = self.encode(input_ids, output_pooled_state=True)
        logits = None
        if BIAS_WITH_CRITIC:
            logits = self.sys1.decode_option(x, z, bias_with_critic=True, s=s)
        else:
            logits = self.sys1.decode_option(x, z)
        # if allowed_tokens is not None and len(allowed_tokens) > 0:
        #     mask = torch.ones_like(logits, dtype=torch.bool)
        #     mask[:, :, allowed_tokens] = False
        #     logits.masked_fill_(mask, float("-inf"))
        value = self.sys1.critic(s)
        return logits, value

    def gen1(self, input_ids, c, return_z=False, **kwargs):
        """
        x: (batch_size, seq_len) input id sequence
        c: (batch_size) option class
        """
        if type(c) is int:
            c = torch.tensor([c], dtype=torch.long).to(input_ids.device)
        # First we get the hidden state representation of the input sequence
        x, s = self.encode(input_ids, output_pooled_state=True)
        # Now we need to produce the option embedding
        z = self.sys1.generate_option(s, c)
        # Let's get the output sequence now
        seqs, scores = self._gen1_decode(input_ids, z, **kwargs)
        if return_z:
            return seqs, scores, z
        return seqs, scores

    def sys1_logits_and_critic_value_with_c(self, input_ids, c, allowed_tokens=None):
        if type(c) is int:
            c = (
                torch.tensor([c], dtype=torch.long)
                .expand(input_ids.shape[0])
                .to(input_ids.device)
            )
        x, s = self.encode(input_ids, output_pooled_state=True)
        z = self.sys1.generate_option(s, c)
        logits = None
        if BIAS_WITH_CRITIC:
            logits = self.sys1.decode_option(x, z, bias_with_critic=True, s=s)
        else:
            logits = self.sys1.decode_option(x, z)
        logits = self.sys1.decode_option(x, z)
        # if allowed_tokens is not None and len(allowed_tokens) > 0:
        #     mask = torch.ones_like(logits, dtype=torch.bool)
        #     mask[:, :, allowed_tokens] = False
        #     logits.masked_fill_(mask, float("-inf"))
        value = self.sys1.critic(s)
        return logits, value

    def sys1_critic_value(self, input_ids):
        x, s = self.encode(input_ids, output_pooled_state=True)
        return self.sys1.critic(s)


if __name__ == "__main__":
    tokenizer = nn.Module()
    tokenizer.decode = lambda x: x
    setattr(tokenizer, "eos_token_id", 2)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import AutoTokenizer
    from based.models.gpt import GPTLMHeadModel

    # Change here to our open-sourced model
    # name = 'ridger/MMfreeLM-1.3B'
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = GPTLMHeadModel.from_pretrained_hf("hazyresearch/based-360m")

    based_model = model.transformer
    lm_head = model.lm_head

    igre = Igre(based_model, lm_head, tokenizer)
    igre.cuda().half()

    result = igre.gen1(torch.tensor([1, 2, 3]).unsqueeze(0).cuda(), c=0)
    print(result)
