"""
Docstring for arena_exercises.intro_mech_interp.intro_mech_interp

Practice from the next chapter in the ARENA notebooks
"""
#%% Imports 
# Standard Library
import functools
import os
import sys
from pathlib import Path
from typing import Callable

# Scientific & Utilities
import numpy as np
import einops
from eindex import eindex
from jaxtyping import Float, Int, Bool
from tqdm import tqdm
from IPython.display import display

# PyTorch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# TransformerLens
import transformer_lens
from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
    utils,
)
from transformer_lens.hook_points import HookPoint

# Visualization
import circuitsvis as cv
from arena_plotting import imshow, plot_loss_difference

# Device Setup (Standard boilerplate)
device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

#%% Model
gpt2_small : HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")

#%% Running your model
model_description_text = """## Loading Models

HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly.

For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out, let's find the loss on this paragraph!"""

loss = gpt2_small(model_description_text, return_type="loss")
print("Model loss:", loss)

#%% Tokenization, useful functions

print(gpt2_small.to_str_tokens("gpt2"))
print(gpt2_small.to_str_tokens(["gpt2","gpt2"]))
print(gpt2_small.to_tokens("gpt2"))
print(gpt2_small.to_string([50256, 70, 457, 17, 3, 3456, 45]))

# %% How many tokens are correct in model_description_text
logits : Tensor = gpt2_small(model_description_text, return_type="logits")
prediction = logits.argmax(dim=-1).squeeze()[:-1]

true_tokens = gpt2_small.to_tokens(model_description_text).squeeze()[1:]
is_correct = prediction == true_tokens

print(f"Model accuracy: {is_correct.sum()} / {len(true_tokens)}")
print(f"Correct tokens: {gpt2_small.to_str_tokens(prediction[is_correct])}")

# %% Cache Cache Activations

gpt2_text = "Here is text from the beginning of the gpt2-paper, Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on task-specific datasets."
gpt2_tokens = gpt2_small.to_tokens(gpt2_text)
gpt2_logits, gpt2_cache = gpt2_small.run_with_cache(gpt2_tokens, remove_batch_dim=True)
print(type(gpt2_logits), type(gpt2_cache))

# %% Some different ways of accessing these activations
attn_patterns_from_shorthand = gpt2_cache["pattern", 0]
attn_patterns_from_full_name = gpt2_cache["blocks.0.attn.hook_pattern"]
print(type(attn_patterns_from_shorthand), attn_patterns_from_shorthand.shape)
print(gpt2_tokens.shape)
t.testing.assert_close(attn_patterns_from_shorthand, attn_patterns_from_full_name)
# The shape is [12, 47, 47]. 47 tokens because the attn pattern is query by keys. and 12 is the number of attn heads 

# %% verify activations 
layer0_pattern_from_cache = gpt2_cache["pattern", 0]
q, k = gpt2_cache["q", 0], gpt2_cache["k", 0]
seq, n_heads, d_head = q.shape
attn_score = einops.einsum(q, k, "posn_query n h, posn_key n h -> n posn_query posn_key")
mask = t.triu(t.ones(seq, seq, dtype=t.bool), diagonal=1).to(device)
attn_score.masked_fill_(mask, -1e9)
layer0_pattern_from_q_k = (attn_score / d_head**0.5).softmax(-1)

t.testing.assert_close(layer0_pattern_from_cache, layer0_pattern_from_q_k)

# %% Visualizing the attention pattern in the attention heads, maybe that wording is confusing
print(type(gpt2_cache))
attention_pattern = gpt2_cache["pattern", 0]
print(attention_pattern.shape)
gpt2_str_tokens = gpt2_small.to_str_tokens(gpt2_text)
print("Layer 0 Head Attention Patterns: ")
display(cv.attention.attention_heads(
    tokens=gpt2_str_tokens,
    attention=attention_pattern,
    attention_head_names=[f"Layer0Head:{i}" for i in range(attention_pattern.shape[0])]
))

# %% Lets find some induction heads
cfg = HookedTransformerConfig(
    d_model=768,
    d_head=64,
    n_heads=12,
    n_layers=2,
    n_ctx=2048,
    d_vocab=50278,
    attention_dir="causal",
    attn_only=True,
    tokenizer_name="EleutherAI/gpt-neox-20b",
    seed=398,
    use_attn_result=True,
    normalization_type=None, # default is LayerNorm
    positional_embedding_type="shortformer"
)

# %% hugging face 
from huggingface_hub import hf_hub_download
REPO_ID = "callummcdougall/attn_only_2L_half"
FILENAME = "attn_only_2L_half.pth"
weights_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

# %% create our model and load in weights
model = HookedTransformer(cfg)
pretrained_weights = t.load(weights_path, map_location=device, weights_only=True)
model.load_state_dict(pretrained_weights)

# %% setup logits and cache of activations
text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
logits, cache = model.run_with_cache(text, remove_batch_dim=True) # only one batch

# %% lets take a look
str_tokens = gpt2_small.to_str_tokens(text)
for layer in range(model.cfg.n_layers):
    attention_pattern = cache["pattern", layer]
    display(cv.attention.attention_heads(tokens=str_tokens, attention=attention_pattern))


# %% write some detectors that tells us which head is attentding to 

def current_attn_detector(cache: ActivationCache) -> list[str]:
    attn_heads = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attention_pattern = cache["pattern", layer][head]
            score = attention_pattern.diagonal().mean()
            if score > 0.4: # why 0.4? from softmax
                attn_heads.append(f"{layer}.{head}")
    return attn_heads

def prev_attn_detector(cache : ActivationCache) -> list[str]:
    attn_heads = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attention_pattern = cache["pattern", layer][head]
            score = attention_pattern.diagonal(offset=-1).mean() # or just diagonal(-1)
            if score > 0.4:
                attn_heads.append(f"{layer}.{head}")
    return attn_heads

def first_attn_detector(cache : ActivationCache) -> list[str]:
    attn_heads = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attention_pattern = cache["pattern", layer][head]
            score = attention_pattern[:, 0].mean()
            if score > 0.4:
                attn_heads.append(f"{layer}.{head}")
    return attn_heads

print("Heads attending to current token = ", ", ".join(current_attn_detector(cache)))
print("Heads attending to previous token = ", ", ".join(prev_attn_detector(cache)))
print("Heads attending to first token = ", ", ".join(first_attn_detector(cache)))
# %% Induction heads

# After reading tokens A B, the layer 0 will create a previous token head
# After reading A1 we will predict B and it will be because the layer 1 will create a connection to the previous token for that token
# when induction heads come into play there is a giant drop in loss
# resposible for a lot of in-context learning
# similar type of circuit used in translation and other settings 

# %% plot induction heads loss on predicting tokens

def generate_repeated_tokens(
        model : HookedTransformer, seq_len : int, batch_size : int = 1
) -> Int[Tensor, "batch seq_len"]:
    """
    generate a seq of repeated random tokens of:
        rep_tokens : [batch, 1 + 2 * seq_len]
    """
    t.manual_seed(1)
    prefix = (t.ones(batch_size, 1) * model.tokenizer.bos_token_id).long()
    rep_tokens_half = t.randint(0, model.cfg.d_vocab, (batch_size, seq_len), dtype=t.int64)
    rep_tokens = t.cat([prefix, rep_tokens_half, rep_tokens_half], dim=-1).to(device)
    return rep_tokens

def run_and_cache_model_repeated_tokens(
        model : HookedTransformer, seq_len : int, batch_size : int = 1
) -> tuple[Tensor, Tensor, ActivationCache]:
    """
    generate sequence, run movel on it w/cache, return:
        tokens : [batch, 1+2*seq]
        logits : [batch, 1+2*seq, d_vocab]
        cache : cache of model run with rep_tokens
    """
    rep_tokens = generate_repeated_tokens(model, seq_len, batch_size)
    rep_logits, rep_cache = model.run_with_cache(rep_tokens)
    return rep_tokens, rep_logits, rep_cache

def get_log_probs(
        logits : Float[Tensor, "batch posn d_vocab"], tokens : Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:
    log_probs = logits.log_softmax(-1)
    # want log probs [batch, seq, tokens[b, s+1]]
    # using eindex
    correct_logprobs = eindex(log_probs, tokens, "b s [b s+1]")
    return correct_logprobs

seq_len = 50
batch_size = 1
(rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(model, seq_len, batch_size)

rep_cache.remove_batch_dim()
rep_str = model.to_str_tokens(rep_tokens)
model.reset_hooks()
log_probs = get_log_probs(rep_logits, rep_tokens)

print(f"Performance on the first half: {log_probs[:seq_len].mean():.3f}")
print(f"Performance on the second half: {log_probs[seq_len:].mean():.3f}")

# plot loss between (log_probs, rep_str, seq_len)

# %% looking for induction attention patterns, cause they are sick
for layer in range(model.cfg.n_layers):
    attention_pattern = rep_cache["pattern", layer]
    display(cv.attention.attention_patterns(tokens=rep_str, attention=attention_pattern))

# %% make an induction head detector
def induction_attn_detector(cache: ActivationCache) -> list[str]:
    attn_heads = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attention_pattern = cache["pattern", layer][head]
            # take avg of (-seq_len+1) - offset elements
            seq_len = (attention_pattern.shape[-1] - 1) // 2
            score = attention_pattern.diagonal(-seq_len+1).mean()
            if score > 0.4:
                attn_heads.append(f"{layer}.{head}")
    return attn_heads

print("Induciton heads = ", ", ".join(induction_attn_detector(rep_cache)))
# both 1.4 and 1.10 are plots that have a strong daigonal that are offset, obviously from the code I guess
# but why in terms of how I understand how induciton head work?
# 
# %% Hooks notes
# a variable or intermediate variable in the network that can be viewed/edited/intervened on
# every activation inside the transformer is surrounded by a hook point

# use a hook function example below
# a hook function has to have two inputs (value, hook point)
# can have an output of the same shape or do some processing and affect something else and not return anything
"""
inputs:
    activation_value : Float[Tensor, "batch heads seq_len seq_len"]
    hook_point : HookPoint

outputs:
    Float[Tensor, "batch heads seq_len seq_len]
"""
# model.run_with_hooks

"""
loss = model.run_with_hooks(
    tokens,
    return_type="loss",
    fwd_hooks=[
        ('blocks.1.attn.hook_pattern',
        hook_function)])
"""
# hooks are nn.Modules
#%%
seq_len = 50
batch_size = 10 # more than 1!
rep_tokens_10 = generate_repeated_tokens(model, seq_len, batch_size)
# moving things between the gpu and cpu can be slow
induction_score_store = t.zeros(
    (model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device
)
# n_layers x n_heads zeros matrix
def induction_score_hook(
        pattern : Float[Tensor, "batch head dest_posn source_posn"],
        hook : HookPoint
):
    induction_stripe = pattern.diagonal(dim1=-2, dim2=-1, offset=1 - seq_len)
    induction_score = einops.reduce(
        induction_stripe, "batch head posn -> head", "mean"
    )
    induction_score_store[hook.layer(), :] = induction_score 

# boolean filter only true on pattern names
pattern_hook_names_filter = lambda name : name.endswith("pattern")
# run with hooks
model.run_with_hooks(
    rep_tokens_10,
    return_type=None,
    fwd_hooks=[(pattern_hook_names_filter, induction_score_hook)]
)
imshow(
    induction_score_store,
    labels={"x":"Head","y":"Layer"},
    title="Induction Score by Head",
    text_auto=".2f",
    width=900,
    height=350,
)
# 1.4 and 1.10 have high (>0.6) induciton head scores

# %% wow induction heads inside gpt2-small
