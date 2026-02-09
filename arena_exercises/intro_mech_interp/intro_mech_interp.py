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
from arena_plotting import imshow, plot_loss_difference, plot_logit_attribution

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
print(gpt2_tokens.shape, gpt2_logits.shape)
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
def visualize_pattern_hook(
        pattern : Float[Tensor, "batch head dest_posn source_posn"],
        hook : HookPoint
):
    print("Layer: ", hook.layer())
    display(cv.attention.attention_heads(tokens=gpt2_small.to_str_tokens(rep_tokens[0]),
                                            attention=pattern.mean()))

seq_len = 50
batch_size = 10
rep_tokens_batch = generate_repeated_tokens(gpt2_small, seq_len, batch_size)

induction_score_store = t.zeros(
    (gpt2_small.cfg.n_layers, gpt2_small.cfg.n_heads), device=gpt2_small.cfg.device
)

gpt2_small.run_with_hooks(
    rep_tokens_batch,
    return_type=None,
    fwd_hooks=[(pattern_hook_names_filter, induction_score_hook)]
)
# same as before but for gpt2_small
imshow(
    induction_score_store,
    labels={"x":"Head","y":"Layer"},
    title="Induction Score by Head",
    text_auto=".1f",
    width=700,
    height=500,
)

induction_head_layers = [5,6,7] # from heatmap
fwd_hooks = [
    (utils.get_act_name("pattern", induction_head_layer), visualize_pattern_hook)
    for induction_head_layer in induction_head_layers
]
gpt2_small.run_with_hooks(
    rep_tokens,
    return_type=None,
    fwd_hooks=fwd_hooks
)

# %% direct attribution to a logit, how each componenet contributes to the output logit
def logit_attribution(
        embed: Float[Tensor, "seq d_model"],
        l1_results : Float[Tensor, "seq n_heads d_model"],
        l2_results : Float[Tensor, "seq n_heads d_model"],
        W_U : Float[Tensor, "d_model d_vocab"],
        tokens : Int[Tensor, "seq"]
) -> Float[Tensor, "seq-1 n_components"]:
    """
    returns:
        tensor of shape (seq_len-1, n_components)
        logits attributions from:
            direct path (seq-1, 1)
            layer 0 logits (seq-1, n_heads)
            layer 1 logits (seq-1, n_heads)
        n_componenets = 1 + 2 * n_heads
    """
    W_U_correct_tokens = W_U[:, tokens[1:]]
    direct_attributions = einops.einsum(W_U_correct_tokens, embed[:-1], "emb seq, seq emb -> seq")
    l1_attributions = einops.einsum(W_U_correct_tokens, l1_results[:-1], "emb seq, seq n_heads emb -> seq n_heads")
    l2_attributions = einops.einsum(W_U_correct_tokens, l2_results[:-1], "emb seq, seq n_heads emb -> seq n_heads")
    return t.concat([direct_attributions.unsqueeze(-1), l1_attributions, l2_attributions], dim=-1)

text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
logits, cache = model.run_with_cache(text, remove_batch_dim=True)
str_tokens = model.to_str_tokens(text)
tokens = model.to_tokens(text)

with t.inference_mode():
    embed = cache["embed"]
    l1_results = cache["result", 0]
    l2_results = cache["result", 1]
    logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens[0])
    correct_token_logits = logits[0, t.arange(len(tokens[0]) - 1), tokens[0, 1:]]
    t.testing.assert_close(logit_attr.sum(1), correct_token_logits, atol=1e-3, rtol=0)
    print("Done.")
# %% visualize logit attribution for each path
embed = cache["embed"]
l1_results = cache["result", 0]
l2_results = cache["result", 1]
logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens.squeeze())

plot_logit_attribution(model, logit_attr, tokens, title="Logit attribution")
# high values are tokens that are common bigrams
# %%
seq_len = 50
embed = rep_cache["embed"]
l1_results = rep_cache["result", 0]
l2_results = rep_cache["result", 1]

logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, rep_tokens.squeeze())
plot_logit_attribution(model, logit_attr, rep_tokens.squeeze(), title="logit attribution")
# %% ablate these guys, I've never seen the word ablate used besides in medical contexts
# set some part of the model to 0, Occam's razor? trying to figure out the smallest possible set that keeps the "feature" we are looking for
def head_zero_ablation_hook(
        z : Float[Tensor, "batch seq n_heads d_head"],
        # hook : HookPoint,
        head_index_to_ablate: int
) -> None:
    z[:, :, head_index_to_ablate, :] = 0.0

def get_ablation_scores(
        model : HookedTransformer,
        tokens : Int[Tensor, "batch seq"],
        ablation_function : Callable = head_zero_ablation_hook
) -> Float[Tensor, "n_layers n_heads"]:
    ablation_scores = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)

    model.reset_hooks()
    seq_len = (tokens.shape[1] - 1) // 2
    logits = model(tokens, return_type="logits")
    loss_no_ablation = -get_log_probs(logits, tokens)[:, -(seq_len - 1) :].mean()

    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            # temp hook
            temp_hook_fn = functools.partial(ablation_function, head_index_to_ablate=head)
            # run model with ablated hooks
            ablated_logits = model.run_with_hooks(
                tokens, fwd_hooks=[(utils.get_act_name("z", layer), temp_hook_fn)]
            )
            # calculate loss difference
            loss = -get_log_probs(ablated_logits, tokens)[:, -(seq_len - 1) :].mean()
            # store result
            ablation_scores[layer, head] = loss - loss_no_ablation

        return ablation_scores

ablation_scores = get_ablation_scores(model, rep_tokens)
# print(ablation_scores)
imshow(
    ablation_scores,
    labels={"x":"Head", "y":"Layer", "color":"Logit diff"},
    title="loss diff after ablating",
    text_auto=".2f",
    width=900,
    height=350
)
# %% mean ablation so that the unimportant heads ahve a value closer to 0
# if the model was trained with dropout then it is trained to be resistant to "knockout"/0 value ablations
def head_mean_ablation_hook(
        z : Float[Tensor, "batch seq n_heads d_head"],
        hook : HookPoint, # I guess it's good practice to include the hook point when adjusting an activation
        head_index_to_ablate : int,
) -> None: 
    z[:, :, head_index_to_ablate, :] = z[:, :, head_index_to_ablate, :].mean(0)

rep_tokens_batch = run_and_cache_model_repeated_tokens(model, seq_len=50, batch_size=10)[0]
mean_ablation_scores = get_ablation_scores(
    model, rep_tokens_batch, ablation_function=head_mean_ablation_hook
)
imshow(
    mean_ablation_scores,
    labels={"x":"Head", "y":"Layer", "color":"Logit diff"},
    title="Loss Diff After Mean Ablation",
    text_auto=".2f",
    width = 900,
    height=350,
)

#%% induction, OV, QK circuits and conceptual problems
# with repeated sequences of tokens, there are heads with induction pattern diagonal strip with offset seq_len - 1
# logit attribution to tell which heads were important for getting better predictions

# I can't define the induction circuit at a lower level including circuits but I can define it as heads that show connection to a previous layer head that connects to previous tokens
# so when the model sees A1 and it saw A B before then it will more likely predict B after seeing A1

# Their summary of the algorithm:
"""
Head 0.7 is a previous token head (QK circuit ensures it alwaus attends to the previous token)
OV circuit of head 0.7 writes a copy of the previous token in a different subspace to the one used by the embedding
the output of head 0.7 is used by the key input of head 1.10 via K-composition to attend to "the source token whose previous token is the destination token' aka B
OV circuit of head 1.10 copies the value of the course token to the same output logit (copying from the embedding space, not the 0.7 output subspace (not using V-Composition at all))
1.4 is also performing the same role as 1.10 (together they are more accurate)

* "Hardest part is computing the attention pattern of the induction head - takes careful composition" * Me: No idea why
"""
#%%
# matricies shapes and interpretations
# (describe the type of input it takes and what the output represents)
"""
W_OV_h = 
    Is at head h, the OV circuit, (W_V_h)(W_O_h)
    (W_E)(W_OV_h)(W_U) is full OV circuit
    Inputs it takes is v and o?
    whats the difference between just W_OV and putting it inbetween the embedding and unembedding matrices

"""
#%% OV copying circuit
# one hot encoding, zeros everywhere except one at the index A
# A^T W_E = embedding vector for A
# W_OV_h = (W_V_h)(W_O_h) is OV circuit for head h and (W_E)(W_OV_h)(W_U) is the full OV circuit
# calculating a circuit by mutiplying matricies

# compute OV circuit for 1.4
head_index = 4
layer = 1

W_O = model.W_O[layer, head_index]
W_V = model.W_V[layer, head_index]
W_E = model.W_E
W_U = model.W_U

OV_circuit = FactoredMatrix(W_V, W_O)
full_OV_circuit = W_E @ OV_circuit @ W_U

print(full_OV_circuit.shape)
print(type(full_OV_circuit))

indices = t.randint(0, model.cfg.d_vocab, (200,))
full_OV_circuit_sample = full_OV_circuit[indices, indices].AB

imshow(
    full_OV_circuit_sample,
    labels={"x":"Logits on output token", "y":"Input token"},
    title="Full OV circuit for copying head",
    width=700,
    height=600
)


#%% complete circuit accuracy 
def top_1_acc(full_OV_circuit : FactoredMatrix, batch_size: int = 1000) -> float:
    total = 0

    for indices in t.split(t.arange(full_OV_circuit.shape[0], device=device), batch_size):
        AB_slice = full_OV_circuit[indices].AB
        total += (t.argmax(AB_slice, dim=1) == indices).float().sum().item()
    
    return total / full_OV_circuit.shape[0]

print(f"Fraction of time that the best logit is on diagonal: {top_1_acc}(full_OV_circuit):.4f}")

#%% compite effective circuit
W_O_both = einops.rearrange(model.W_O[1, [4, 10]], "ehad d_head d_model -> (head d_head) d_model")
W_V_both = einops.rearrange(model.W_V[1, [4, 10]], "ehad d_head d_model -> head (d_head d_model)")

W_OV_eff = W_E @ FactoredMatrix(W_V_both, W_O_both) @ W_U
print(f"Fraction of the time that the best logit is on the diagonal: {top_1_acc(W_OV_eff):.4f}")

#%% QK prev-token circuit
# W_QK_h = (W_Q_h)(W_K_h)^T is the QK circuit for head h and (W_E)(W_QK_h)(W_E)^T is the full QK circuit
# order is slightly different from mathematical framework for transformers paper, why? (transformersLens library is different)
layer = 0
head_index = 7

# compute full QK matrix for pos embedding
W_pos = model.W_pos
W_QK = model.W_Q[layer, head_index] @ model.W_K[layer, head_index].T
pos_by_pos_scores = W_pos @ W_QK @ W_pos.T

# mask, scale and softmax the scores
mask = t.triu(t.ones_like(pos_by_pos_scores)).bool()
pos_by_pos_pattern = t.where(mask, pos_by_pos_scores / model.cfg.d_head**0.5, -1.0e6).softmax(-1)

print(f"Avg lower-diagonal value : {pos_by_pos_pattern.diag(-1).mean():.4f}")
imshow(
    utils.to_numpy(pos_by_pos_pattern[:200, :200]),
    labels={"x":"Key", "y":"Query"},
    title="Attention patterns for prev-token QK circuit, first 100 indices",
    width = 700,
    height=600
)
#%% K-composition circuit
def decomposed_qk_input(cache: ActivationCache) -> Float[Tensor, "n_heads+2 posn d_model"]:
    y0 = cache["embed"].unsqueeze(0)
    y1 = cache["pos_embed"].unsqueeze(0)
    y_rest = cache["result", 0].transpose(0, 1)

    return t.concat([y0, y1, y_rest], dim=0)

def decompose_q(
        decomposed_qk_input : Float[Tensor, "n_heads+2 posn d_model"],
        ind_head_index: int, 
        model : HookedTransformer,
) -> Float[Tensor, "n_heads+2 posn d_head"]:
    W_Q = model.W_Q[1, ind_head_index]
    return einops.einsum(decomposed_qk_input, W_Q, "n seq d_model, d_model d_head -> n seq d_head")

def decompose_k(
        decomposed_qk_input: Float[Tensor, "n_heads+2 posn d_model"],
        ind_head_index: int,
        model : HookedTransformer,
) -> Float[Tensor, "n_heads+2 posn d_head"]:
    W_K = model.W_K[1, ind_head_index]
    return einops.einsum(
        decomposed_qk_input,
        W_K,
        "n seq d_model, d_model d_head -> n seq d_head"
    )

seq_len = 50
batch_size = 1
(rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(model, seq_len, batch_size)
rep_cache.remove_batch_dim()

ind_head_index = 4

decomposed_qk_input = decomposed_qk_input(rep_cache)
decomposed_q = decompose_q(decomposed_qk_input, ind_head_index, model)
decomposed_k = decompose_k(decomposed_qk_input, ind_head_index, model)

component_labels = ["Embed", "PosEmbed"] + [f"0.{h}" for h in range(model.cfg.n_heads)]
for decomposed_input, name in [(decompose_q, "query"), (decompose_k, "key")]:
    imshow(
        utils.to_numpy(decomposed_input.pow(2).sum([-1])),
        label={"x":"Position", "y":"Component"},
        y=component_labels,
        width=800,
        height=400
    )
# %% decompose attention scores
def decompose_attn_scores(
        decomposed_q : Float[Tensor, "q_comp q_pos d_head"],
        decomposed_k : Float[Tensor, "k_comp k_pos d_head"],
        model : HookedTransformer,
) -> Float[Tensor, "q_comp k_comp q_pos k_pos"]:
    return einops.einsum(
        decompose_q,
        decompose_k,
        "q_comp q_pos d_head, k_comp k_pos d_head -> q_comp k_comp q_pos k_pos"
    ) / (model.cfg.d_head**0.5)

decomposed_scores = decompose_attn_scores(decompose_q, decompose_k, model)
q_label = "Embed"
k_label = "0.7"
decomposed_scores_from_pair = decomposed_scores[
    component_labels.index(q_label), component_labels.index(k_label)
]
imshow(
    utils.to_numpy(t.tril(decomposed_scores_from_pair)),
    title=f"Attention score contributions from query = {q_label}, key = {k_label}<br>(by query & key sequence positions)",
    width=700
)

decomposed_stds = einops.reduce(
    decomposed_scores,
    "query_decomp, key_decomp query_pos key_pos -> query_decomp key_decomp",
    t.std
)
imshow(
    utils.to_numpy(decomposed_stds),
    labels={"x":"Key Component", "y":"QUery Component"},
    title="std dev of attn score contributions across sequence positions<br>(by query & key comp)",
    z=component_labels,
    y=component_labels,
    width=700
)
#%% Still attention score contributions
decomposed_scores_centered = t.tril(
    decomposed_scores - decomposed_scores.mean(dim=-1, keepdim=True)
)

decomposed_scores_reshaped = einops.rearrange(
    decomposed_scores_centered,
    "q_comp k_comp q_token k_token -> (q_comp q_token) (k_comp k_token)"
)

fig = imshow(
    decomposed_scores_reshaped,
    titel="Attention score contributions from all pairs of (key, query) components",
    width=1200,
    height=1200.
    return_fig=True,
)
full_seq_len = seq_len * 2 + 1
for i in range(0, full_seq_len * len(component_labels), full_seq_len):
    fig.add_hline(y=i, line_color="black", line_width=1)
    fig.add_vline(x=i, line_color="black", line_width=1)

fig.show(config={"staticPlot": True})
#%% K - comp circuit
def find_K_comp_circuit(
        model : HookedTransformer,
        prev_token_head_index: int,
        ind_head_index : int
) -> FactoredMatrix:
    W_E = model.W_E
    W_Q = model.W_O[1, ind_head_index]
    W_K = model.W_K[1, ind_head_index]
    W_O = model.W_O[0, prev_token_head_index]
    W_V = model.W_V[0, prev_token_head_index]

    Q = W_E @ W_Q
    K = W_E @ W_V @ W_K
    return FactoredMatrix(Q, K.T)

prev_token_head_index = 7
ind_head_index = 4
K_comp_circuit = find_K_comp_circuit(model, prev_token_head_index, ind_head_index)
print(f"Token frac wehre max-activating key equals same token: {top_1_acc(K_comp_circuit.T):.4f}")

