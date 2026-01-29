"""
Docstring for arena_exercises.transformer_from_scratch.transformer_from_scratch

following along with the ARENA notebooks and this is my transformer from memory practice, modeled after their architecture 
"""

# imports
import einops
import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader
from jaxtyping import Float, Int
from torch import Tensor
import datasets
from tqdm import tqdm
from dataclasses import dataclass
from transformer_lens.utils import gelu_new, tokenize_and_concatenate
from transformers import GPT2TokenizerFast
import wandb
import numpy as np

# device
device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)

# model config
@dataclass
class Config:
    d_model : int = 768
    d_vocab : int = 50257
    n_ctx : int = 1024
    layer_norm_eps : float = 1e-2
    init_range : float = 0.02
    n_heads : int = 12
    d_head : int = 64
    d_mlp : int = 3072
    n_layers : int = 12

# training config
@dataclass
class TrainingConfig:
    batch_size : int = 4
    epochs : int = 2
    max_steps : int = 100
    lr : float = 3e-4
    weight_decay : float = 1e-2
    wandb_project : str | None = "transformer_from_scratch"
    wandb_name : str | None = "transformer_from_scratch"

# embed
class Embed(nn.Module):
    def __init__(self, cfg : Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=cfg.init_range)
    
    def forward(self, tokens : Int[Tensor, "batch posn"]) -> Float[Tensor, "batch posn d_model"]:
        return self.W_E[tokens]

# pos embed
class PosEmbed(nn.Module):
    def __init__(self, cfg : Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=cfg.init_range)
    
    def forward(self, tokens : Int[Tensor, "batch seq"]) -> Float[Tensor, "batch posn d_model"]:
        batch, seq = tokens.shape
        return einops.repeat(self.W_pos[:seq], "seq d_model -> batch seq d_model", batch=batch)
    
# layernorm
class LayerNorm(nn.Module):
    def __init__(self, cfg : Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones((cfg.d_model)))
        self.b = nn.Parameter(t.zeros((cfg.d_model)))

    def forward(self, residual : Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        resiudal_mean = residual.mean(dim=-1, keepdim=True)

        residual_std = (residual.var(dim=-1, keepdim=True, unbiased=False) + self.cfg.layer_norm_eps).sqrt()

        residual = (residual - resiudal_mean) / residual_std

        return residual * self.w + self.b
    
# attention
class Attention(nn.Module):

    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg : Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_Q, std=cfg.init_range)
        nn.init.normal_(self.W_K, std=cfg.init_range)
        nn.init.normal_(self.W_V, std=cfg.init_range)
        nn.init.normal_(self.W_O, std=cfg.init_range)
        self.register_buffer("IGNORE", t.tensor(float('-inf'), dtype=t.float32, device=device))
    
    def forward(self, residual : Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        q = einops.einsum(
            residual,
            self.W_Q,
            "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head"
        ) + self.b_Q

        k = einops.einsum(
            residual,
            self.W_K,
            "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head"
        ) + self.b_K

        v = einops.einsum(
            residual,
            self.W_V,
            "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head"
        ) + self.b_V

        attn_scores = einops.einsum(
            q,
            k,
            "batch posn_query n_heads d_head, batch posn_key n_heads d_head -> batch n_heads posn_query posn_key"
        )

        attn_scores_masked = self.apply_causal_mask(attn_scores / self.cfg.d_head**0.5)
        attn_pattern = attn_scores_masked.softmax(-1)

        z = einops.einsum(
            v,
            attn_pattern,
            "batch posn_key n_heads d_head, batch n_heads posn_query posn_key -> batch posn_query n_heads d_head"
        ) 

        attn_out = einops.einsum(
            z,
            self.W_O,
            "batch posn n_heads d_head, n_heads d_head d_model -> batch posn d_model"
        ) + self.b_O

        return attn_out

    def apply_causal_mask(self, attn_scores):
        all_ones = t.ones(attn_scores.size(-2), attn_scores.size(-1), device=device)
        mask = t.triu(all_ones, diagonal=1).bool()
        attn_scores.masked_fill_(mask, self.IGNORE)

        return attn_scores

# mlp
class MLP(nn.Module):
    def __init__(self, cfg : Config):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
        self.W_out = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)))
        self.b_in = nn.Parameter(t.zeros((cfg.d_mlp)))
        self.b_out = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_in, std=cfg.init_range)
        nn.init.normal_(self.W_out, std=cfg.init_range)
    
    def forward(self, residual : Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        pre = einops.einsum(
            residual,
            self.W_in,
            "batch posn d_model, d_model d_mlp -> batch posn d_mlp"
        ) + self.b_in

        act = gelu_new(pre)

        out = einops.einsum(
            act,
            self.W_out,
            "batch posn d_mlp, d_mlp d_model -> batch posn d_model"
        ) + self.b_out

        return out
    
# t block
class TBlock(nn.Module):
    def __init__(self, cfg : Config):
        super().__init__()
        self.cfg = cfg
        self.ln_attn = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln_mlp = LayerNorm(cfg)
        self.mlp = MLP(cfg)
    
    def forward(self, residual : Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        resid_mid = self.attn(self.ln_attn(residual)) + residual
        resid_post = self.mlp(self.ln_mlp(resid_mid)) + resid_mid
        return resid_post

# unembed
class Unembed(nn.Module):
    def __init__(self, cfg : Config):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)))
        self.b_U = nn.Parameter(t.zeros((cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=cfg.init_range)

    def forward(self, residual : Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_vocab"]:
        return einops.einsum(
            residual,
            self.W_U,
            "batch posn d_model, d_model d_vocab -> batch posn d_vocab"
        ) + self.b_U
    
# model
class TransformerModel(nn.Module):
    def __init__(self, cfg : Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)
    
    def forward(self, tokens : Int[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_vocab"]:
        residual = self.embed(tokens) + self.pos_embed(tokens)

        for block in self.blocks:
            residual = block(residual)

        logits = self.unembed(self.ln_final(residual))        
        return logits
        
# get dataset
def get_dataset(tokenizer : GPT2TokenizerFast, cfg : Config):
    dataset = datasets.load_dataset("roneneldan/TinyStories", split="train")
    tokenized_dataset = tokenize_and_concatenate(
        dataset,
        tokenizer,
        streaming=False,
        max_length=cfg.n_ctx,
        add_bos_token=True,
        column_name="text",
        num_proc=4
    )
    dataset_dict = tokenized_dataset.train_test_split(test_size=1000)
    return dataset_dict

# get log probs
def get_log_probs(
    logits : Float[Tensor, "batch posn d_vocab"],
    tokens : Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:
    log_logits = logits.log_softmax(-1)
    log_logits_for_token = (
        log_logits[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    )
    return log_logits_for_token

# trainer
class Trainer:
    def __init__(self, args : TrainingConfig, model : TransformerModel, tokenizer : GPT2TokenizerFast, dataset_dict):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.sampler = Sampler(model, tokenizer)
        self.optimizer = t.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        self.step = 0
        self.train_loader = DataLoader(
            dataset_dict["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        self.test_loader = DataLoader(
            dataset_dict["test"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
    
    def training_step(self, batch : dict[str, Int[Tensor, "batch seq"]]) -> Float[Tensor, ""]:
        tokens = batch["tokens"].to(device)
        logits = self.model(tokens)
        loss = -get_log_probs(logits, tokens).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.step += 1
        wandb.log({"loss":loss}, step=self.step)
        return loss
    
    @t.inference_mode()
    def evaluate(self) -> float:
        self.model.eval()
        total_correct, total_samples = 0,0

        for batch in tqdm(self.test_loader, desc="evaluating"):
            tokens = batch["tokens"].to(device)
            logits = self.model(tokens)[:, :-1]
            predicted_tokens = logits.argmax(dim=-1)
            total_correct += (predicted_tokens == tokens[:, 1:]).sum().item()
            total_samples += tokens.size(0) * (tokens.size(1) - 1)

        accuracy = total_correct / total_samples
        wandb.log({"accuracy": accuracy}, step=self.step)
        self.model.train()
        return accuracy
    
    def train(self):
        wandb.init(
            project=self.args.wandb_project,
            name=self.args.wandb_name,
            config=self.args
        )

        pbar = tqdm(total=self.args.epochs * self.args.max_steps)
        accuracy = np.nan

        for epoch in range(self.args.epochs):
            for i,batch in enumerate(self.train_loader):
                loss = self.training_step(batch)
                pbar.update()
                pbar.set_description(
                    f"Epoch: {epoch+1}, loss: {loss:.3f}, accuracy: {accuracy:.3f}"
                )
                if i >= self.args.max_steps:
                    break
            accuracy = self.evaluate()

        wandb.finish()

# sampler
class Sampler:
    def __init__(self, model : TransformerModel, tokenizer : GPT2TokenizerFast):
        self.model = model
        self.cfg = model.cfg
        self.tokenizer = tokenizer
    
    @t.inference_mode()
    def sample(self, prompt: str, max_tokens_generated = 100, **kwargs) -> str:
        self.model.eval()
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)[0]

        for _ in range(max_tokens_generated):
            logits = self.model(input_ids[None, -self.cfg.n_ctx:])
            logits = logits[0, -1]
            next_token = t.tensor(
                [Sampler.sample_next_token(input_ids, logits, **kwargs)], device=device
            )
            input_ids = t.cat([input_ids, next_token], dim=-1)
            if next_token == getattr(self.tokenizer, "eos_token_id", None):
                break
        return self.tokenizer.decode(input_ids)

    @staticmethod
    def sample_next_token(
        input_ids : Int[Tensor, "posn"],
        logits : Float[Tensor, "d_vocab"],
        temperature : float = 1.0,
        top_k : int = 0,
        top_p : float = 0.5,
        frequency_penalty= 0.0,
        seed=None
    ) : 
        # a bunch of asserts of inputs
        if seed is not None:
            t.manual_seed(seed)
            np.random.seed(seed)
        
        # if temperature == 0:
        #     return Sampler.greedy_search(logits)
        # elif temperature != 1.0:
        #     logits = Sampler.apply_temperature(logits, temperature)
        # if frequency_penalty != 0.0:
        #     logits = Sampler.apply_frequency_penalty(
        #         input_ids, logits, frequency_penalty
        #     )
        # if top_k > 0:
        #     return Sampler.sample_top_l(logits, top_k)
        if top_p > 0.0:
            return Sampler.sample_top_p(logits, top_p)
        # return Sampler.sample_basic(logits)
    
    @staticmethod
    def sample_top_p(logits : Float[Tensor, "d_vocab"], top_p, min_tokens_to_keep=1):
        logits_sorted, indices = logits.sort(descending=True, stable=True)
        cumul_probs = logits_sorted.softmax(-1).cumsum(-1)
        n_keep = t.searchsorted(cumul_probs, top_p, side="left").item() + 1
        n_keep = max(n_keep, min_tokens_to_keep)
        keep_idx = indices[:n_keep]
        keep_logits = logits[keep_idx]
        sample = t.distributions.categorical.Categorical(logits=keep_logits).sample()
        return keep_idx[sample].item()
    
# main
if __name__ == "__main__":
    cfg = Config()
    args = TrainingConfig()
    model = TransformerModel(cfg).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    dataset_dict = get_dataset(tokenizer, cfg)
    trainer = Trainer(args, model, tokenizer, dataset_dict)
    trainer.train()

    prompts = [
        "Is this real life? Is this just fantasy?",
        "Caught in a landslide",
        "Because I'm easy come easy go",
        "Little high, little low",
        "Mama, just killed a man",
        "Thunderbolt and lightning, very, very frightening me",
    ]

    for prompt in prompts:
        output = trainer.sampler.sample(
            prompt,
            max_tokens_generated=50,
            top_p = 0.5,
        )

        print(f"Generated text: {output}")
    
    print("Done.")