#%% 
"""
Toy Models and Sparse Auto-encoders

Chapter 1 - coding a transformer from scratch
Chapter 2 - learning circuits and composition
Chapter 3 - superposition and sae's
Chapter 4 - ...
"""
# imports
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import einops
import numpy as np
import pandas as pd
import plotly.express as px
import torch as t
from IPython.display import HTML, display
from jaxtyping import Float
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
from tqdm.auto import tqdm
from plotly_utils import imshow, line

device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)

# %% cosine sims between feature embeddings

t.manual_seed(2)
W = t.randn(2,5)
W_normed = W / W.norm(dim=0, keepdim=True)

imshow(
    W_normed.T @ W_normed,
    titel="Cosine sim of each pair of 2D feature embeddings",
    width=600
)