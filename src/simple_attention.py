# %% [markdown]
#
# # Attention? Attention???
#
# This file contains implementations of some attention mechanisms as described
# in  Lilian Weng's excellent blog post [Attention?
# Attention!](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html).

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from torch import Tensor
from typing import Optional

# %% [markdown]
# ## A Base Class
#
# Attention mechanisms compute a weighted average of some values. In the
# original papers, the inputs were explicitly given as hidden states of some
# recurrent network; later some authors used a more abstract description in
# terms of queries, keys, and values.
#
# We can then describe an attention mechanism as a function that computes some
# score based on the query and keys, then normalizes the result and multiplies
# it with the values. Various attention mechanisms differ in how the score is
# computed.

# $$ \mathrm{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \mathrm{softmax}(\mathrm{score}(\mathbf{Q}, \mathbf{K})\otimes\mathbf{V})$$

# We can introduce an abstract base class that encapsulates the core
# functionality and then define subclasses for the different types of attention.
#
# To keep the interface similar to the attention mechanism from torchnlp, and
# also to make the class slightly more convenient to use, we make `values` an
# optional parameter and use the value of `keys` if it is not provided.


# %%
class AttentionBase(nn.Module, ABC):
    """The base class for all attention modules."""
    def __init__(self, dimensions: int):
        super().__init__()
        self.dimensions: int = dimensions

    @abstractmethod
    def score(self, query: Tensor, key: Tensor) -> Tensor:
        """Compute how well query matches key."""

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Optional[Tensor] = None):
        """Perform the forward computation of the attention layer."""
        if value is None:
            value = key
        score = self.score(query, key)
        weights = F.softmax(score, dim=1)
        return weights.bmm(value)


# %% [markdown]
# ## Dot Product Attention
#
# Probably the simplest type of attention mechanism: We simply multiply the keys
# and the values.


# %%
class DotProductAttention(AttentionBase):
    """Compute simple dot-product attention.

    Note that this attention mechanism does not have any parameters that can be
    trained; it relies on its inputs being multiplied by a paramter matrix when
    it is used as component in a multi-head attention mechanism."""
    def score(self, query: Tensor, key: Tensor) -> Tensor:
        return query.bmm(key.transpose(1, 2))


# %%
# Batch size: 16
# Number of key/value pairs: 8
# Dimensionality of the attention mechanism: 128
_dpa = DotProductAttention(128)
_query = torch.randn(16, 1, 128)
_key = torch.randn(16, 8, 128)
_value = torch.randn(16, 8, 128)
assert _dpa.score(_query, _key).shape == torch.Size([16, 1, 8])
assert _dpa(_query, _key).shape == torch.Size([16, 1, 128])


# %%
class ScaledDotProductAttention(AttentionBase):
    """Compute scaled dot-product attention.

    This algorithm is very similar to `DotProductAttention`, except that it
    scales the score by the square root of the number of dimensions used to get
    steeper gradients from the softmax.

    Note that this attention mechanism does not have any parameters that can be
    trained; it relies on its inputs being multiplied by a paramter matrix when
    it is used as component in a multi-head attention mechanism."""
    def __init__(self, dimensions: int):
        super().__init__(dimensions)
        self.sqrt_dimensions: Tensor = torch.sqrt(
            torch.tensor([dimensions]).to(torch.float))

    def score(self, query: Tensor, key: Tensor) -> Tensor:
        return query.bmm(key.transpose(1, 2)) / self.sqrt_dimensions


# %%
# Batch size: 16
# Number of key/value pairs: 8
# Dimensionality of the attention mechanism: 128
_sdpa = ScaledDotProductAttention(128)
_query = torch.randn(16, 1, 128)
_key = torch.randn(16, 8, 128)
_value = torch.randn(16, 8, 128)
assert _sdpa.score(_query, _key).shape == torch.Size([16, 1, 8])
assert _sdpa(_query, _key).shape == torch.Size([16, 1, 128])


# %%
class GeneralDotProductAttention(AttentionBase):
    """Compute weighted scaled dot product attention.

    This algorithm computes a scaled dot product attention, but it inserts a
    trainable weight matrix into the attention layer. Note that in Lilian Weng's
    blog post the general dot product attention is unscaled."""
    def __init__(self, dimensions: int):
        super().__init__(dimensions)
        self.weights = nn.Linear(dimensions, dimensions, bias=False)
        self.sqrt_dimensions: Tensor = torch.sqrt(
            torch.tensor([dimensions]).to(torch.float))

    def score(self, query: Tensor, key: Tensor) -> Tensor:
        weighted_keys = self.weights(key).transpose(1, 2)
        return query.bmm(weighted_keys) / self.sqrt_dimensions


# %%
# Batch size: 16
# Number of key/value pairs: 8
# Dimensionality of the attention mechanism: 128
_gdpa = GeneralDotProductAttention(128)
_query = torch.randn(16, 1, 128)
_key = torch.randn(16, 8, 128)
_value = torch.randn(16, 8, 128)
assert _gdpa.score(_query, _key).shape == torch.Size([16, 1, 8])
assert _gdpa(_query, _key).shape == torch.Size([16, 1, 128])

# %%
