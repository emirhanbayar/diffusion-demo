"""Fully connected layers."""

import torch
import torch.nn as nn

from .embed import LearnableSinusoidalEncoding, ClassEmbedding
from .utils import make_activation


class CondDense(nn.Module):
    """Conditional fully connected layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str | None = "leaky_relu",
        embed_dim: int | None = None,
        num_classes: int | None = None,
    ):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.activation = make_activation(activation)

        # create multi-layer positional embedding
        if embed_dim is not None:
            self.emb = LearnableSinusoidalEncoding(
                [
                    embed_dim,
                    out_features,
                    out_features,
                ],  # stack two learnable dense layers after the sinusoidal encoding
                activation=activation,
            )
        else:
            self.emb = None

        # create lookup table class embedding
        if num_classes is not None:
            self.class_embed = ClassEmbedding(num_classes, out_features)
        else:
            self.class_embed = None

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        cids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self.linear(x)

        # add positional embedding (conditioning)
        if t is not None and self.emb is not None:
            emb = self.emb(t)
            out = out + emb
        elif t is not None and self.emb is None:
            raise TypeError("No temporal embedding")
        elif t is None and self.emb is not None:
            raise TypeError("No time passed")

        # add class embedding
        if cids is not None and self.class_embed is not None:
            c_emb = self.class_embed(cids)
            out = out + c_emb
        elif cids is not None and self.class_embed is None:
            raise TypeError("No class embedding")
        elif cids is None and self.class_embed is not None:
            raise TypeError("No class label passed")

        if self.activation is not None:
            out = self.activation(out)

        return out
