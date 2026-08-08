from typing import Optional, Type, Union, Tuple, Callable

import torch.nn as nn

from ai_engine.models.utils import to_2tuple


class MLP(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = None,
            out_features: Optional[int] = None,
            act_layer: Union[Type[nn.Module], Callable[..., nn.Module], nn.Module] = nn.GELU,
            norm_layer: Optional[Type[nn.Module]] = None,
            bias: Union[bool, Tuple[bool, bool]] = True,
            drop: Union[float, Tuple[float, float]] = 0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])

        if isinstance(act_layer, nn.Module):
            self.act = act_layer
        else:
            self.act = act_layer()

        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x