import timm
import torch
import torch.nn as nn
import math, json
from functools import reduce
from operator import mul

import clip
from class_name import text


class Source_Transformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed

        return self.pos_drop(x)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x):
        pre_x = x[:, 0]
        x = self.fc_norm(pre_x)
        return self.head(x), pre_x

    def forward(self, x):
        x = self.forward_features(x)
        x, pre_x = self.forward_head(x)
        return {"outputs": x, "pre_outputs": pre_x}
