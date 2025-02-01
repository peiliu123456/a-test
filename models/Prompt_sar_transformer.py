import timm
import torch
import torch.nn as nn
import math, json
from functools import reduce
from operator import mul
import clip
import torch.nn.functional as F


class Prompt_sar_Transformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, args, **kwargs):
        # super().__init__(**kwargs)
        super(Prompt_sar_Transformer, self).__init__(
            img_size=args.img_size, num_classes=args.num_classes)
        self.args = args
        self.dual_prompt_tokens = args.dual_prompt_tokens
        prompt_dim = self.embed_dim
        patch_size = self.patch_embed.patch_size
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))
        self.num_layers = len(self.blocks)
        self.prompt_deep = args.prompt_deep
        self.prompt_emb = nn.Parameter(torch.zeros(self.num_layers, self.dual_prompt_tokens, prompt_dim))
        nn.init.uniform_(self.prompt_emb.data, -val, val)

    def _pos_embed(self, x):  ##增加位置编码
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

    def forward_deep_prompt(self, x):
        B = x.shape[0]
        for i in range(self.num_layers):
            if i == 0:
                prompt = self.prompt_emb[i].expand(B, -1, -1)
                x = torch.cat((prompt, x), dim=1)
                hidden_states = self.blocks[i](x)
            else:
                prompt = self.prompt_emb[i].expand(B, -1, -1)
                hidden_states = torch.cat(
                    (prompt, hidden_states[:, self.dual_prompt_tokens:, :]), dim=1)
                hidden_states = self.blocks[i](hidden_states)
        return hidden_states

    def forward_features(self, x):
        x = self.patch_embed(x)  # 把 x(64,224,224,3)patch ->>(64,198,768)
        x = self._pos_embed(x)  # 增加位置信息
        x = self.forward_deep_prompt(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:  # 使用cls token 而不使用 avg pool
            x = x[:, self.dual_prompt_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:,
                                                                                             self.dual_prompt_tokens]
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
