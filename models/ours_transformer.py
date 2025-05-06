import timm
import torch
import torch.nn as nn
import math, json
from functools import reduce
from operator import mul
import clip
import torch.nn.functional as F


class OURS_Transformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, args, **kwargs):
        # super().__init__(**kwargs)
        super(OURS_Transformer, self).__init__(
            img_size=args.img_size, num_classes=args.num_classes)
        self.args = args
        self.dual_prompt_tokens = args.dual_prompt_tokens
        prompt_dim = self.embed_dim
        self.num_layers = len(self.blocks)
        self.block_gradients = [None] * self.num_layers
        # Register backward hooks
        for i, block in enumerate(self.blocks):
            block.register_full_backward_hook(self.save_gradients(i))
        self.prompt_deep = args.prompt_deep
        self.prompt_emb = torch.zeros(self.dual_prompt_tokens, prompt_dim).cuda()
        self.domain_predictors = nn.ModuleList([
            LoRA(in_dim=prompt_dim, r=64, out_dim=prompt_dim) for _ in range(self.num_layers)])
        self.G_predictors = nn.ModuleList([
            LoRA(in_dim=prompt_dim, r=4, out_dim=prompt_dim) for _ in range(self.num_layers)])
        self.E_predictors = nn.ModuleList([
            LoRA(in_dim=prompt_dim, r=64, out_dim=prompt_dim) for _ in range(self.num_layers)])

    def save_gradients(self, index):
        def hook(module, grad_input, grad_output):
            # Save gradients of the specific block
            self.block_gradients[index] = grad_output[0]

        return hook

    def clear_gradients(self):
        """Clears the stored gradients after usage."""
        self.block_gradients = [None] * self.num_layers

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

    def forward_deep_prompt(self, x, importance):
        B = x.shape[0]
        hidden_layers = []
        for i in range(self.num_layers):
            if i == 0:
                prompt = self.prompt_emb.expand(B, -1, -1)
                prompt = prompt * (32 ** 2 / 384 ** 2) if self.args.num_classes in [10, 100] else prompt
                x = torch.cat((prompt, x), dim=1)
                hidden_states = self.blocks[i](x)
            else:
                domain = self.domain_predictors[i](hidden_states)
                E_prompt = self.E_predictors[i](domain[:, self.dual_prompt_tokens, ]).unsqueeze(dim=1)
                G_prompt = self.G_predictors[i](
                    hidden_states[:, self.dual_prompt_tokens, ] - domain[:, self.dual_prompt_tokens, ]).unsqueeze(dim=1)

                Balance_Prompt = torch.cat((E_prompt, G_prompt), dim=1)
                Balance_Prompt = Balance_Prompt * (32 ** 2 / 384 ** 2) if self.args.num_classes in [10, 100] else Balance_Prompt
                hidden_states = hidden_states - domain
                hidden_states = torch.cat(
                    (Balance_Prompt, hidden_states[:, self.dual_prompt_tokens:, :]), dim=1)
                hidden_states = self.blocks[i](hidden_states)
            hidden_layers.append(hidden_states)
        return hidden_states, hidden_layers

    def forward_features(self, x, importance):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        if self.prompt_deep:
            x, hidden_layer = self.forward_deep_prompt(x, importance)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x, hidden_layer

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.dual_prompt_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:,
                                                                                             self.dual_prompt_tokens]
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)

    def forward(self, x, importance=None):
        x, hidden_layer = self.forward_features(x, importance)
        x = self.forward_head(x)
        return x, hidden_layer




class LoRA(nn.Module):
    def __init__(self, in_dim, r, out_dim):
        super().__init__()
        self.act = nn.GELU()
        self.lora_B = nn.Linear(in_dim, r, bias=False)
        self.lora_A = nn.Linear(r, out_dim, bias=False)
        nn.init.normal_(self.lora_B.weight, std=1 / r ** 2)
        nn.init.zeros_(self.lora_A.weight)

    def forward(self, x):
        x = self.act(self.lora_B(x))
        x = self.act(self.lora_A(x))
        return x
