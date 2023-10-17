# 3D version of Transformer decoder; Copyright Johns Hopkins University
#  Modified from Mask2former

import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from fvcore.common.registry import Registry

from .position_encoding import PositionEmbeddingSine
from torch.cuda.amp import autocast

from nn_transunet.networks.d2util import configurable, Conv2d
from nn_transunet.networks.vit_modeling import LayerScale

TRANSFORMER_DECODER_REGISTRY = Registry("TRANSFORMER_MODULE")

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False, is_mhsa_float32=False, use_layer_scale=False):
        super().__init__()
        self.is_mhsa_float32 = is_mhsa_float32
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.ls1 = LayerScale(d_model, init_values=1e-5) if use_layer_scale else nn.Identity()

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        if self.is_mhsa_float32:
            with autocast(enabled=False):
                tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        else:
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0] 
        tgt = tgt + self.dropout(self.ls1(tgt2))
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        if self.is_mhsa_float32:
            with autocast(enabled=False):
                tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        else:
            tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(self.ls1(tgt2))
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False, is_mhsa_float32=False, use_layer_scale=False):
        super().__init__()
        self.is_mhsa_float32 = is_mhsa_float32
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.ls1 = nn.Identity()

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        if self.is_mhsa_float32:
            with autocast(enabled=False):
                tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                        key=self.with_pos_embed(memory, pos),
                                        value=memory, attn_mask=memory_mask,
                                        key_padding_mask=memory_key_padding_mask)[0]
        else:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                    key=self.with_pos_embed(memory, pos),
                                    value=memory, attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask)[0]  
        tgt = tgt + self.dropout(self.ls1(tgt2))
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        if self.is_mhsa_float32:
            with autocast(enabled=False):
                tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                        key=self.with_pos_embed(memory, pos),
                                        value=memory, attn_mask=memory_mask,
                                        key_padding_mask=memory_key_padding_mask)[0]

        else:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                    key=self.with_pos_embed(memory, pos),
                                    value=memory, attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask)[0]
        
        tgt = tgt + self.dropout(self.ls1(tgt2))

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False, use_layer_scale=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.ls1 = LayerScale(d_model, init_values=1e-5) if use_layer_scale else nn.Identity()

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(self.ls1(tgt))
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(self.ls1(tgt2))
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@TRANSFORMER_DECODER_REGISTRY.register()
class MultiScaleMaskedTransformerDecoder3d(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        non_object: bool,
        num_feature_levels: int, # new
        is_masking: bool, # new
        is_masking_argmax: bool, # new
        is_mhsa_float32: bool, # new
        no_max_hw_pe: bool, # new
        use_layer_scale: bool, # new
    ):
        super().__init__()
        
        self.no_max_hw_pe = no_max_hw_pe
        self.is_masking = is_masking
        self.is_masking_argmax = is_masking_argmax
        self.num_classes = num_classes
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 3
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                    is_mhsa_float32=is_mhsa_float32,
                    use_layer_scale=use_layer_scale,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                    is_mhsa_float32=is_mhsa_float32,
                    use_layer_scale=use_layer_scale,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                    use_layer_scale=use_layer_scale,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.num_feature_levels = num_feature_levels 
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes+int(non_object))
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        if isinstance(cfg, dict):
            ret = {}
            ret["in_channels"] = in_channels
            ret["mask_classification"] = mask_classification
            ret["num_classes"] = cfg["num_classes"] 
            ret["hidden_dim"] = cfg["hidden_dim"]
            ret["num_queries"] = cfg["num_queries"]
            ret["nheads"] = cfg["nheads"]
            ret["dim_feedforward"] = cfg["dim_feedforward"]
            ret["dec_layers"] = cfg["dec_layers"] - 1
            ret["pre_norm"] = cfg["pre_norm"]
            ret["enforce_input_project"] = cfg["enforce_input_project"]
            ret["mask_dim"] = cfg["mask_dim"]

        else:
            ret = {}
            ret["in_channels"] = in_channels
            ret["mask_classification"] = mask_classification
            ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
            ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
            ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
            ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
            ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
            assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
            ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
            ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
            ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ
            ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        return ret

    def forward(self, x, mask_features, mask = None):
        if self.num_feature_levels>1 and not isinstance(x, torch.Tensor):
            assert len(x) == self.num_feature_levels, "x {} num_feature_levels {} ".format(x.shape, self.num_feature_levels)
        else:
            x = [x]
        src = []
        pos = []
        size_list = []

        del mask


        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-3:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None]) 
            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1) # p.e.
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0], mask_classification=self.mask_classification)
        if self.mask_classification:
            predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            if not self.is_masking:
                attn_mask = None 

            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,
                pos=pos[level_index] if not self.no_max_hw_pe else None, query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels], mask_classification=self.mask_classification)
            if self.mask_classification:
                predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)


        if self.mask_classification:
            assert len(predictions_class) == self.num_layers + 1

        out = {
            'pred_logits': predictions_class[-1] if self.mask_classification else None,
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            )
        }

        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size, mask_classification=True):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        

        outputs_class = self.class_embed(decoder_output) if mask_classification else None  # (b, num_query, n_class+1)

        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bcdhw->bqdhw", mask_embed, mask_features)

        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="trilinear", align_corners=False) 
        if self.is_masking_argmax:
            attn_mask = torch.argmax(attn_mask.flatten(2), dim=1)
            attn_mask = nn.functional.one_hot(attn_mask, num_classes=self.num_classes)
            attn_mask = attn_mask.permute((0, 2, 1)).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1).bool()
        else:
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()

        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
        
