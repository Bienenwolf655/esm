from gensim.models.doc2vec import Doc2Vec
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import (
    TransformerLayer,
    LearnedPositionalEmbedding,
    RobertaLMHead,
    ESM1bLayerNorm,
    ContactPredictionHead,
    Classifier,
)


class EmbBenchProtBert(nn.Module):
    @classmethod
    def add_args(cls, parser):
            parser.add_argument(
                "--num_layers", default=33, type=int, metavar="N", help="number of layers"
            )
            parser.add_argument(
                "--embed_dim", default=1280, type=int, metavar="N", help="input size transformer layer"
            )
            parser.add_argument(
                "--logit_bias", action="store_true", help="whether to apply bias to logits"
            )
            parser.add_argument('--arch', default='roberta_large', type=str)
            
            parser.add_argument('--layers', default =33, type = int)
            
            parser.add_argument('--max_positions', default=1024, type = int)
            
            parser.add_argument(
                "--ffn_embed_dim",
                default=5120,
                type=int,
                metavar="N",
                help="embedding dimension for FFN",
            )
            parser.add_argument(
                "--attention_heads",
                default=20,
                type=int,
                metavar="N",
                help="number of attention heads",
            )
            parser.add_argument(
                "--embed_dim_prev",
                default=1280,
                type=int,
                metavar="N",
                help="embedding dimension",
            )
            parser.add_argument(
                "--classifier",
                type = int,
                default=10,
                help="whether to predeict labels or not and the number of classes to predict",
            )

    def weights_from_pretrained(self, pretrained_model):
        pretrained_dict = nn.Sequential(*list(pretrained_model.children())[1:]).state_dict()
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def __init__(self, args, alphabet):
        super().__init__()
        self.args = args
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        self.emb_layer_norm_before = getattr(self.args, "emb_layer_norm_before", False)
        self.model_version = "ESM-1b"
        self._init_submodules_esm1b()

    def _init_submodules_common(self):
        # -c for elmo
        self.embed_tokens = nn.Embedding(
            self.alphabet_size, self.args.embed_dim_prev, padding_idx=self.padding_idx
        )
        #embs_npa = np.load('../Glove/embs_npa.npy')
        #self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embs_npa).float())

        #initilalize transformer layers, the number is determined by self.args.layers
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.args.embed_dim,
                    self.args.ffn_embed_dim,
                    self.args.attention_heads,
                    add_bias_kv=(self.model_version != "ESM-1b"),
                    use_esm1b_layer_norm=(self.model_version == "ESM-1b"),
                )
                for _ in range(self.args.num_layers)
            ]
        )

        self.contact_head = ContactPredictionHead(
            self.args.layers * self.args.attention_heads,
            self.prepend_bos,
            self.append_eos,
            eos_idx=self.eos_idx,
        )

    def _init_submodules_esm1b(self):
        if self.args.classifier is not None:
            self.classifier = Classifier(self.args.embed_dim, self.args.max_positions, self.args.classifier)
        self._init_submodules_common()
        self.input_scale = nn.Linear(self.args.embed_dim_prev, self.args.embed_dim)
        self.embed_scale = 1
        self.embed_positions = LearnedPositionalEmbedding(
            self.args.max_positions, self.args.embed_dim_prev, self.padding_idx
        )
        self.emb_layer_norm_before = (
            ESM1bLayerNorm(self.args.embed_dim) if self.emb_layer_norm_before else None
        )
        self.emb_layer_norm_after = ESM1bLayerNorm(self.args.embed_dim)
        # Head for masked language modeling. -- c for Elmo
        #self.lm_head = RobertaLMHead(
        #    embed_dim=self.args.embed_dim,
        #    output_dim=self.embed_tokens.num_embeddings,
        #    weight=self.embed_tokens.weight,
        #)


    def forward(self, tokens, repr_layers=[33], need_head_weights=False, return_contacts=False):
        if return_contacts:
            need_head_weights = True

        assert tokens.ndim == 2                     # ndim gets dimention of input vector -- c for Elmo
        padding_mask = tokens.eq(self.padding_idx)  # What this means is that wherever you have an item equal to padding_idx,
                                                    # the output of the embedding layer at that index will be all zeros.

        x  = self.embed_scale * self.embed_tokens(tokens)    # get embedidng from vocab tokens are the idx for the embed_tokens matrix vocab(nn.embedding)                                                    # can be used to scale the embedding with factor self.embed_scale --c for Elmo

        if getattr(self.args, "token_dropout", False):
             x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
             # x: B x T x C
             mask_ratio_train = 0.15 * 0.8
             src_lengths = (~padding_mask).sum(-1)
             mask_ratio_observed = (tokens == self.mask_idx).sum(-1).float() / src_lengths
             x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        # add LearnedPositionalEmbedding to input which simply computes the position of every token and embedds it derived from nn.embedding
        x = x + self.embed_positions(tokens)  #--c for ELmo
        # x = self.embed_positions(tokens)     # --c for Elmo
        # x = tokens # --c for Elmo
        if self.model_version == "ESM-1b":
            if self.emb_layer_norm_before:
                # normalizarion layer
                x = self.emb_layer_norm_before(x)
            if padding_mask is not None:
                 # pad with the padding mask as layout
                x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        x = self.input_scale(x)
        repr_layers = set(repr_layers)
        hidden_representations = {}
        # if you repr the embedding layer then it is now saved in the hidden repr dic
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None
        
        # use actual transformer layer for i in self.layer and get output in x and save attention in attn
        for layer_idx, layer in enumerate(self.layers):
            x, attn = layer(
                x, self_attn_padding_mask=None, need_head_weights=need_head_weights
            )

            # if the used layer is in the repr_layers set for which we don't want the ouput then print it to the hidden_representations dict
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            
            # simply save self_attention
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))
        
        
        if self.model_version == "ESM-1b":
            # mini batch layer normalization
            x = self.emb_layer_norm_after(x)
            x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

            # last hidden representation should have layer norm applied
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x
            # Head for masked language modeling. --c for Elmo
           # x = self.lm_head(x) 
        else:
            x = F.linear(x, self.embed_out, bias=self.embed_out_bias)
            x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        result = {"logits": x, "representations": hidden_representations}
        
        if self.args.classifier is not None:
            mean_repr = result["representations"][33].mean(1)
            label_pred = self.classifier(mean_repr)
            result['classification'] = label_pred 
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            # if padding_mask is not None:
            #     attention_mask = 1 - padding_mask.type_as(attentions)
            #     attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
            #     attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
            if return_contacts:
                contacts = self.contact_head(tokens, attentions)
                result["contacts"] = contacts

        return result

    def predict_contacts(self, tokens):
        return self(tokens, return_contacts=True)["contacts"]

    @property
    def num_layers(self):
        return self.args.layers
