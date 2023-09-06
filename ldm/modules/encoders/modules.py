import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel, CLIPVisionModel
from transformers import CLIPVisionModelWithProjection
from transformers import PvtModel

import open_clip
from ldm.util import default, count_params


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class IdentityEncoder(AbstractEncoder):

    def encode(self, x):
        return x


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class', ucg_rate=0.1):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.n_classes = n_classes
        self.ucg_rate = ucg_rate

    def forward(self, batch, key=None, disable_dropout=False):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        if self.ucg_rate > 0. and not disable_dropout:
            mask = 1. - torch.bernoulli(torch.ones_like(c) * self.ucg_rate)
            c = mask * c + (1-mask) * torch.ones_like(c)*(self.n_classes-1)
            c = c.long()
        c = self.embedding(c)
        return c

    def get_unconditional_conditioning(self, bs, device="cuda"):
        uc_class = self.n_classes - 1  # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)
        uc = torch.ones((bs,), device=device) * uc_class
        uc = {self.key: uc}
        return uc


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class FrozenT5Embedder(AbstractEncoder):
    """Uses the T5 transformer encoder for text"""
    def __init__(self, version="google/t5-v1_1-large", device="cuda", max_length=77, freeze=True):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length   # TODO: typical value?
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = [
        "last",
        "pooled",
        "hidden"
    ]
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77,
                 freeze=True, layer="last", layer_idx=None):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        """ Input text is a list of text strings """
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer=="hidden")
        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        return z

    def encode(self, text):
        return self(text)


class FrozenCLIPTextImageEmbedder(AbstractEncoder):
    """Uses the CLIP encoder for text and image (from huggingface)"""
    TEXT_LAYERS = [
        "last",
        "pooled",
        "hidden"
    ]
    IMAGE_LAYERS = [
        "last",
        "projection"
    ]
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77,
                 freeze=True, text_layer="last", text_layer_idx=None,
                 use_text: bool = False,
                 image_layer: str = "projection",
                 ):
        super().__init__()
        assert text_layer in self.TEXT_LAYERS
        assert image_layer in self.IMAGE_LAYERS
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.use_text = use_text
        # self.image_encoder = CLIPVisionModel.from_pretrained(version)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(version)
        image_in_channels = 768 if image_layer == "projection" else 1024
        self.image_layer = image_layer
        self.image_projection = nn.Linear(image_in_channels, 768, bias=False)  # learnable 2nd projection

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.text_layer = text_layer
        self.text_layer_idx = text_layer_idx
        if text_layer == "hidden":
            assert text_layer_idx is not None
            assert 0 <= abs(text_layer_idx) <= 12

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(
            self.image_projection.weight,
            std=1024 ** -0.5 * 0.02,
        )

    def named_trainable_params_list(self):
        named_trainable = []
        for name, param in self.named_parameters():
            if param.requires_grad is True:
                named_trainable.append((name, param))
        return named_trainable

    def freeze(self):
        self.transformer = self.transformer.eval()
        self.image_encoder = self.image_encoder.eval()
        for param in self.parameters():
            param.requires_grad = False
        for param in self.image_projection.parameters():  # unfreeze the projection layer
            param.requires_grad = True
        print("[FrozenCLIPTextImageEmbedder] params except image_projection frozen.")

    def forward(self, text_image_dict):
        assert isinstance(text_image_dict, dict)
        text = text_image_dict["text"]  # List[String]
        image = text_image_dict["image"]  # (B,C,H,W), in [-1,1]

        ''' 1. text '''
        catted = []
        if self.use_text:
            batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length,
                                            return_length=True, return_overflowing_tokens=False,
                                            padding="max_length", return_tensors="pt")
            tokens = batch_encoding["input_ids"].to(self.device)
            outputs = self.transformer(input_ids=tokens, output_hidden_states=self.text_layer=="hidden")
            if self.text_layer == "last":
                z = outputs.last_hidden_state  # (B,77,768)
            elif self.text_layer == "pooled":
                z = outputs.pooler_output[:, None, :]
            else:
                z = outputs.hidden_states[self.layer_idx]
            catted.append(z)

        ''' 2. image '''
        # z_image = self.image_encoder(image).last_hidden_state
        # z_image = self.image_projection(z_image)
        #
        # # return torch.cat([z, z_image], dim=1)  # (B,77+257,768)
        # return torch.cat([z_image], dim=1)  # (B,257,768)  # TODO: single or multiple?

        if self.image_layer == "projection":
            z_image = self.image_encoder(image).image_embeds  # (B,768)
            z_image = z_image[:, None, :]  # (B,1,768)
        else:
            z_image = self.image_encoder(image).last_hidden_state  # (B,257,1024)
        z_image = self.image_projection(z_image)  # (B,y,768)
        catted.append(z_image)
        return torch.cat(catted, dim=1)  # (B,77+y,768)

    def encode(self, text_image_dict):
        return self(text_image_dict)


class FrozenOpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        #"pooled",
        "last",
        "penultimate"
    ]
    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77,
                 freeze=True, layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)


class FrozenCLIPT5Encoder(AbstractEncoder):
    def __init__(self, clip_version="openai/clip-vit-large-patch14", t5_version="google/t5-v1_1-xl", device="cuda",
                 clip_max_length=77, t5_max_length=77):
        super().__init__()
        self.clip_encoder = FrozenCLIPEmbedder(clip_version, device, max_length=clip_max_length)
        self.t5_encoder = FrozenT5Embedder(t5_version, device, max_length=t5_max_length)
        print(f"{self.clip_encoder.__class__.__name__} has {count_params(self.clip_encoder)*1.e-6:.2f} M parameters, "
              f"{self.t5_encoder.__class__.__name__} comes with {count_params(self.t5_encoder)*1.e-6:.2f} M params.")

    def encode(self, text):
        return self(text)

    def forward(self, text):
        clip_z = self.clip_encoder.encode(text)
        t5_z = self.t5_encoder.encode(text)
        return [clip_z, t5_z]


class FrozenPVTImageEncoder(AbstractEncoder):
    """Uses the PVT encoder for image pyramid features (from huggingface)"""
    def __init__(self, version="Zetatech/pvt-large-224", device="cuda",
                 freeze=True,
                 fpn_layer_indices: list = None,
                 mlp_in_channels: list = None,
                 **kwargs
                 ):
        super().__init__()
        self.pvt_model = PvtModel.from_pretrained(version)
        if fpn_layer_indices is None:
            fpn_layer_indices = [(0, 0, 1, 2),  # stage 1
                                 (4, 6, 8, 10), # stage 2
                                 (23, 37),      # stage 3
                                 (38, 40, 41),  # stage 4
                                 ]
        self.fpn_layer_indices = fpn_layer_indices
        if mlp_in_channels is None:
            mlp_in_channels = [64, 128, 320, 512]  # by stage
        self.mlp_in_channels = mlp_in_channels
        self.image_projections = torch.nn.ModuleList()
        for k in range(len(self.mlp_in_channels)):
            self.image_projections.append(
                nn.Linear(self.mlp_in_channels[k], 768, bias=False)  # shared and learnable projection
            )

        self.device = device
        if freeze:
            self.freeze()

        self._init_weights()

    def _init_weights(self):
        for proj in self.image_projections:
            nn.init.normal_(
                proj.weight,
                std=1024 ** -0.5 * 0.02,
            )

    def named_trainable_params_list(self):
        named_trainable = []
        for name, param in self.named_parameters():
            if param.requires_grad is True:
                named_trainable.append((name, param))
        return named_trainable

    def freeze(self):
        self.pvt_model = self.pvt_model.eval()
        for param in self.parameters():
            param.requires_grad = False
        for proj in self.image_projections:
            for param in proj.parameters():  # unfreeze the projection layer
                param.requires_grad = True
        print("[FrozenPVTImageEncoder] params except image_projections frozen.")

    def forward(self, text_image_dict):
        assert isinstance(text_image_dict, dict)
        text = text_image_dict["text"]  # List[String], not used
        image = text_image_dict["image"]  # (B,C,H,W), in [-1,1]

        ''' 1. image '''
        ret_feats = []
        fpn_feats = self.pvt_model(image, output_hidden_states=True).hidden_states
        # (B,3136=56x56,64)x3; (B,784=28x28,128)x8; (B,196=14x14,320)x27; (B,50=7x7+1,512)x4;
        for stage in range(len(self.fpn_layer_indices)):
            proj = self.image_projections[stage]
            for fpn_layer_index in self.fpn_layer_indices[stage]:
                fpn_feat = proj(fpn_feats[fpn_layer_index])  # (B,?,768)
                ret_feats.append(fpn_feat)
        return ret_feats

    def encode(self, text_image_dict):
        return self(text_image_dict)


if __name__ == "__main__":
    import torch
    import transformers
    from transformers import AutoTokenizer, CLIPTextModel, CLIPProcessor, CLIPVisionModel, CLIPModel
    from transformers import CLIPVisionModelWithProjection

    transformers.utils.logging.set_verbosity_error()

    text = ["hello", "world"]
    xt = torch.ones((2, 7))
    xt = xt.long()
    xi = torch.randn((2, 3, 224, 224)).clamp(-1, 1)

    visual_proj = nn.Linear(1024, 768, bias=False)

    version = "openai/clip-vit-large-patch14"

    ''' text '''
    # text_tokenizer = CLIPTokenizer.from_pretrained(version)
    # text_encoder = CLIPTextModel.from_pretrained(version)
    #
    # tokens = text_tokenizer(text, max_length=77, return_length=True,
    #                         return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    # tokens = tokens["input_ids"]
    # print("[text] tokenizer output:", tokens.shape)
    # outputs = text_encoder(input_ids=tokens)
    # print("[text] encoder output:", outputs.last_hidden_state.shape)
    #
    # outputs = text_encoder(input_ids=xt)
    # print("[text] encoder directly output:", outputs.last_hidden_state.shape)

    ''' image '''
    # image_encoder = CLIPVisionModel.from_pretrained(version)
    # outputs = image_encoder(xi, output_hidden_states=True)
    # # {"last_hidden_state":(B,257,1024), "pooler_output":(B,1024)}
    # print("[image] encoder output:")
    # for k in outputs.keys():
    #     v = outputs[k]
    #     if isinstance(v, tuple):
    #         print(f"|---{k}: {len(v)};")
    #         for i, x in enumerate(v):
    #             print(f"   |---{i}: {x.shape}; after projection: {visual_proj(x).shape}")
    #     else:
    #         print(f"|---{k}: {v.shape}; after projection: {visual_proj(v).shape}")
    #
    # image_encoder_proj = CLIPVisionModelWithProjection.from_pretrained(version)
    # outputs = image_encoder_proj(xi)  # {"last_hidden_state":(B,257,1024), "image_embeds":(B,768)}
    # print("[image] encoder_with_projection output:")
    # for k in outputs.keys():
    #     print(f"|---{k}: {outputs[k].shape}")

    ''' both '''
    # both_encoder = CLIPModel.from_pretrained(version)
    # output_text = both_encoder.get_text_features(xt)
    # output_image = both_encoder.get_image_features(xi)
    # print("[both] encoder output:")
    # print(f"|---text: {output_text.shape}")
    # print(f"|---image: {output_image.shape}")

    ''' my FrozenCLIPTextImageEmbedder '''
    # net = FrozenCLIPTextImageEmbedder(
    #     image_layer="last"
    # ).cuda()
    # z = net({"text": text, "image": xi.cuda()})
    # print("[FrozenCLIPTextImageEmbedder] output:", z.shape)

    ''' Pyramid Vision Transformer '''
    # from transformers import AutoImageProcessor, PvtModel, PvtForImageClassification
    # import torch
    # import numpy as np
    # from einops import rearrange
    # from PIL import Image
    #
    # image = Image.open("./test_imgs/cloth_00013_00.jpg").convert("RGB").resize((224, 224))
    # image = np.array(image)
    #
    # image_processor = AutoImageProcessor.from_pretrained("Zetatech/pvt-tiny-224")
    # model = PvtModel.from_pretrained("Zetatech/pvt-tiny-224")
    # # model = PvtForImageClassification.from_pretrained("Zetatech/pvt-tiny-224")
    #
    # with torch.no_grad():
    #     # inputs = image_processor(image, return_tensors="pt")
    #     # outputs = model(**inputs, output_hidden_states=True)
    #
    #     inputs = torch.from_numpy(image).unsqueeze(0).float()
    #     inputs = rearrange(inputs, "n h w c -> n c h w").contiguous()
    #     inputs = (inputs / 127.5) - 1.
    #     outputs = model(inputs, output_hidden_states=True)
    #     print(outputs.keys())
    #
    # if hasattr(outputs, "last_hidden_state"):
    #     last_hidden_states = outputs.last_hidden_state
    #     print(last_hidden_states.shape)
    #     for i, hidden in enumerate(outputs.hidden_states):
    #         add_str = ""
    #         if hidden.shape == last_hidden_states.shape:
    #             add_str = f", diff = {torch.abs(last_hidden_states - hidden).sum()}"
    #         print(f"|---{i}: {hidden.shape} {add_str}")
    #     # (B,3136=56x56,64)x3; (B,784=28x28,128)x8; (B,196=14x14,320)x27; (B,50=7x7+1,512)x4;
    #
    # if hasattr(outputs, "logits"):
    #     # model predicts one of the 1000 ImageNet classes
    #     logits = outputs.logits
    #     predicted_label = logits.argmax(-1).item()
    #     print("Predict class:", model.config.id2label[predicted_label])

    ''' my FrozenPVTImageEncoder '''
    net = FrozenPVTImageEncoder(

    ).cuda()
    feats = net({"text": text, "image": xi.cuda()})
    for i, feat in enumerate(feats):
        print(f"[FrozenPVTImageEncoder] output {i}:", feat.shape)
