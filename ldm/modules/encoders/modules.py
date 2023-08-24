import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel, CLIPVisionModel

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
        self.image_encoder = CLIPVisionModel.from_pretrained(version)
        self.image_projection = nn.Linear(1024, 768, bias=False)  # learnable

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(
            self.image_projection.weight,
            std=1024 ** -0.5 * 0.02,
        )

    def freeze(self):
        self.transformer = self.transformer.eval()
        self.image_encoder = self.image_encoder.eval()
        for param in self.parameters():
            param.requires_grad = False
        for param in self.image_projection.parameters():  # unfreeze the projection layer
            param.requires_grad = True

    def forward(self, text_image_dict):
        assert isinstance(text_image_dict, dict)
        text = text_image_dict["text"]  # List[String]
        image = text_image_dict["image"]  # (B,C,H,W), in [-1,1]

        ''' 1. text '''
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

        ''' 2. image '''
        z_image = self.image_encoder(image).last_hidden_state
        z_image = self.image_projection(z_image)

        return torch.cat([z, z_image], dim=1)  # (B,77+257,768)

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
    # outputs = image_encoder(xi)
    # print("[image] encoder output:")
    # for k in outputs.keys():
    #     print(f"|---{k}: {outputs[k].shape}; after projection: {visual_proj(outputs[k]).shape}")
    #
    # image_encoder_proj = CLIPVisionModelWithProjection.from_pretrained(version)
    # outputs = image_encoder_proj(xi)
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
    net = FrozenCLIPTextImageEmbedder().cuda()
    z = net({"text": text, "image": xi.cuda()})
    print("[FrozenCLIPTextImageEmbedder] output:", z.shape)
