import einops
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config, default
from ldm.data.tryon_dataset import TryOnDataset
from cldm.ddim_hacked import DDIMSamplerTryOn as DDIMSampler  # use hacked ddim

class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        """
        Callable
        @param x:
        @param timesteps:
        @param context: (B,257,768), insensitive to Channel dimension (dim=1), cloth features
        @param control: List[(B,320,64,48)x3,(B,640,32,24)x3,(B,1280,16,12)x3,(B,1280,8,6)x4], pose
        @param only_mid_control:
        """
        hs = []
        if isinstance(context, torch.Tensor):
            contexts = [context] * (len(self.input_blocks) + 1)
        elif isinstance(context, list):
            contexts = context  # 3136x4, 784x4, 196x2, 50x3
        else:
            raise TypeError("[ControlledUnetModel] context type not supported!")

        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)  # always (B,1280)
            h = x.type(self.dtype)
            for i, module in enumerate(self.input_blocks):  # len=12
                h = module(h, emb, contexts[i])
                # print(f"[DEBUG][UNet] layer {i}: context={contexts[i].shape if contexts[i] is not None else None}, len={len(contexts)}, out_h={h.shape}")
                hs.append(h)
            # h = self.middle_block(h, emb, contexts.pop())
            h = self.middle_block(h, emb, contexts[-1])
            # print(f"[DEBUG][UNet] layer m: context={contexts[-1].shape}, len={len(contexts)}, out_h={h.shape}")
            ''' input h shape:  |   context FPN
            0:  (B,9,64,48)
            1:  (B,320,64,48)
            2:  (B,320,64,48)
            3:  (B,320,64,48)
            4:  (B,320,32,24)   -   (B,28x28,128)
            5:  (B,640,32,24)   -   (B,28x28,128)
            6:  (B,640,32,24)   -   (B,28x28,128)
            7:  (B,640,16,12)   -   (B,28x28,128)
            8:  (B,1280,16,12)  -   (B,14x14,320)
            9:  (B,1280,16,12)  -   (B,14x14,320)
            10: (B,1280,8,6)    -   (B,7x7+1,512)
            11: (B,1280,8,6)    -   (B,7x7+1,512)
            m:  (B,1280,8,6)    -   (B,7x7+1,512)
            '''

        if control is not None:
            h += control.pop()

        # print("[DEBUG][UNet] contexts and output_blocks len=", len(contexts), " and ", len(self.output_blocks))
        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            # print(f"[DEBUG][UNet] out {i}: context={contexts[-1].shape if contexts[-1] is not None else None}, len={len(contexts)}, out_h={h.shape}")
            # h = module(h, emb, contexts.pop())
            # print(
            #     f"[DEBUG][UNet] out {i}: context={contexts[-i-1].shape if contexts[-i-1] is not None else None}, len={len(contexts)}, out_h={h.shape}")
            h = module(h, emb, contexts[-i-1])

        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if isinstance(context, torch.Tensor):
            contexts = [context] * (len(self.input_blocks) + 1)
        elif isinstance(context, list):
            contexts = context
        else:
            raise TypeError("[ControlNet] context type not supported!")

        guided_hint = self.input_hint_block(hint, emb, context)  # context should be useless
        # (hint:(B,3,704,512))
        # (x:(B,4,hint.H//8(88),hint.W//8(64)))
        # (emb:(B,1280))
        # (context:(B,77,768))
        # (guided_hint:(B,model_channels(320),hint.H//8(88),hint.W//8(64)))

        outs = []

        h = x.type(self.dtype)
        l = 0
        # print(
        #     f"[DEBUG][Control] input: contexts.len={len(contexts)}, in_h={h.shape}")
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):  # repeat 12 times
            context = contexts[l]
            if guided_hint is not None:
                h = module(h, emb, context)  # "emb" and "context" are spatial-agnostic of "x"; "hint" won't join CA
                h += guided_hint  # h:same shape with guided_hint
                guided_hint = None  # only applied to the 1st block
            else:
                h = module(h, emb, context)
            # print(f"[DEBUG][Control] layer {l}: context={context.shape if context is not None else None}, len={len(contexts)}, out_h={h.shape}")
            outs.append(zero_conv(h, emb, context))
            l += 1

        context = contexts[-1]  # contexts will be used by UNet, cannot pop here!
        h = self.middle_block(h, emb, context)
        # print(f"[DEBUG][Control] layer m: context={context.shape if context is not None else None}, len={len(contexts)}, out_h={h.shape}")
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control, agnostic_key,
                 drop_cond_rate: float = 0.,
                 feed_cloth_to_controlnet: bool = False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        self.agnostic_key = agnostic_key
        self.val_dataset = TryOnDataset(
            root="/cfs/yuange/datasets/VTON-HD",
            mode="test",
        )
        self.drop_cond_rate = drop_cond_rate
        assert 0. <= drop_cond_rate <= 1., "Dropping condition rate should range in [0., 1.]!"
        self.feed_cloth_to_controlnet = feed_cloth_to_controlnet

    def shared_step(self, batch, **kwargs):
        x, c = self.get_input(batch, self.first_stage_key)
        loss = self(x, c)  # call self.forward()
        return loss

    def _bhwc_to_bchw(self, x: torch.Tensor, bs=None):
        if x.ndim == 3:
            x = x.unsqueeze(-1)  # (B,H,W,1)
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        """ (Overridden) Called by self.shared_step and self.log_images
        @param: batch:
                {
                    "jpg": ground-truth person, (B,H,W,C)
                    "txt": {"prompt": List[String], "cloth": masked cloth (B,H,W,C)
                    "hint": openpose, (B,H,W,C)
                    "agnostic": cloth-agnostic person, (B,H,W,C)
                    "agnostic_mask": mask of agnostic pixels, (B,H,W,C), in [0,1]
                }
        @returns:
                ret_x: {
                    "z_x": x,
                    "z_ag": z_ag,
                    "z_ag_m": mask,
                },
                ret_c: {
                    "c_crossattn": [ori_condition:{"prompt":List[String], "cloth":(B,224,224,RGB)}],
                    "c_concat": [control:(B,3,H,W), cloth(optional):(B,3,H,W)]
                }
        """
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        # x: z of "jpg", processed by first_stage_config.target
        # c: "txt" condition, processed by cond_stage_config.target

        ag = batch[self.agnostic_key]  # "agnostic"
        # if bs is not None:
        #     ag = ag[:bs]
        # ag = ag.to(self.device)
        # ag = rearrange(ag, 'b h w c -> b c h w')
        # ag = ag.to(memory_format=torch.contiguous_format).float()
        ag = self._bhwc_to_bchw(ag, bs)
        encoder_posterior = self.encode_first_stage(ag)
        z_ag = self.get_first_stage_encoding(encoder_posterior).detach()  # z_ag: latent of "agnostic"

        z_ag_m = batch["agnostic_mask"]
        # if z_ag_m.ndim == 3:
        #     z_ag_m = z_ag_m.unsqueeze(-1)  # (B,H,W,1)
        # if bs is not None:
        #     z_ag_m = z_ag_m[:bs]
        # z_ag_m = z_ag_m.to(self.device)
        # z_ag_m = rearrange(z_ag_m, 'b h w c -> b c h w')
        # z_ag_m = z_ag_m.to(memory_format=torch.contiguous_format).float()
        z_ag_m = self._bhwc_to_bchw(z_ag_m, bs)
        z_ag_m = F.interpolate(z_ag_m, size=z_ag.shape[-2:], mode="bilinear", align_corners=True)

        control_list = []
        control = batch[self.control_key]  # "hint"
        # if bs is not None:
        #     control = control[:bs]
        # control = control.to(self.device)  # (B,H,W,RGB), in [-1,1]
        # control = einops.rearrange(control, 'b h w c -> b c h w')
        # control = control.to(memory_format=torch.contiguous_format).float()
        control = self._bhwc_to_bchw(control, bs)
        control_list.append(control)

        if self.feed_cloth_to_controlnet:
            cloth = batch["txt"]["cloth"]
            cloth = self._bhwc_to_bchw(cloth, bs)
            cloth = F.interpolate(cloth, size=control.shape[-2:], mode="bilinear", align_corners=True)
            control_list.append(cloth)

        return {"z_x": x, "z_ag": z_ag, "z_ag_m": z_ag_m}, dict(c_crossattn=[c], c_concat=control_list)

    def forward(self, x, c, *args, **kwargs):
        """ Called by self.shared_step """
        z_x = x["z_x"]
        t = torch.randint(0, self.num_timesteps, (z_x.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:  # the grads can be backward propagated here
                # call cond_stage_config.target: FrozenCLIPTextImageEmbedder
                c_cloth = c["c_crossattn"][0]["cloth"].to(self.device)  # cloth image input
                c_cloth = rearrange(c_cloth, 'b h w c -> b c h w')
                c_cloth = c_cloth.to(memory_format=torch.contiguous_format).float()
                c_text_image_dict = {
                    "text": c["c_crossattn"][0]["prompt"],
                    "image": c_cloth,
                }
                if np.random.uniform(0., 1.) >= self.drop_cond_rate:
                    c_crossattn = self.get_learned_conditioning(c_text_image_dict)  # cond_stage_config.target output
                else:
                    c_crossattn = self.get_unconditional_conditioning(z_x.shape[0])  # drop conditioning
                c = dict(c_crossattn=[c_crossattn], c_concat=c["c_concat"])
        return self.p_losses(x, c, t, *args, **kwargs)

    def p_losses(self, x_start, cond, t, noise=None):
        """ Called by self.forward
        @param x_start: {"z_x":..., "z_ag":..., "z_ag_m":...} dict of z_x, z_ag, and z_ag_m
        @param cond: {"c_crossattn(context)":[cond_stage_model outputs], "c_concat(control)":[(B,RGB,H,W)]}
        @param t: time embedding
        @param noise: noise
        """
        z_x = x_start["z_x"]
        z_ag = x_start["z_ag"]
        z_ag_m = x_start["z_ag_m"]
        noise = default(noise, lambda: torch.randn_like(z_x))
        x_noisy = self.q_sample(x_start=z_x, t=t, noise=noise)
        # ag_noisy = self.q_sample(x_start=z_ag, t=t, noise=torch.randn_like(noise))  # agnostic also added noise?
        ag_noisy = z_ag
        model_output = self.apply_model({"noisy": x_noisy,
                                         "z_ag": ag_noisy,
                                         "z_ag_m": z_ag_m,  # no need to add noise
                                         }, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = z_x
        elif self.parameterization == "eps":  # default
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(z_x, noise, t)
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(x_noisy, dict)
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        # self.model created by DDPM(super.super): DiffusionWrapper
        # self.model.diffusion_model is unet_config.target: ControlledUnetModel

        crossattn_first_element = cond['c_crossattn'][0]
        if isinstance(crossattn_first_element, torch.Tensor):
            cond_txt = torch.cat(cond['c_crossattn'], 1)  # prompts features by cond_stage_model, (B,y,768)
        elif isinstance(crossattn_first_element, list):
            cond_txt = crossattn_first_element
        else:
            raise TypeError("[ControlLDM] c_crossattn type not supported!")

        z_x_noisy = x_noisy["noisy"]
        z_ag = x_noisy["z_ag"]
        z_ag_m = x_noisy["z_ag_m"]
        z_cat = torch.cat([z_x_noisy, z_ag, z_ag_m], dim=1)  # concatenate at channel dimension
        if cond['c_concat'] is None:  # control is None
            eps = diffusion_model(x=z_cat, timesteps=t, context=cond_txt, control=None,
                                  only_mid_control=self.only_mid_control)
        else:  # control is not None
            control = self.control_model(x=z_cat, hint=torch.cat(cond['c_concat'], 1),
                                         timesteps=t, context=cond_txt)  # control_stage_config: ControlNet
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=z_cat, timesteps=t, context=cond_txt, control=control,
                                  only_mid_control=self.only_mid_control)

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        c_unconditional_text_image_dict = {
            "text": [""] * N,
            "image": torch.zeros(N, 3, 224, 224).to(self.device),
        }
        return self.get_learned_conditioning(c_unconditional_text_image_dict)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        """ Called by ImageLogger.log_img """
        use_ddim = ddim_steps is not None
        unconditional_guidance_scale = 2.0
        sample = True

        log = dict()
        z_dict, c_dict = self.get_input(batch, self.first_stage_key, bs=N)
        z = z_dict["z_x"][:N]
        z_ag = z_dict["z_ag"][:N]
        z_ag_m = z_dict["z_ag_m"][:N]
        c_cat = [x[:N] for x in c_dict["c_concat"]]
        c_cloth = c_dict["c_crossattn"][0]["cloth"][:N].to(self.device)
        c_cloth = rearrange(c_cloth, 'b h w c -> b c h w')
        c_cloth = c_cloth.to(memory_format=torch.contiguous_format).float()
        c_text_image_dict = {
            "text": c_dict["c_crossattn"][0]["prompt"][:N],
            "image": c_cloth,
        }
        c = self.get_learned_conditioning(c_text_image_dict)

        # ag = batch[self.agnostic_key]  # "agnostic"
        # if ag is not None:
        #     ag = ag[:N]
        # ag = ag.to(self.device)
        # ag = rearrange(ag, 'b h w c -> b c h w')
        # ag = ag.to(memory_format=torch.contiguous_format).float()

        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["recon_agnostic"] = self.decode_first_stage(z_ag)
        log["cloth"] = c_cloth
        log["recon_z"] = self.decode_first_stage(z)  # reconstruction by VAE Auto-encoder,
        log["control"] = c_cat[0] * 2.0 - 1.0  # the 1st element
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key]["prompt"], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            t = torch.ones((z.shape[0],), device=self.device).long() * (self.num_timesteps - 1)
            z_x_noisy = self.q_sample(x_start=z, t=t, noise=torch.randn_like(z))
            samples, z_denoise_row = self.sample_log(cond={"c_concat": c_cat, "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta,
                                                     x_T={"z_x": z_x_noisy, "z_ag": z_ag,
                                                          "z_ag_m": z_ag_m}
                                                     )
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row["pred_x0"])
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": uc_cat, "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": c_cat, "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             x_T={"z_x": None, "z_ag": z_dict["z_ag"],
                                                  "z_ag_m": z_dict["z_ag_m"]}
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self, tryon_dataset=self.val_dataset)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    @staticmethod
    def _add_parameters(named_parameters, prefix: str, to_params: list, to_names: list,
                        specific_names: tuple = None,
                        ):
        for name, param in named_parameters:
            if specific_names is not None and name not in specific_names:
                continue
            if name in to_names:
                print(f"[TryOnCLDM][Warning] duplicate parameter name {name}, skipping.")
                continue
            to_params.append(param)
            full_name = f"{prefix}.{name}" if prefix is not None else name
            to_names.append(full_name)

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list()
        names = list()
        # TODO: close control_model optimization
        self._add_parameters(self.control_model.named_parameters(),
                             "control_model", params, names)

        # TODO: unfreeze which parts of the UNet
        self._add_parameters(self.model.diffusion_model.input_blocks.named_parameters(),
                             "model.diffusion_model.input_blocks", params, names)
        self._add_parameters(self.model.diffusion_model.middle_block.named_parameters(),
                             "model.diffusion_model.middle_block", params, names)
        if not self.sd_locked:
            self._add_parameters(self.model.diffusion_model.output_blocks.named_parameters(),
                                 "model.diffusion_model.output_blocks", params, names)
            self._add_parameters(self.model.diffusion_model.out.named_parameters(),
                                 "model.diffusion_model.out", params, names)

        # TODO: unfreeze cond_stage_model
        self._add_parameters(self.cond_stage_model.named_trainable_params_list(),
                             "cond_stage_model", params, names)

        # TODO: input_0 should be optimized?
        additional_param_names = (
            "model.diffusion_model.input_blocks.0.0.weight",
            "model.diffusion_model.input_blocks.0.0.bias",
            "control_model.input_blocks.0.0.weight",
            "control_model.input_blocks.0.0.bias",
        )
        self._add_parameters(self.named_parameters(),
                             "", params, names, specific_names=additional_param_names)

        # print("[TryOnCLDM] optimized params are below:")
        # for name in names:
        #     print(f"optimized---{name}")
        print("[TryOnCLDM] optimized params cnt = ", len(params))

        # print("[TryOnCLDM] requires_grad params are below:")
        grad_names, grad_params = [], []
        for name, param in self.named_parameters():
            if param.requires_grad:
                grad_names.append(name)
        print("[TryOnCLDM] requires_grad params cnt = ", len(grad_names))

        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
