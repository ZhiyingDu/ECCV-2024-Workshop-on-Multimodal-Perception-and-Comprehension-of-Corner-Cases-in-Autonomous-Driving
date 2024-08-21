# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from diffusers.configuration_utils import register_to_config
from diffusers.models.motion_module import VersatileAttention
from ..misc.common import _get_module, _set_module, load_module
from .blocks import (
    BasicMultiviewTransformerBlock,
    TemporalMultiviewTransformerBlock,
)
from .unet_2d_condition_multiview import (
    UNet2DConditionModelMultiview,
    UNet2DConditionOutput,
)
from torch.cuda.amp import autocast

class UNet2DConditionModelMultiviewT(UNet2DConditionModelMultiview):
    r"""
    UNet2DConditionModel is a conditional 2D UNet model that takes in a noisy sample, conditional state, and a timestep
    and returns sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the models (such as downloading or saving, etc.)

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample.
        in_channels (`int`, *optional*, defaults to 4): The number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4): The number of channels in the output.
        center_input_sample (`bool`, *optional*, defaults to `False`): Whether to center the input sample.
        flip_sin_to_cos (`bool`, *optional*, defaults to `False`):
            Whether to flip the sin to cos in the time embedding.
        freq_shift (`int`, *optional*, defaults to 0): The frequency shift to apply to the time embedding.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")`):
            The tuple of downsample blocks to use.
        mid_block_type (`str`, *optional*, defaults to `"UNetMidBlock2DCrossAttn"`):
            The mid block type. Choose from `UNetMidBlock2DCrossAttn` or `UNetMidBlock2DSimpleCrossAttn`, will skip the
            mid block layer if `None`.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",)`):
            The tuple of upsample blocks to use.
        only_cross_attention(`bool` or `Tuple[bool]`, *optional*, default to `False`):
            Whether to include self-attention in the basic transformer blocks, see
            [`~models.attention.BasicTransformerBlock`].
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2): The number of layers per block.
        downsample_padding (`int`, *optional*, defaults to 1): The padding to use for the downsampling convolution.
        mid_block_scale_factor (`float`, *optional*, defaults to 1.0): The scale factor to use for the mid block.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        norm_num_groups (`int`, *optional*, defaults to 32): The number of groups to use for the normalization.
            If `None`, it will skip the normalization and activation layers in post-processing
        norm_eps (`float`, *optional*, defaults to 1e-5): The epsilon to use for the normalization.
        cross_attention_dim (`int` or `Tuple[int]`, *optional*, defaults to 1280):
            The dimension of the cross attention features.
        encoder_hid_dim (`int`, *optional*, defaults to None):
            If given, `encoder_hidden_states` will be projected from this dimension to `cross_attention_dim`.
        attention_head_dim (`int`, *optional*, defaults to 8): The dimension of the attention heads.
        resnet_time_scale_shift (`str`, *optional*, defaults to `"default"`): Time scale shift config
            for resnet blocks, see [`~models.resnet.ResnetBlock2D`]. Choose from `default` or `scale_shift`.
        class_embed_type (`str`, *optional*, defaults to None):
            The type of class embedding to use which is ultimately summed with the time embeddings. Choose from `None`,
            `"timestep"`, `"identity"`, `"projection"`, or `"simple_projection"`.
        addition_embed_type (`str`, *optional*, defaults to None):
            Configures an optional embedding which will be summed with the time embeddings. Choose from `None` or
            "text". "text" will use the `TextTimeEmbedding` layer.
        num_class_embeds (`int`, *optional*, defaults to None):
            Input dimension of the learnable embedding matrix to be projected to `time_embed_dim`, when performing
            class conditioning with `class_embed_type` equal to `None`.
        time_embedding_type (`str`, *optional*, default to `positional`):
            The type of position embedding to use for timesteps. Choose from `positional` or `fourier`.
        time_embedding_dim (`int`, *optional*, default to `None`):
            An optional override for the dimension of the projected time embedding.
        time_embedding_act_fn (`str`, *optional*, default to `None`):
            Optional activation function to use on the time embeddings only one time before they as passed to the rest
            of the unet. Choose from `silu`, `mish`, `gelu`, and `swish`.
        timestep_post_act (`str, *optional*, default to `None`):
            The second activation function to use in timestep embedding. Choose from `silu`, `mish` and `gelu`.
        time_cond_proj_dim (`int`, *optional*, default to `None`):
            The dimension of `cond_proj` layer in timestep embedding.
        conv_in_kernel (`int`, *optional*, default to `3`): The kernel size of `conv_in` layer.
        conv_out_kernel (`int`, *optional*, default to `3`): The kernel size of `conv_out` layer.
        projection_class_embeddings_input_dim (`int`, *optional*): The dimension of the `class_labels` input when
            using the "projection" `class_embed_type`. Required when using the "projection" `class_embed_type`.
        class_embeddings_concat (`bool`, *optional*, defaults to `False`): Whether to concatenate the time
            embeddings with the class embeddings.
        mid_block_only_cross_attention (`bool`, *optional*, defaults to `None`):
            Whether to use cross attention with the mid block when using the `UNetMidBlock2DSimpleCrossAttn`. If
            `only_cross_attention` is given as a single boolean and `mid_block_only_cross_attention` is None, the
            `only_cross_attention` value will be used as the value for `mid_block_only_cross_attention`. Else, it will
            default to `False`.
    """

    _supports_gradient_checkpointing = True
    _WARN_ONCE = 0

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: Union[int, Tuple[int]] = 1280,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        resnet_skip_time_act: bool = False,
        resnet_out_scale_factor: int = 1.0,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        time_embedding_act_fn: Optional[str] = None,
        timestep_post_act: Optional[str] = None,
        time_cond_proj_dim: Optional[int] = None,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        projection_class_embeddings_input_dim: Optional[int] = None,
        class_embeddings_concat: bool = False,
        mid_block_only_cross_attention: Optional[bool] = None,
        cross_attention_norm: Optional[str] = None,
        addition_embed_type_num_heads=64,
        # parameter added, we should keep all above (do not use kwargs)
        trainable_state="only_new",
        neighboring_view_pair: Optional[dict] = None,
        neighboring_attn_type: str = "add",
        zero_module_type: str = "zero_linear",
        crossview_attn_type: str = "basic",
        epipolar_mask_type: str = "binary",
        img_size: Optional[Tuple[int, int]] = None,
        # for temporal
        video_length=16,
        zero_module_type2: str = "zero_linear",
        temp_attn_type='t_last',
        temp_pos_emb='learnable',
        spatial_trainable=False,

        # Additional
        use_motion_module              = True,
        motion_module_resolutions      = ( 1,2,4,8 ),
        motion_module_mid_block        = True,
        motion_module_decoder_only     = False,
        motion_module_type             = "Vanilla",
        motion_module_kwargs           = {'num_attention_heads': 8, 'num_transformer_block': 1, 'attention_block_types':[ "Temporal_Self", "Temporal_Self" ], 'temporal_position_encoding': True, 'temporal_attention_dim_div': 1, 'zero_initialize':True},
        unet_use_cross_frame_attention = False,
        unet_use_temporal_attention    = False,

    ):
        super().__init__(
            sample_size=sample_size, in_channels=in_channels,
            out_channels=out_channels, center_input_sample=center_input_sample,
            flip_sin_to_cos=flip_sin_to_cos, freq_shift=freq_shift,
            down_block_types=down_block_types, mid_block_type=mid_block_type,
            up_block_types=up_block_types,
            only_cross_attention=only_cross_attention,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            downsample_padding=downsample_padding,
            mid_block_scale_factor=mid_block_scale_factor, act_fn=act_fn,
            norm_num_groups=norm_num_groups, norm_eps=norm_eps,
            cross_attention_dim=cross_attention_dim,
            encoder_hid_dim=encoder_hid_dim,
            encoder_hid_dim_type=encoder_hid_dim_type,
            attention_head_dim=attention_head_dim,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            class_embed_type=class_embed_type,
            addition_embed_type=addition_embed_type,
            num_class_embeds=num_class_embeds,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resnet_skip_time_act=resnet_skip_time_act,
            resnet_out_scale_factor=resnet_out_scale_factor,
            time_embedding_type=time_embedding_type,
            time_embedding_dim=time_embedding_dim,
            time_embedding_act_fn=time_embedding_act_fn,
            timestep_post_act=timestep_post_act,
            time_cond_proj_dim=time_cond_proj_dim,
            conv_in_kernel=conv_in_kernel, conv_out_kernel=conv_out_kernel,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            class_embeddings_concat=class_embeddings_concat,
            mid_block_only_cross_attention=mid_block_only_cross_attention,
            cross_attention_norm=cross_attention_norm,
            addition_embed_type_num_heads=addition_embed_type_num_heads,
            trainable_state=trainable_state,
            neighboring_view_pair=neighboring_view_pair,
            neighboring_attn_type=neighboring_attn_type,
            zero_module_type=zero_module_type,
            crossview_attn_type=crossview_attn_type,
            epipolar_mask_type=epipolar_mask_type, img_size=img_size,
            # Additional
            video_length                   = video_length,
            use_motion_module              = use_motion_module,
            motion_module_resolutions      = motion_module_resolutions,
            motion_module_mid_block        = motion_module_mid_block,
            motion_module_decoder_only     = motion_module_decoder_only,
            motion_module_type             = motion_module_type,
            motion_module_kwargs           = motion_module_kwargs,
            unet_use_cross_frame_attention = unet_use_cross_frame_attention,
            unet_use_temporal_attention    = unet_use_temporal_attention,)
        
        for name, mod in list(self.named_modules()):
            assert crossview_attn_type == "basic"
            if isinstance(mod, BasicMultiviewTransformerBlock):
                _set_module(self, name, TemporalMultiviewTransformerBlock(
                    **mod._args,
                    zero_module_type2=zero_module_type2,
                    video_length=video_length,
                    type=temp_attn_type,
                    pos_emb=temp_pos_emb,
                    spatial_trainable=spatial_trainable,
                ))
        self.trainable_state = trainable_state

    @autocast(True)
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # TODO: actually, we do not change logic in forward

        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        # print(sample.shape)
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        # # force HalfTensor transfer to FloatTensor
        # sample = sample.float()

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            if self._WARN_ONCE == 0:
                logging.warning(
                    f"[{self.__class__.__name__}] Forward upsample size to force interpolation output size.")
                self._WARN_ONCE = 1
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor(
                [timesteps],
                dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError(
                    "class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # there might be better ways to encapsulate this.
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)

            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        if self.config.addition_embed_type == "text":
            aug_emb = self.add_embedding(encoder_hidden_states)
            emb = emb + aug_emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        if self.encoder_hid_proj is not None:
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)

        # 2. pre-process
        # print(sample.shape)
        sample = self.conv_in(sample)
        # print(sample.shape)
        # assert 0 == 1
        # 3. down
        down_block_res_samples = (sample,)
        if self.is_gradient_checkpointing:
            if sample.is_leaf:
                sample.requires_grad = True
            elif sample.requires_grad == False:
                logging.warn(
                    f"[{self.__class__.__name__}] Please check you model, it may not enable gradient.")
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=copy.deepcopy(cross_attention_kwargs),
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples += (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=copy.deepcopy(cross_attention_kwargs),
            )

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(
                upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample, temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=copy.deepcopy(cross_attention_kwargs),
                    upsample_size=upsample_size, attention_mask=attention_mask,)
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size)

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)
    
    @classmethod
    def from_pretrained_2d(cls, pretrained_model_path, subfolder=None, unet_additional_kwargs=None):
        import os
        import json
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        print(f"loaded 3D unet's pretrained weights from {pretrained_model_path} ...")

        config_file = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)

        from diffusers.utils import WEIGHTS_NAME
        model = cls.from_config(config)
        # model_file = os.path.join("/share/ckpt/duzhiying/animatediff/outputs/training_256-2024-04-11T23-42-19/checkpoints/checkpoint-epoch-4.ckpt")
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)

        if not os.path.isfile(model_file):
            raise RuntimeError(f"{model_file} does not exist")

        state_dict = torch.load(model_file, map_location="cpu")

        new_state_dict = {}

        for key, value in state_dict.items():
            if 'lora' in key:
                # Skip keys that contain 'lora'
                continue
            if 'linear.' in key:
                # Replace 'linear.' with ''
                new_key = key.replace('linear.', '')
            else:
                # Keep the original key if 'linear.' is not found
                new_key = key
            
            # Add the modified key and value to the new state_dict
            new_state_dict[new_key] = value
        # param_keys = state_dict.keys()
        # output_file_path = 'parameter_keys_weight.txt'
        # # 将键写入文本文件
        # with open(output_file_path, 'w') as file:
        #     for key in param_keys:
        #         file.write(key + '\n')
        
        # state_dict = model.state_dict()
        # # 提取 state_dict 中的键
        # param_keys = state_dict.keys()

        # # 指定输出文件路径
        # output_file_path = 'parameter_keys_model.txt'

        # # 将键写入文本文件
        # with open(output_file_path, 'w') as file:
        #     for key in param_keys:
        #         file.write(key + '\n')
        # assert 0 == 1
        m, u = model.load_state_dict(new_state_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        if len(m) != 0:
            print(m)
        # params = [p.numel() if "motion_modules."  in n and "adapter" not in n else 0 for n, p in model.named_parameters()]
        # print(f"### Stable Diffusion Module Parameters: {sum(params) / 1e6} M")

        return model