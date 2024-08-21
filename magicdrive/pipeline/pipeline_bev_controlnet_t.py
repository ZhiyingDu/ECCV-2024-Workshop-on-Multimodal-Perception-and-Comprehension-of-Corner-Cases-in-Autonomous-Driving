
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging

import torch
import numpy as np
from einops import rearrange, repeat
import math

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from ..misc.common import move_to
from .pipeline_bev_controlnet import (
    StableDiffusionBEVControlNetPipeline,
    BEVStableDiffusionPipelineOutput,
)

TARGET_ORDER = {
    "right": [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_RIGHT",
        "CAM_BACK",
    ],
    "left": [
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT",
    ],
}

def ordered_halving(val):
    bin_str = f"{val:064b}"
    bin_flip = bin_str[::-1]
    as_int = int(bin_flip, 2)

    return as_int / (1 << 64)

def uniform(
    step: int = ...,
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    if num_frames <= context_size:
        yield list(range(num_frames))
        return

    context_stride = min(context_stride, int(np.ceil(np.log2(num_frames / context_size))) + 1)

    for context_step in 1 << np.arange(context_stride):
        pad = int(round(num_frames * ordered_halving(step)))
        for j in range(
            int(ordered_halving(step) * context_step) + pad,
            num_frames + pad + (0 if closed_loop else -context_overlap),
            (context_size * context_step - context_overlap),
        ):
            yield [e % num_frames for e in range(j, j + context_size * context_step, context_step)]


def get_context_scheduler(name: str) -> Callable:
    if name == "uniform":
        return uniform
    else:
        raise ValueError(f"Unknown context_overlap policy {name}")
    



def _concat_side(latents, orders, view_order):
    _lats = []
    for cami, cam in enumerate(orders):
        idx = view_order.index(cam)
        latent = latents[idx]
        c, h, w = latent.shape
        if cami == 0:
            latent = latents[idx][..., w // 2:]  # first frame take second half
        elif cami == len(orders) - 1:
            latent = latents[idx][..., :w // 2]  # last frame take first half
        else:
            pass
        _lats.append(latent)
    return torch.cat(_lats, dim=-1)


def _split_side(latent, orders, width):
    first = latent[..., :width // 2]
    last = latent[..., -width // 2:]
    others = latent[..., width // 2:-width // 2].chunk(len(orders) - 2, dim=-1)
    return [first] + list(others) + [last]


def roll_latents(latents, view_order, roll_length=None):
    _latents = latents.clone()
    _, _, h, w = latents.shape  # n_cam, c, h, w
    if roll_length is None:
        roll_length = w // 2
    rights = _concat_side(_latents, TARGET_ORDER["right"], view_order)
    lefts = _concat_side(_latents, TARGET_ORDER["left"], view_order)
    # start roll
    rights = rights.roll(roll_length, dims=[-1])  # c, h, w * n
    lefts = lefts.roll(-roll_length, dims=[-1])  # c, h, w * n
    rights = _split_side(rights, TARGET_ORDER["right"], w)
    lefts = _split_side(lefts, TARGET_ORDER["left"], w)
    new_latents = [None for _ in view_order]
    new_latents[view_order.index("CAM_FRONT_RIGHT")] = rights[1]
    new_latents[view_order.index("CAM_BACK_RIGHT")] = rights[2]
    new_latents[view_order.index("CAM_BACK_LEFT")] = lefts[1]
    new_latents[view_order.index("CAM_FRONT_LEFT")] = lefts[2]
    new_latents[view_order.index("CAM_FRONT")] = torch.cat(
        [lefts[-1], rights[0]], dim=-1,
    )
    new_latents[view_order.index("CAM_BACK")] = torch.cat(
        [rights[-1], lefts[0]], dim=-1,
    )
    new_latents = torch.stack(new_latents, dim=0)  # N_cam, c, h, w
    return new_latents

def select_controlnet_res_samples(controlnet_res_samples_cache_dict, context):
        _down_block_res_samples = []
        _mid_block_res_sample = []
        _encoder_hidden_states_with_cam = []
        for i in np.concatenate(np.array(context)):
            _down_block_res_samples.append(controlnet_res_samples_cache_dict[i][0])
            _mid_block_res_sample.append(controlnet_res_samples_cache_dict[i][1])
            _encoder_hidden_states_with_cam.append(controlnet_res_samples_cache_dict[i][2])
        down_block_res_samples = [[] for _ in range(len(controlnet_res_samples_cache_dict[i][0]))]
        for res_t in _down_block_res_samples:
            for i, res in enumerate(res_t):
                down_block_res_samples[i].append(res)
        down_block_res_samples = [torch.cat(res) for res in down_block_res_samples]
        mid_block_res_sample = torch.cat(_mid_block_res_sample)
        encoder_hidden_states_with_cam = torch.cat(_encoder_hidden_states_with_cam)

        return down_block_res_samples, mid_block_res_sample, encoder_hidden_states_with_cam

class StableDiffusionBEVControlNetTPipeline(
        StableDiffusionBEVControlNetPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        unet: UNet2DConditionModel,
        controlnet,
        scheduler: KarrasDiffusionSchedulers,
        tokenizer: CLIPTokenizer,
        safety_checker: StableDiffusionSafetyChecker = None,
        feature_extractor: CLIPImageProcessor = None,
        requires_safety_checker: bool = False,
    ):
        super().__init__(
            vae,
            text_encoder,
            unet,
            controlnet,
            scheduler,
            tokenizer,
            safety_checker,
            feature_extractor,
            requires_safety_checker,
        )

    def decode_latents(self, latents, decode_bs=None):
        if decode_bs is not None:
            num_batch = latents.shape[0] // decode_bs
            latents = latents.chunk(num_batch)
            results = []
            for _latents in latents:
                results.append(super().decode_latents(_latents))
            return np.concatenate(results, axis=0)
        else:
            return super().decode_latents(latents)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: torch.FloatTensor,
        camera_param: Union[torch.Tensor, None],
        height: int,
        width: int,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: float = 1,
        guess_mode: bool = False,
        use_zero_map_as_unconditional: bool = False,
        bev_controlnet_kwargs={},
        bbox_max_length=None,
        # for temporal
        init_noise="rand",
        view_order=None,
        double_cfg_inference=False,
        decode_bs=None,
        keyframe_rate=None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image (`torch.FloatTensor`, `PIL.Image.Image`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`,
                    `List[List[torch.FloatTensor]]`, or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition. ControlNet uses this input condition to generate guidance to Unet. If
                the type is specified as `Torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can
                also be accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If
                height and/or width are passed, `image` is resized according to them. If multiple ControlNets are
                specified in init, images must be passed as a list such that each element of the list can be correctly
                batched for input to a single controlnet.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the controlnet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original unet. If multiple ControlNets are specified in init, you can set the
                corresponding scale as a list.
            guess_mode (`bool`, *optional*, defaults to `False`):
                In this mode, the ControlNet encoder will try best to recognize the content of the input image even if
                you remove all prompts. The `guidance_scale` between 3.0 and 5.0 is recommended.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        # BEV: we cannot use the size of image
        # height, width = self._default_height_width(height, width, None)

        # 1. Check inputs. Raise error if not correct
        # we do not need this, only some type assertion
        # self.check_inputs(
        #     prompt,
        #     image,
        #     height,
        #     width,
        #     callback_steps,
        #     negative_prompt,
        #     prompt_embeds,
        #     negative_prompt_embeds,
        #     controlnet_conditioning_scale,
        # )

        # 2. Define call parameters
        # NOTE: we get batch_size first from prompt, then align with it.
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        ### BEV, check camera_param ###
        if camera_param is None:
            # use uncond_cam and disable classifier free guidance
            N_cam = 6  # TODO: hard-coded
            camera_param = self.controlnet.uncond_cam_param((batch_size, N_cam))
            do_classifier_free_guidance = False
        N_cam = camera_param.shape[1]
        ### done ###

        # if isinstance(self.controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
        #     controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(self.controlnet.nets)

        # 3. Encode input prompt
        # NOTE: here they use padding to 77, is this necessary?
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )  # (2 * video_length, 77 + 1, 768)   uncond + cond, 77, 768
        # print("prompt_embeds.shape:", prompt_embeds.shape)
        # 4. Prepare image
        # NOTE: if image is not tensor, there will be several process.
        assert not self.control_image_processor.config.do_normalize, "Your controlnet should not normalize the control image."
        image = self.prepare_image(
            image=image,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=self.controlnet.dtype,
            do_classifier_free_guidance=do_classifier_free_guidance,
            guess_mode=guess_mode,
        )  # (2 * b, c_26, 200, 200)
        if use_zero_map_as_unconditional and do_classifier_free_guidance:
            # uncond in the front, cond in the tail
            _images = list(torch.chunk(image, 2))
            _images[0] = torch.zeros_like(_images[0])
            image = torch.cat(_images)

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        if init_noise == "rand" or init_noise == "rand_all":
            _bs = batch_size * num_images_per_prompt
        elif init_noise == "same":
            _bs = 1
            real_bs = batch_size * num_images_per_prompt
        elif init_noise.startswith("cycle"):
            assert num_images_per_prompt == 1
            _bs = N_cam  # we generate different noise to each cam
            real_bs = batch_size * num_images_per_prompt
        else:
            raise NotImplementedError(f'Unknown init_noise type {init_noise}.')
        latents = self.prepare_latents(
            _bs,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,  # will use if not None, otherwise will generate
        )  # (b, c, h/8, w/8) -> (bs, 4, 28, 50)
        if init_noise == "same":
            latents = repeat(latents, '1 ... -> b ...', b=real_bs)
        elif init_noise.startswith("cycle"):
            if "_" in init_noise:
                roll_length = int(init_noise.split("_")[1])
            else:
                roll_length = None
            _latents = [latents]
            _latents_i = latents
            for i in range(1, real_bs):
                _latents_i = roll_latents(
                    _latents_i, view_order, roll_length=roll_length)
                if init_noise.startswith("cyclemix"):
                    if i % keyframe_rate != 0:  # not keyframe
                        _latents.append(torch.randn(
                            _latents_i.shape,
                            generator=generator, dtype=_latents_i.dtype,
                            device=_latents_i.device
                        ))
                    else:  # keyframe, we use the rolled latents
                        logging.debug(f"we set frame {i} will rolled version.")
                        _latents.append(_latents_i)
                else:
                    _latents.append(_latents_i)
            latents = torch.stack(_latents, dim=0)
        else:
            pass

        # 7. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        ###### BEV: here we reconstruct each input format ######
        assert camera_param.shape[0] == batch_size, \
            f"Except {batch_size} camera params, but you have bs={len(camera_param)}"
        if init_noise == "rand_all":
            if generator is None:
                latents = torch.stack(
                    [latents] +
                    [torch.randn_like(latents) for _ in range(N_cam - 1)],
                    dim=1)  # bs, 6, 4, 28, 50
            else:
                latents = torch.stack(
                    [latents] +
                    [torch.randn(
                        latents.shape,
                        generator=generator, dtype=latents.dtype,
                        device=latents.device,
                    )
                        for _ in range(N_cam - 1)],
                    dim=1)  # bs, 6, 4, 28, 50
        elif not init_noise.startswith("cycle"):
            latents = torch.stack([latents] * N_cam, dim=1)  # bs, 6, 4, 28, 50
        # prompt_embeds, no need for b, len, 768
        # image, no need for b, c, 200, 200
        camera_param = camera_param.to(self.device)
        if do_classifier_free_guidance and not guess_mode:
            # uncond in the front, cond in the tail
            _images = list(torch.chunk(image, 2))
            kwargs_with_uncond = self.controlnet.add_uncond_to_kwargs(
                camera_param=camera_param,
                image=_images[0],  # 0 is for unconditional
                max_len=bbox_max_length,
                **bev_controlnet_kwargs,
            )
            kwargs_with_uncond.pop("max_len", None)  # some do not take this.
            camera_param = kwargs_with_uncond.pop("camera_param")
            _images[0] = kwargs_with_uncond.pop("image")
            image = torch.cat(_images)
            bev_controlnet_kwargs = move_to(kwargs_with_uncond, self.device)
        ###### BEV end ######

        # 8. Denoising loop
        num_warmup_steps = len(
            timesteps) - num_inference_steps * self.scheduler.order
        
        context_scheduler = get_context_scheduler("uniform")
        context_frames = 16
        context_stride = 1
        context_batch_size = 1
        context_overlap = 4
        controlnet_res_samples_cache_dict = []
        # print("total_frame:",len(controlnet_res_samples_cache_dict))
        # print("double_cfg_inference:", double_cfg_inference)
        # assert 0 == 1
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for l, t in enumerate(timesteps):
                controlnet_res_samples_cache_dict = []
                # print(latents.shape)
                noise_pred_total = torch.zeros(
                    (latents.shape[0] * (2 if do_classifier_free_guidance else 1), *latents.shape[1:]),
                    device=latents.device,
                    dtype=latents.dtype,
                )
                counter = torch.zeros(
                    (latents.shape[0] * (2 if do_classifier_free_guidance else 1), 1, 1, 1, 1), device=latents.device, dtype=latents.dtype
                )

                context_queue = list(context_scheduler(
                    0, num_inference_steps, latents.shape[0], context_frames, context_stride, context_overlap
                ))
                num_context_batches = math.ceil(len(context_queue) / context_batch_size)
                # assert 0 == 1
                # expand the latents if we are doing classifier free guidance
                # bs*2, 6, 4, 28, 50
                for i in range(num_context_batches):
                    context = context_queue[i*context_batch_size: (i+1)*context_batch_size]
                    # print(context)
                    latent_model_input = (
                        torch.cat([latents[c] for c in context])
                        .to(device)
                    )
                    latent_model_input = (
                        torch.cat([latent_model_input] * 2)
                        if do_classifier_free_guidance else latent_model_input)
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )
                    # print("latent_model_input.shape:", latent_model_input.shape)
                    # controlnet(s) inference
                    controlnet_t = t.unsqueeze(0)
                    # guess_mode & classifier_free_guidance -> only guidance use controlnet
                    # not guess_mode & classifier_free_guidance -> all use controlnet
                    # guess_mode -> normal input, take effect in controlnet
                    if guess_mode and do_classifier_free_guidance:
                        # Infer ControlNet only for the conditional batch.
                        controlnet_latent_model_input = latent_model_input
                        controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                    else:
                        controlnet_latent_model_input = latent_model_input
                        controlnet_prompt_embeds = prompt_embeds
                    controlnet_t = controlnet_t.repeat(
                        len(controlnet_latent_model_input))

                    # print("controlnet_t.shape:", controlnet_t.shape)
                    # print("controlnet_latent_model_input.shape:", controlnet_latent_model_input.shape)
                    # # fmt: off
                    # print("camera_param.shape:", camera_param.shape)
                    # print("controlnet_prompt_embeds,shape:", controlnet_prompt_embeds.shape)
                    # print("image.shape:", image.shape)
                    # assert 0 == 1
                    # 生成新的一部分数据
                    new_part = [x + _bs for x in context[0]]
                    new_context = [context[0] + new_part]
                    # print(new_context)

                    down_block_res_samples, mid_block_res_sample, \
                    encoder_hidden_states_with_cam = self.controlnet(
                        controlnet_latent_model_input,
                        controlnet_t,
                        torch.cat([camera_param[c] for c in new_context]),  # for BEV
                        encoder_hidden_states=torch.cat([controlnet_prompt_embeds[c] for c in new_context]),
                        controlnet_cond=torch.cat([image[c] for c in new_context]),
                        context = new_context,
                        conditioning_scale=controlnet_conditioning_scale,
                        guess_mode=guess_mode,
                        return_dict=False,
                        **bev_controlnet_kwargs, # for BEV
                    )
                    # fmt: on

                    # print(len(down_block_res_samples))
                    # print(down_block_res_samples[0].shape)
                    # print(mid_block_res_sample.shape)
                    # print(encoder_hidden_states_with_cam.shape)
                    # assert 0 == 1
                    # 96 * 

                    controlnet_res_samples_cache_dict.append([down_block_res_samples, mid_block_res_sample, encoder_hidden_states_with_cam])
                    # for j, k in enumerate(np.concatenate(np.array(context))):
                    #     controlnet_res_samples_cache_dict[k] = ([sample[j:j+1] for sample in down_block_res_samples], mid_block_res_sample[j:j+1], encoder_hidden_states_with_cam[j:j+1])
                

                context_queue = list(context_scheduler(
                    0, num_inference_steps, latents.shape[0], context_frames, context_stride, context_overlap
                ))
                num_context_batches = math.ceil(len(context_queue) / context_batch_size)
                for i in range(num_context_batches):
                    context = context_queue[i*context_batch_size: (i+1)*context_batch_size]
                    # print(context)
                    latent_model_input = (
                        torch.cat([latents[c] for c in context])
                        .to(device)
                    )
                    latent_model_input = (
                        torch.cat([latent_model_input] * 2)
                        if do_classifier_free_guidance else latent_model_input)
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    # =============================================================
                    # Strating from here, we use 4-dim data.
                    # encoder_hidden_states_with_cam: (2b x N), 78, 768
                    # latent_model_input: 2b, N, 4, 28, 50 -> 2b x N, 4, 28, 50
                    # print("latent_model_input.shape:", latent_model_input.shape)
                    latent_model_input = rearrange(
                        latent_model_input, 'b n ... -> (b n) ...')
                    # print("latent_model_input.shape:", latent_model_input.shape)

                    down_block_res_samples = controlnet_res_samples_cache_dict[i][0]
                    mid_block_res_sample   = controlnet_res_samples_cache_dict[i][1]
                    encoder_hidden_states_with_cam = controlnet_res_samples_cache_dict[i][2]

                    # print("encoder_hidden_states_with_cam.shape:",encoder_hidden_states_with_cam.shape)

                    if guess_mode and do_classifier_free_guidance:
                        down_block_res_samples = [
                            torch.cat([torch.zeros_like(d), d])
                            for d in down_block_res_samples
                        ]
                        mid_block_res_sample = torch.cat(
                            [torch.zeros_like(mid_block_res_sample), mid_block_res_sample]
                        )
                        # add uncond encoder_hidden_states_with_cam here
                        # print("1111111111111111111111111111111111111111")
                        # print("encoder_hidden_states_with_cam.shape:",encoder_hidden_states_with_cam.shape)
                        encoder_hidden_states_with_cam = self.controlnet.add_uncond_to_emb(
                            prompt_embeds.chunk(2)[0], N_cam,
                            encoder_hidden_states_with_cam,
                        )
                    
                    # predict the noise residual: 2bxN, 4, 28, 50
                    if double_cfg_inference and do_classifier_free_guidance:
                        assert cross_attention_kwargs == None
                        latent_model_input = latent_model_input.chunk(2)
                        encoder_hidden_states_with_cam = encoder_hidden_states_with_cam.chunk(
                            2)
                        mid_block_res_sample = mid_block_res_sample.chunk(2)
                        _down_block_res_samples = [None, None]
                        _down_block_res_samples[0] = [
                            _downs.chunk(2)[0] for _downs in
                            down_block_res_samples]
                        _down_block_res_samples[1] = [
                            _downs.chunk(2)[1] for _downs in
                            down_block_res_samples]
                        # print("latent_model_input[0].shape:",latent_model_input[0].shape)
                        # print("encoder_hidden_states_with_cam[0].shape:",encoder_hidden_states_with_cam[0].shape)
                        # print("_down_block_res_samples[0].shape:",_down_block_res_samples[0][0].shape)
                        # print("mid_block_res_sample[0].shape:",mid_block_res_sample[0].shape)
                        # assert 0 == 1
                        unet_out_uncond = self.unet(
                            latent_model_input[0], t,
                            encoder_hidden_states=encoder_hidden_states_with_cam[0],
                            # if use original unet, it cannot take kwargs
                            cross_attention_kwargs=cross_attention_kwargs,
                            down_block_additional_residuals=_down_block_res_samples[0],
                            mid_block_additional_residual=mid_block_res_sample[0],
                        ).sample
                        unet_out_cond = self.unet(
                            latent_model_input[1], t,
                            encoder_hidden_states=encoder_hidden_states_with_cam[1],
                            # if use original unet, it cannot take kwargs
                            cross_attention_kwargs=cross_attention_kwargs,
                            down_block_additional_residuals=_down_block_res_samples[1],
                            mid_block_additional_residual=mid_block_res_sample[1],
                        ).sample
                        noise_pred = torch.cat([unet_out_uncond, unet_out_cond])
                        del unet_out_uncond, unet_out_cond
                    else:
                        noise_pred = self.unet(
                            latent_model_input,  # may with unconditional
                            t,
                            encoder_hidden_states=encoder_hidden_states_with_cam,
                            cross_attention_kwargs=cross_attention_kwargs,
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample,
                        ).sample
                    # print("noise_pred_shape:", noise_pred.shape)
                    pred_uc, pred_c = noise_pred.chunk(2)
                    # pred = torch.cat([pred_uc.unsqueeze(0), pred_c.unsqueeze(0)])
                    # print(pred.shape)
                    pred_uc = rearrange(pred_uc, '(b n) ... -> b n ...', n=N_cam)
                    pred_c = rearrange(pred_c, '(b n) ... -> b n ...', n=N_cam)
                    # print("pred_c.shape:", pred_c.shape)
                    # print("noise_pred_total.shape:", noise_pred_total.shape)
                    # print(type(_bs))
                    # print(_bs)
                    # assert 0 == 1
                    for j, c in enumerate(context[0]):
                        # print(type(noise_pred_total[c]), type(pred_uc[j]))   <class 'torch.Tensor'> <class 'torch.Tensor'>
                        noise_pred_total[c] = noise_pred_total[c] + pred_uc[j]
                        # print(type(noise_pred_total[c + _bs]), type(pred_c[j]))
                        noise_pred_total[c + _bs] = noise_pred_total[c + _bs] + pred_c[j]
                        counter[c] = counter[c] + 1
                        counter[c + _bs] = counter[c + _bs] + 1
                    # assert 0 == 1
                # perform guidance
                # assert 0 == 1
                if do_classifier_free_guidance:
                    # for each: bxN, 4, 28, 50
                    noise_pred_uncond, noise_pred_text = (noise_pred_total / counter).chunk(2)
                    noise_pred_total = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                noise_pred_total = rearrange(noise_pred_total, 'b n ... -> (b n) ...', n=N_cam)
                # noise_pred_total.shape = (44 , 6, 4, 28, 50)
                # compute the previous noisy sample x_t -> x_t-1
                # NOTE: is the scheduler use randomness, please handle the logic
                # for generator.
                latents = rearrange(latents, 'b n ... -> (b n) ...')
                latents = self.scheduler.step(
                    noise_pred_total, t, latents, **extra_step_kwargs
                ).prev_sample

                # =============================================================
                # now we add dimension back, use 5-dim data.
                # NOTE: only `latents` is updated through the loop
                latents = rearrange(latents, '(b n) ... -> b n ...', n=N_cam)

                # call the callback, if provided
                if l == len(timesteps) - 1 or ((l + 1) > num_warmup_steps and (
                        l + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and l % callback_steps == 0:
                        callback(l, t, latents)

        # BEV: here rebuild the shapes back. post-process still assume
        # latents, no need for b, n, 4, 28, 50
        # prompt_embeds, no need for b, len, 768
        # image, no need for b, c, 200, 200
        # BEV end

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(
                self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()

        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil":
            # 8. Post-processing
            image = self.decode_latents(latents, decode_bs)

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(
                image, device, prompt_embeds.dtype
            )

            # 10. Convert to PIL
            image = self.numpy_to_pil_double(image)
        else:
            # 8. Post-processing
            image = self.decode_latents(latents, decode_bs)

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(
                image, device, prompt_embeds.dtype
            )

        # Offload last model to CPU
        if hasattr(
                self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return BEVStableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )