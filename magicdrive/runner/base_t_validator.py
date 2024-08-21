from typing import Tuple, List
import logging
from tqdm import tqdm
from PIL import Image

import numpy as np
import torch
import torchvision
from torchvision.transforms.functional import to_pil_image, to_tensor

from accelerate.tracking import GeneralTracker

import os

from magicdrive.runner.utils import (
    visualize_map,
    img_m11_to_01,
    concat_6_views,
    img_concat_v,
)
from magicdrive.runner.base_validator import (
    BaseValidator,
)
from magicdrive.misc.common import move_to
from magicdrive.misc.test_utils import draw_box_on_imgs
from magicdrive.pipeline.pipeline_bev_controlnet import (
    BEVStableDiffusionPipelineOutput,
)
from magicdrive.dataset.utils import collate_fn_single
from magicdrive.runner.utils import concat_6_views
from magicdrive.networks.unet_addon_rawbox import BEVControlNetModel

from moviepy.editor import *

def format_image(image_list):
    formatted_images = []
    for image in image_list:
        formatted_images.append(np.asarray(image))

    # formatted_images = np.stack(formatted_images)
    # 0-255 np -> 0-1 tensor -> grid -> 0-255 pil -> np
    formatted_images = torchvision.utils.make_grid(
        [to_tensor(im) for im in formatted_images], nrow=1)
    formatted_images = np.asarray(
        to_pil_image(formatted_images))
    return formatted_images

def output_func(x): return concat_6_views(x, oneline=True)

def make_video_with_filenames(filenames, outname, fps=2):
    clips = [ImageClip(m).set_duration(1 / fps) for m in filenames]
    concat_clip = concatenate_videoclips(clips, method="compose")
    concat_clip.write_videofile(outname, fps=fps)


class BaseTValidator(BaseValidator):
    def construct_visual(self, images_list, val_input, with_box):
        frame_list = []
        frame_list_wb = []
        for idx, framei in enumerate(images_list):
            frame = concat_6_views(framei, oneline=True)
            if with_box:
                frame_with_box = concat_6_views(
                    draw_box_on_imgs(
                        self.cfg, idx, val_input, framei),
                    oneline=True)
            frame_list.append(frame)
            frame_list_wb.append(frame_with_box)
        frames = img_concat_v(*frame_list)
        if with_box:
            frames_wb = img_concat_v(*frame_list_wb)
        else:
            frames_wb = None
        return frames, frames_wb

    def validate(
        self,
        controlnet: BEVControlNetModel,
        unet,
        trackers: Tuple[GeneralTracker, ...],
        step, save_path, weight_dtype, device
    ):
        logging.info(f"[{self.__class__.__name__}] Running validation... ")
        print("Running validation' weight_dtype:", weight_dtype)
        pipeline = self.prepare_pipe(controlnet, unet, weight_dtype, device)
        ori_save_path = save_path
        image_logs = []
        total_run_times = len(
            self.cfg.runner.validation_index) * self.cfg.runner.validation_times
        if self.cfg.runner.pipeline_param['init_noise'] == 'both':
            total_run_times *= 2
        progress_bar = tqdm(
            range(0, total_run_times), desc="Val Steps")

        for validation_i in self.cfg.runner.validation_index:
            raw_data = self.val_dataset[validation_i]  # cannot index loader
            val_input = collate_fn_single(
                raw_data, self.cfg.dataset.template, is_train=False,
                bbox_mode=self.cfg.model.bbox_mode,
                bbox_view_shared=self.cfg.model.bbox_view_shared,
            )
            # camera_emb = self._embed_camera(val_input["camera_param"])
            camera_param = val_input["camera_param"].to(weight_dtype)

            # let different prompts have the same random seed
            if self.cfg.seed is None:
                generator = None
            else:
                generator = torch.Generator(device=device).manual_seed(
                    self.cfg.seed
                )
            gen_imgs_list_ori = [[] for _ in range(len(val_input["captions"]))]
            def run_once(pipe_param):
                for _ in range(self.cfg.runner.validation_times):
                    with torch.autocast("cuda"):
                        image: BEVStableDiffusionPipelineOutput = pipeline(
                            prompt=val_input["captions"],
                            image=val_input["bev_map_with_aux"],
                            camera_param=camera_param,
                            height=self.cfg.dataset.image_size[0],
                            width=self.cfg.dataset.image_size[1],
                            generator=generator,
                            bev_controlnet_kwargs=val_input["kwargs"],
                            **pipe_param,
                        )
                    image: List[List[Image.Image]] = image.images
                    for bi, imgs in enumerate(image):
                        gen_imgs_list_ori[bi].append(imgs)

                    progress_bar.update(1)

            # for each input param, we generate several times to check variance.
            pipeline_param = {
                k: v for k, v in self.cfg.runner.pipeline_param.items()}
            if self.cfg.runner.pipeline_param['init_noise'] != 'both':
                run_once(pipeline_param)
            else:
                pipeline_param['init_noise'] = "same"
                run_once(pipeline_param)
                pipeline_param['init_noise'] = "rand"
                run_once(pipeline_param)

            # save gen
            batch_img_index = 0
            gen_img_paths = {}
            
            for gen_imgs_list in gen_imgs_list_ori:
                for ti, gen_imgs in enumerate(gen_imgs_list):
                    gen_img = output_func(gen_imgs)
                    
                    save_path = os.path.join(
                        str(ori_save_path), "frames",
                    )
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    
                    save_path = os.path.join(
                        save_path, f"{validation_i}_{batch_img_index}_gen{ti}_.png"
                    )
                    gen_img.save(save_path)
                    if ti in gen_img_paths:
                        gen_img_paths[ti].append(save_path)
                    else:
                        gen_img_paths[ti] = [save_path]
                batch_img_index += 1

            for k, v in gen_img_paths.items():
                make_video_with_filenames(
                    v, os.path.join(
                        str(ori_save_path),
                        f"{validation_i}_{batch_img_index}_gen{k}.mp4"),
                    fps=12)
