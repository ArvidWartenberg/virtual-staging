# Dirty inference code to run on single example
from cldm.ddim_hacked import DDIMSampler
from cldm.model import create_model, load_state_dict
from annotator.util import resize_image, HWC3
from pytorch_lightning import seed_everything
from owlsam_endpoint import OwlSamSegmentor
from constants import DATA_DIR, EMPTY, STAGED, CAPTION_STAGED
from share import *
import config
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
import random
import torch
import numpy as np
import gradio as gr
import einops
import cv2
import sys
sys.path.append('./ControlNet')


# Default params taken from example script in controlnet

def process(model, ddim_sampler, input_image, prompt, a_prompt, n_prompt, num_samples=1, image_resolution=512, ddim_steps=30, guess_mode=False, strength=1, scale=9.0, seed=42, eta=0.0):
    """Generate image using specific model/sampler and rgb control input image."""
    with torch.no_grad():
        detected_map = HWC3(input_image)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(
            detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [
            model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [
            model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else (
            [strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c')
                     * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return results[0]


if __name__ == "__main__":

    # Load validation pairs, unseen by models
    with open(os.path.join(DATA_DIR, "paired_eval_datalist.json"), 'rb') as f:
        eval_datalist = json.load(f)

    # Run inference
    a_prompt = 'best quality, extremely detailed'
    n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

    # For the layout generation phase (just predicting "diff mask" in this case), we use the empty->staged ControlNet
    # to generate a rough scene in the room, and then extract a somewhat refined diff mask by segmenting out the
    # inpainted features that we are interested in. This is later used to guide which regions to "furnish" in the second stage.
    empty_to_staged = create_model(
        './models/control_v11p_sd15_inpaint.yaml').cpu()
    empty_to_staged.load_state_dict(load_state_dict(
        './models/empty_to_staged.ckpt', location='cuda'), strict=False)
    empty_to_staged = empty_to_staged.cuda()
    empty_to_staged_ddim_sampler = DDIMSampler(empty_to_staged)
    # Segmentation model
    segmentation_model = OwlSamSegmentor()

    # For the generative phase, we use a different checkpoint that is specifically trained on masked inputs
    # s.t. it does not make too big changes in regions where we want to perserve the original content
    masked_to_staged = create_model(
        './models/control_v11p_sd15_inpaint.yaml').cpu()
    masked_to_staged.load_state_dict(load_state_dict(
        './models/masked_to_staged.ckpt', location='cuda'), strict=False)
    masked_to_staged = masked_to_staged.cuda()
    masked_to_staged_ddim_sampler = DDIMSampler(masked_to_staged)

    for ix, element in tqdm(enumerate(eval_datalist)):

        # Extract the empty and staged images
        # We dont touch the staged before using it for comparison
        empty = cv2.imread(element[EMPTY])
        staged = cv2.imread(element[STAGED])

        h, w, c = staged.shape

        # Do not forget that OpenCV read images in BGR order.
        staged = cv2.cvtColor(staged, cv2.COLOR_BGR2RGB)
        empty = cv2.cvtColor(empty, cv2.COLOR_BGR2RGB)

        # Resize ControlNet condition for network
        empty_input = cv2.resize(
            empty, (512, 512), interpolation=cv2.INTER_LINEAR)

        # Extract prompt (this needs to be predicted somehow in the real world, but out of scope here)
        prompt = element[CAPTION_STAGED]

        # Dirty initial scene generation based on empty room
        initial_scene = process(
            empty_to_staged, empty_to_staged_ddim_sampler, empty_input, "a furnished room", a_prompt, n_prompt)

        # Retrieve the "generated diff mask", corresponding to
        # a binary mask covering all objects of interest generated in the scene
        scene_layout_mask = segmentation_model.get_staged_objects_mask(
            initial_scene)

        # Compute the agnostic representation
        agnostic = (
            empty_input*(1-scene_layout_mask[..., None]/255).astype(np.float32)).astype(np.uint8)

        # Generate final scene using the agnostic room as input to the inpainting controlnet
        # finetuned on the small dataset.
        result = process(masked_to_staged, masked_to_staged_ddim_sampler,
                         agnostic, "a furnished room", a_prompt, n_prompt)

        # Resize to original resolution and make a fancy plot
        scene_layout_mask = cv2.resize(scene_layout_mask, (w, h))
        initial_scene = cv2.resize(initial_scene, (w, h))
        agnostic = cv2.resize(agnostic, (w, h))
        result = cv2.resize(result, (w, h))

        # Ugly ugly ugly code to make a pretty figure
        plt.figure(figsize=(9, 4))
        plt.subplot(2, 3, 1)
        plt.imshow(empty)
        plt.title("Empty", fontsize=10)
        plt.axis("off")
        plt.subplot(2, 3, 2)
        plt.imshow(scene_layout_mask)
        plt.title("Generated layout mask", fontsize=10)
        plt.axis("off")
        plt.subplot(2, 3, 3)
        plt.imshow(agnostic)
        plt.title("Agnostic", fontsize=10)
        plt.axis("off")
        plt.subplot(2, 3, 4)
        plt.imshow(staged)
        plt.title("staged", fontsize=10)
        plt.axis("off")
        plt.subplot(2, 3, 5)
        plt.imshow(initial_scene)
        plt.title("Initial scene", fontsize=10)
        plt.axis("off")
        plt.subplot(2, 3, 6)
        plt.imshow(result)
        plt.title("Result", fontsize=10)
        plt.axis("off")
        plt.savefig(f"inference_example_{ix}.png", dpi=300)
        plt.tight_layout()
        plt.close()
