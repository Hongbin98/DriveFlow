from typing import Optional, Tuple, Union
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import label
import os
import kornia


def scale_noise(
    scheduler,
    sample: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    noise: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """
    Foward process in flow-matching

    Args:
        sample (`torch.FloatTensor`):
            The input sample.
        timestep (`int`, *optional*):
            The current timestep in the diffusion chain.

    Returns:
        `torch.FloatTensor`:
            A scaled input sample.
    """
    # if scheduler.step_index is None:
    scheduler._init_step_index(timestep)

    sigma = scheduler.sigmas[scheduler.step_index]
    sample = sigma * noise + (1.0 - sigma) * sample

    return sample


# for flux
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def calc_v_sd3(pipe, src_tar_latent_model_input, src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, src_guidance_scale, tar_guidance_scale, t, edit_info=None):
    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timestep = t.expand(src_tar_latent_model_input.shape[0])

    with torch.no_grad():
    # predict the noise for the source prompt
        noise_pred_src_tar = pipe.transformer(
            hidden_states=src_tar_latent_model_input,
            timestep=timestep,
            encoder_hidden_states=src_tar_prompt_embeds,
            pooled_projections=src_tar_pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False
        )[0]

    # perform guidance source
    if pipe.do_classifier_free_guidance:
        src_noise_pred_uncond, src_noise_pred_text, tar_noise_pred_uncond, tar_noise_pred_text = noise_pred_src_tar.chunk(4)
        noise_pred_src = src_noise_pred_uncond + src_guidance_scale * (src_noise_pred_text - src_noise_pred_uncond)
        noise_pred_tar = tar_noise_pred_uncond + tar_guidance_scale * (tar_noise_pred_text - tar_noise_pred_uncond)

    return noise_pred_src, noise_pred_tar


def calc_v_flux(pipe, latents, prompt_embeds, pooled_prompt_embeds, guidance, text_ids, latent_image_ids, t, edit_info=None):
    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timestep = t.expand(latents.shape[0])

    with torch.no_grad():
        # # predict the noise for the source prompt
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
            edit_info=edit_info
        )[0]

    return noise_pred


def get_token_mask_with_soft_weight_kitti(
    bboxes, image_size, is_SD3=True,
    min_ratio=0.0, max_ratio=0.5,
    max_obj_ratio=0.25
):
    # image_size: (width, height)
    if is_SD3:
        patch_size = 8
    else:
        patch_size = 16

    width, height = image_size
    W_feat = width // patch_size
    H_feat = height // patch_size

    mask = torch.zeros((H_feat, W_feat), dtype=torch.bool)
    weight = torch.ones((H_feat, W_feat), dtype=torch.float32) * 1.0 

    img_area = width * height

    scale_x = W_feat / width
    scale_y = H_feat / height

    for box in bboxes:
        x1, y1, x2, y2 = box["bbox_corners"]
        w, h = x2 - x1, y2 - y1
        area_ratio = (w * h) / img_area 

        interp = min(area_ratio / max_obj_ratio, 1.0)
        weight_val = min_ratio + (max_ratio - min_ratio) * interp

        x1_t, x2_t = int(x1 * scale_x), int(x2 * scale_x)
        y1_t, y2_t = int(y1 * scale_y), int(y2 * scale_y)

        x1_t = max(x1_t, 0)
        x2_t = min(x2_t, W_feat)
        y1_t = max(y1_t, 0)
        y2_t = min(y2_t, H_feat)

        mask[y1_t:y2_t, x1_t:x2_t] = True
        weight[y1_t:y2_t, x1_t:x2_t] = weight_val

    return mask, weight


def split_freq(x, sigma=3):
    low  = kornia.filters.gaussian_blur2d(x, (5, 5), (sigma, sigma))
    high = x - low
    return low, high


def Driveflow_adapt(pipe,
    scheduler,
    x_src,
    src_prompt,
    tar_prompt,
    negative_prompt,
    T_steps: int = 50,
    n_avg: int = 1,
    src_guidance_scale: float = 3.5,
    tar_guidance_scale: float = 13.5,
    n_min: int = 0,
    n_max: int = 15,
    bboxes=None,
    img_size=None,
    noise_dict=None,
    ):
    
    device = x_src.device

    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None)

    num_warmup_steps = max(len(timesteps) - T_steps * scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)
    pipe._guidance_scale = src_guidance_scale
    
    # src prompts
    (
        src_prompt_embeds,
        src_negative_prompt_embeds,
        src_pooled_prompt_embeds,
        src_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    # tar prompts
    pipe._guidance_scale = tar_guidance_scale
    (
        tar_prompt_embeds,
        tar_negative_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )
 
    # CFG prep
    src_tar_prompt_embeds = torch.cat([src_negative_prompt_embeds, src_prompt_embeds, tar_negative_prompt_embeds, tar_prompt_embeds], dim=0)
    src_tar_pooled_prompt_embeds = torch.cat([src_negative_pooled_prompt_embeds, src_pooled_prompt_embeds, tar_negative_pooled_prompt_embeds, tar_pooled_prompt_embeds], dim=0)
    
    # initialize our ODE Zt_edit_1=x_src
    zt_edit = x_src.clone()

    # ===== object preservation =====
    mask, _ = get_token_mask_with_soft_weight_kitti(
        bboxes, image_size=img_size, is_SD3=True
    )
    mask = mask.to(zt_edit.device).unsqueeze(0).unsqueeze(0)

    edit_info = {'obj_mask': mask}
    for i, t in tqdm(enumerate(timesteps)):
        
        if T_steps - i > n_max:
            continue
        
        t_i = t/1000
        if i+1 < len(timesteps): 
            t_im1 = (timesteps[i+1])/1000
        else:
            t_im1 = torch.zeros_like(t_i).to(t_i.device)
        
        if T_steps - i > n_min:
            # Calculate the average of the V predictions
            V_delta_avg = torch.zeros_like(x_src)
            for k in range(n_avg):
                fwd_noise = noise_dict[i].to(x_src.dtype).to(x_src.device) if noise_dict else torch.randn_like(x_src).to(x_src.device)
                if fwd_noise.shape[0] == 1: fwd_noise.expand(x_src.shape[0], -1, -1, -1)
                
                zt_src = (1-t_i)*x_src + (t_i)*fwd_noise

                zt_tar = zt_edit + zt_src - x_src

                src_tar_latent_model_input = torch.cat([zt_src, zt_src, zt_tar, zt_tar]) if pipe.do_classifier_free_guidance else (zt_src, zt_tar) 

                Vt_src, Vt_tar = calc_v_sd3(pipe, src_tar_latent_model_input,src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, src_guidance_scale, tar_guidance_scale, t, edit_info)

                # pre-process
                V_src = Vt_src.detach()                          # reference
                V_tar = Vt_tar.detach().clone().requires_grad_(True)

                # inner loop
                n_inner    = 5
                base_lr    = 9e-2
                momentum   = 0.9
                loss_scale = 1e4          
                λ_h, λ_l   = 5, 1
                λ_h_bg = 1
                sigma = 1

                optimizer  = torch.optim.SGD([V_tar], lr=base_lr, momentum=momentum)

                for _ in range(n_inner):
                    optimizer.zero_grad(set_to_none=True)

                    # frequency decomposition
                    low_tar,  high_tar  = split_freq(V_tar.float(), sigma=sigma)
                    low_src,  high_src  = split_freq(V_src.float(), sigma=sigma)

                    loss_struct = ((high_tar - high_src).pow(2) * mask).mean()
                    loss_struct_bg = ((high_tar - high_src).pow(2) * (1.0 - mask.float())).mean()

                    cos_sim = torch.nn.functional.cosine_similarity(low_tar, low_src, dim=1)  # [B,H,W]
                    loss_style = (cos_sim * (1.0 - mask.float())).mean()                    

                    loss = (λ_h * loss_struct + λ_l * loss_style + λ_h_bg * loss_struct_bg) * loss_scale
                    loss.backward()
                    optimizer.step()                

                V_delta_avg += (1.0 / n_avg) * (V_tar.detach() - V_src)

            # propagate direct ODE
            zt_edit = zt_edit.to(torch.float32)
            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg
            zt_edit = zt_edit.to(V_delta_avg.dtype)

    return zt_edit
