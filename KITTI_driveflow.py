import torch
from diffusers import StableDiffusion3Pipeline
from diffusers import FluxPipeline
from PIL import Image
import argparse
import random 
import numpy as np
import yaml
import os
from utils import FlowEditSD3_with_TTA_adapt_abla_0728
import time 


def load_kitti_bboxes(img_id, label_root, orig_size, target_size):
    label_path = os.path.join(label_root, f"{img_id}.txt")
    orig_w, orig_h = orig_size
    target_w, target_h = target_size
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    bboxes = []

    with open(label_path, "r") as f:
        for idx, line in enumerate(f):
            items = line.strip().split()
            x1, y1, x2, y2 = map(float, items[4:8])
            x1_s = x1 * scale_x
            x2_s = x2 * scale_x
            y1_s = y1 * scale_y
            y2_s = y2 * scale_y

            bboxes.append({
                "category_name": items[0],
                "bbox_corners": [x1_s, y1_s, x2_s, y2_s],
                "token": f"{items[0]}_{idx}"
            })
    return bboxes


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device_number", type=int, default=0, help="device number to use")
    parser.add_argument("--exp_yaml", type=str, default="SD3_exp.yaml", help="experiment yaml file")

    args = parser.parse_args()

    # set device
    device_number = args.device_number
    device = torch.device(f"cuda:{device_number}" if torch.cuda.is_available() else "cpu")

    # load exp yaml file to dict
    exp_yaml = args.exp_yaml
    with open(exp_yaml) as file:
        exp_configs = yaml.load(file, Loader=yaml.FullLoader)

    device = torch.device(f"cuda:{device_number}" if torch.cuda.is_available() else "cpu")
    model_type = exp_configs[0]["model_type"] # currently only one model type per run

    if model_type == 'FLUX':
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
    elif model_type == 'SD3':
        pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")
    
    scheduler = pipe.scheduler
    pipe = pipe.to(device)

    for exp_dict in exp_configs:
        T_steps = exp_dict["T_steps"]
        n_avg = exp_dict["n_avg"]
        src_guidance_scale = exp_dict["src_guidance_scale"]
        tar_guidance_scale = 13.5
        n_min = exp_dict["n_min"]
        n_max = 33
        seed = 4

        # set seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        src_prompt = 'An urban scene in the daytime.'
        tar_prompt = 'A snowy scene in the winter.'
        negative_prompt =  "" # optionally add support for negative prompts (SD3)

        all_image_todo = []
        img_root = 'data/kitti/training/image_2'
        label_root = 'data/kitti/training/label_2'
        train_txt = 'data/kitti/training/ImageSets/train.txt'
        with open(train_txt, 'r') as f:
            lines = f.readlines()

        #### for visualization
        lines = lines[:5]
        for line in lines:
            img_id = line.strip()
            img_path = os.path.join(img_root, f"{img_id}.png")
            if os.path.exists(img_path):
                all_image_todo.append(img_path)
            else:
                print(f"Warning: {img_path} not found!")

        print(f"Total images: {len(all_image_todo)}")

        noise_dict = {}
        all_image_todo.sort()
        for i in range(T_steps):
            noise_dict[i] = torch.randn(1, 16, 46, 156,)

        save_dir = "outputs/kitti_sd3_driveflow/snow/"
        for i in range(len(all_image_todo)):
            # if os.path.exists(f"{save_dir}/{all_image_todo[i].split('/')[-1]}"): continue
            image_src_path = all_image_todo[i]
            
            # load image
            image = Image.open(image_src_path)
            orig_w, orig_h = image.size
            image = image.resize((1248, 368), resample=Image.LANCZOS)
            image_src = pipe.image_processor.preprocess(image)
            # cast image to half precision
            image_src = image_src.to(device).half()
            with torch.autocast("cuda"), torch.inference_mode():
                x0_src_denorm = pipe.vae.encode(image_src).latent_dist.mode()
            x0_src = (x0_src_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
            # send to cuda
            x0_src = x0_src.to(device)
            
            #### load bboxes
            start_time = time.time()
            bboxes = load_kitti_bboxes(all_image_todo[i].split('/')[-1].replace('.png', ''), label_root, (orig_w, orig_h), target_size=image.size)
            if model_type == 'SD3':
                x0_tar = FlowEditSD3_with_TTA_adapt_abla_0728(pipe,
                                    scheduler,
                                    x0_src,
                                    src_prompt,
                                    tar_prompt,
                                    negative_prompt,
                                    T_steps,
                                    n_avg,
                                    src_guidance_scale,
                                    tar_guidance_scale,
                                    n_min,
                                    n_max,
                                    bboxes,
                                    image.size,
                                    noise_dict=noise_dict)
            end_time = time.time()
            total_time = end_time - start_time
            print(f"generation time: {total_time:.2f} seconds")
            x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
            with torch.autocast("cuda"), torch.inference_mode():
                image_tar = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
            image_tar = pipe.image_processor.postprocess(image_tar)

            # make sure to create the directories before saving
            os.makedirs(save_dir, exist_ok=True)

            img_restored = image_tar[0].resize((orig_w, orig_h), resample=Image.LANCZOS)                    
            img_restored.save(f"{save_dir}/{all_image_todo[i].split('/')[-1]}")
            print(f"{save_dir}/{all_image_todo[i].split('/')[-1]}")

    print("Done")
