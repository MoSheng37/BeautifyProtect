from __future__ import annotations

import math
import random
import sys
from argparse import ArgumentParser

import einops
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast
import matplotlib.pyplot as plt

sys.path.append("./stable_diffusion")
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

from stable_diffusion.ldm.util import instantiate_from_config
import os

from scipy.ndimage import correlate
from skimage.metrics import structural_similarity as ssim
from Image_Blend import blend_main
import cv2
from loguru import logger
from PIL import Image,ImageEnhance


from transformers import logging
logging.set_verbosity_warning()
logging.set_verbosity_error()

import time

import random
random.seed(37)

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)


def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model.") :]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }

    model = instantiate_from_config(config.model)

    m, u = model.load_state_dict(sd, strict=False)

    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png", ".PNG",".JPEG"]
def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


# 双边滤波,image为numpy
def apply_bilateral_filter(image,param1,param2,param3):
    filtered_img = cv2.bilateralFilter(image,param1,param2,param3)
    return filtered_img
# 锐度和对比度调节
def sharpness_and_contrast(image,param6,param7,param8,param9):
    image_pil = Image.fromarray(image)
    #锐化
    enh_img = ImageEnhance.Sharpness(image_pil)
    image_sharped = enh_img.enhance(param6)
    # 对比度
    con_img = ImageEnhance.Contrast(image_sharped)
    image_con = con_img.enhance(param7)

    #饱和度
    col_image = ImageEnhance.Color(image_con)
    image_color = col_image.enhance(param8)
    
    # 亮度
    bri_image = ImageEnhance.Brightness(image_color)
    image_brightness = bri_image.enhance(param9)
    
    return np.array(image_brightness)
# 图像融合
def blend_images(image,filtered_img,param4,param5):
    blended_image = cv2.addWeighted(image,param4,filtered_img,param5,0)
    return blended_image    
def image_processing(img,population,edit_name,i):
    
    file_name,file_extension = os.path.splitext(edit_name)
    edit_image_name = file_name + "_{}".format(i) +file_extension
    
    param1,param2,param3,param4,param5,param6,param7,param8,param9 = population
    if isinstance(img,str):
        image_cv2 = cv2.imread(img)#[0,255],numpy,[w,h,c]
        # 将 BGR 图像转换为 RGB
        image = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        basename = os.path.basename(img)
    if isinstance(img,np.ndarray):
        image = img
        basename = "test123"
    # print(f"image={image.shape}")
    # print(f"image={type(image)}")
    # print(f"image={image}")
    # 双边滤波
    filtered_img = apply_bilateral_filter(image,param1,param2,param3)
    logger.info(f"filtered_img={filtered_img.shape}")
    # 图像融合
    blended_image = blend_images(image,filtered_img,param4,param5)
    logger.info(f"blended_image={blended_image.shape}")
    # 锐度和对比度调节,result_image为numpy类型
    result_image = sharpness_and_contrast(blended_image,param6,param7,param8,param9) 
    #logger.info(f"result_image={result_image}")
    result_image_max = np.max(result_image)
    result_image_min = np.min(result_image)
    logger.info(f"result_image_max={result_image_max},result_image_min={result_image_min}")
    logger.info(f"result_image={result_image.shape}")
    logger.info(f"result_image={type(result_image)}")
    
    
    save_dir = "/abspath/record/"
    os.makedirs(save_dir,exist_ok=True)
    save_path = os.path.join(save_dir,edit_image_name)
    image_pil = Image.fromarray(result_image.astype('uint8'))
    #image_pil = Image.fromarray(superposition_mask)
    image_pil.save(save_path)

    return save_path
    

def process_image(input_path):
    input_image = Image.open(input_path).convert("RGB")
    input_image = ImageOps.fit(input_image, (512, 512), method=Image.LANCZOS)
    
    return input_image
    


def edit_main_copy(input_image,output_path,instruction,seed):
    parser = ArgumentParser()
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="configs/generate.yaml", type=str)
    parser.add_argument("--ckpt", default="checkpoints/instruct-pix2pix-00-22000.ckpt", type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)

    parser.add_argument("--cfg-text", default=7.5, type=float)
    parser.add_argument("--cfg-image", default=1.5, type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    
    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.eval().cuda(6)
    
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    print(f"33333")
    null_token = model.get_learned_conditioning([""])
    print(f"44444")
    #seed = random.randint(0, 100000) if args.seed is None else args.seed
    print(f"seed={seed}")
    
    
    #if args.edit == "":
    if instruction == "":
        #input_image.save(args.output)
        input_image.save(output_path)
        return
    print(f"55555")
    with torch.no_grad(), autocast("cuda"), model.ema_scope():
        cond = {}
        #cond["c_crossattn"] = [model.get_learned_conditioning([args.edit])]
        cond["c_crossattn"] = [model.get_learned_conditioning([instruction])]
        #input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
        print(f"1111")
        input_image = rearrange(input_image, "h w c -> 1 c h w").to(model.device)
        cond["c_concat"] = [model.encode_first_stage(input_image).mode()]
        print(f"222")
        uncond = {}
        uncond["c_crossattn"] = [null_token]
        uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

        sigmas = model_wrap.get_sigmas(args.steps)

        extra_args = {
            "cond": cond,
            "uncond": uncond,
            "text_cfg_scale": args.cfg_text,
            "image_cfg_scale": args.cfg_image,
        }
        torch.manual_seed(seed)
        z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
        z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
        x = model.decode_first_stage(z)
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        x_copy = x.clone()
        x = 255.0 * rearrange(x, "1 c h w -> h w c")
        edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
    #edited_image.save(args.output)
    edited_image.save(output_path)
    #zh
    #return edited_image,seed
    return x_copy
    
def edit_main(input_path,output_path,instruction):

    #device= torch.device("cuda:3")
    parser = ArgumentParser()
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="configs/generate.yaml", type=str)
    parser.add_argument("--ckpt", default="checkpoints/instruct-pix2pix-00-22000.ckpt", type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)
    # parser.add_argument("--input", required=True, type=str)
    # parser.add_argument("--output", required=True, type=str)
    # parser.add_argument("--edit", required=True, type=str)
    parser.add_argument("--cfg-text", default=7.5, type=float)
    parser.add_argument("--cfg-image", default=1.5, type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    #zh
    # args.input = path1
    # args.output = path2
    # args.edit = instruction
    #zh
    
    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.eval().cuda(5)
    
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    print(f"33333")
    null_token = model.get_learned_conditioning([""])
    print(f"44444")
    seed = random.randint(0, 100000) if args.seed is None else args.seed

    print(f"seed={seed}")
    #input_image = Image.open(args.input).convert("RGB")
    input_image = Image.open(input_path).convert("RGB")
    width, height = input_image.size
    factor = args.resolution / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width = int((width * factor) // 64) * 64
    height = int((height * factor) // 64) * 64
    #input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)
    input_image = ImageOps.fit(input_image, (width, height), method=Image.LANCZOS)
    logger.info(f"input_image={input_image.size}")
    #if args.edit == "":
    if instruction == "":
        #input_image.save(args.output)
        input_image.save(output_path)
        return
    print(f"55555")
    with torch.no_grad(), autocast("cuda"), model.ema_scope():
        cond = {}
        #cond["c_crossattn"] = [model.get_learned_conditioning([args.edit])]
        cond["c_crossattn"] = [model.get_learned_conditioning([instruction])]
        input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
        print(f"1111")
        input_image = rearrange(input_image, "h w c -> 1 c h w").to(model.device)
        logger.info(f"input_image={input_image.shape}")
        cond["c_concat"] = [model.encode_first_stage(input_image).mode()]

        # exit()
        print(f"222")
        uncond = {}
        uncond["c_crossattn"] = [null_token]
        uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

        sigmas = model_wrap.get_sigmas(args.steps)

        extra_args = {
            "cond": cond,
            "uncond": uncond,
            "text_cfg_scale": args.cfg_text,
            "image_cfg_scale": args.cfg_image,
        }
        torch.manual_seed(seed)
        z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
        z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
        x = model.decode_first_stage(z)
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        x = 255.0 * rearrange(x, "1 c h w -> h w c")
        edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
    #edited_image.save(args.output)
    edited_image.save(output_path)
    #zh
    return edited_image,seed

def protected_edit_main(input_path,output_path,instruction,seed):

    
    parser = ArgumentParser()
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="configs/generate.yaml", type=str)
    parser.add_argument("--ckpt", default="checkpoints/instruct-pix2pix-00-22000.ckpt", type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)

    parser.add_argument("--cfg-text", default=7.5, type=float)
    parser.add_argument("--cfg-image", default=1.5, type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()


    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.eval().cuda(5)
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    print(f"33333")

    null_token = model.get_learned_conditioning([""])
    print(f"44444")

    
    print(f"seed={seed}")
    #input_image = Image.open(args.input).convert("RGB")
    input_image = Image.open(input_path).convert("RGB")
    width, height = input_image.size
    logger.info(f"input_image11={input_image.size}")
    factor = args.resolution / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width = int((width * factor) // 64) * 64
    height = int((height * factor) // 64) * 64

    input_image = ImageOps.fit(input_image, (width, height), method=Image.LANCZOS)
    
    #if args.edit == "":
    if instruction == "":
        #input_image.save(args.output)
        input_image.save(output_path)
        return
    print(f"55555")
    with torch.no_grad(), autocast("cuda"), model.ema_scope():
        cond = {}
        #cond["c_crossattn"] = [model.get_learned_conditioning([args.edit])]
        cond["c_crossattn"] = [model.get_learned_conditioning([instruction])]
        
        input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
        
        print(f"1111")
        input_image = rearrange(input_image, "h w c -> 1 c h w").to(model.device)
        cond["c_concat"] = [model.encode_first_stage(input_image).mode()]
        print(f"222")
        uncond = {}
        uncond["c_crossattn"] = [null_token]
        uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

        sigmas = model_wrap.get_sigmas(args.steps)

        extra_args = {
            "cond": cond,
            "uncond": uncond,
            "text_cfg_scale": args.cfg_text,
            "image_cfg_scale": args.cfg_image,
        }
        torch.manual_seed(seed)
        z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
        z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
        print(f"z={z.shape}")
        x = model.decode_first_stage(z)
        x_max = torch.max(x)
        x_min = torch.min(x)
        print(f"x_max={x_max},x_min={x_min}")####经过decoder得到的x范围在(-1,1)
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)#tensor,将张量的范围从[-1,1]缩放到[0,1]
        print(f"torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0),x={x.shape}")
        x = 255.0 * rearrange(x, "1 c h w -> h w c")#tensor -> numpy,[0,1]
        print(f"x={x.shape}")
        edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
        print(f"edited_image={edited_image.size}")
    #edited_image.save(args.output)
    edited_image.save(output_path)
    #zh
    return edited_image,seed
    #zh




if __name__ == "__main__":
    
    start_time = time.time()
    
    instruction = "Make it a Van Gogh painting"  
    
    images_name = get_image_list("/home/Newdisk2/zhanghe/Data/img_align_celeba")
    
    for k in range(len(images_name)):
        if k == 100:
            break
        #设置seed
        seeds = []
        
        
        #处理图像路径
        input_path = images_name[k]
        image_basename = os.path.basename(input_path)
        file_name,file_extension = os.path.splitext(image_basename)
        
        #正常编辑的输出路径
        output_dir = "abspath/edited_test_images/record/{}/".format(k)
        os.makedirs(output_dir,exist_ok=True)
        
        
        for i in range(10):
            edit_image_name = file_name + "_{}".format(i) +file_extension    
            
            output_path = os.path.join(output_dir,edit_image_name)
            _,seed = edit_main(input_path,output_path,instruction)
            seeds.append(seed)
            

            uap_path = "/home/Newdisk2/uap_perturbation_0.21121.npy"
            
            uap_image = np.load(uap_path)#uap_image为[-1,1],float32
            
            input_image = process_image(input_path)
            print(f"input_image={input_image.size}")
            
            perturb_image  = (2 * torch.tensor(np.array(input_image)).float() / 255 - 1) + uap_image
           
            perturb_dir = "abspath/perturb_test_images/record/{}/".format(k)
            os.makedirs(perturb_dir,exist_ok=True)
            perturb_path = os.path.join(perturb_dir,file_name + "_{}".format(i) +file_extension)
            x = torch.clamp((perturb_image + 1.0) / 2.0, min=0.0, max=1.0)#tensor,将张量的范围从[-1,1]缩放到[0,1]
           
            
            edited_image = Image.fromarray((x*255).type(torch.uint8).cpu().numpy())
            print(f"edited_image={edited_image.size}")
            edited_image.save(perturb_path)
            
            population = [5,50,50,0.6,0.4,1,1.0,1.1,1.1]#美白
            beatuy_path = image_processing(input_path,population,image_basename,i)
            beauty_output_dir = "abspath/record/{}/".format(k)
            os.makedirs(beauty_output_dir,exist_ok=True)
            beauty_output_path = os.path.join(beauty_output_dir,edit_image_name)
            protected_edit_main(beatuy_path,beauty_output_path,instruction,seed)
            
            beauty_image = process_image(beatuy_path)
            print(f"beauty_image={beauty_image.size}")
            beauty_perturb_image  = (2 * torch.tensor(np.array(beauty_image)).float() / 255 - 1) + uap_image
            beauty_perturb_dir = "abspath/record/{}/".format(k)
            os.makedirs(beauty_perturb_dir,exist_ok=True)
            beauty_perturb_path = os.path.join(beauty_perturb_dir,file_name + "_{}".format(i) +file_extension)
            x1 = torch.clamp((beauty_perturb_image + 1.0) / 2.0, min=0.0, max=1.0)#tensor,将张量的范围从[-1,1]缩放到[0,1]
            
            beauty_edited_image = Image.fromarray((x1*255).type(torch.uint8).cpu().numpy())
            print(f"beauty_edited_image={beauty_edited_image.size}")
            beauty_edited_image.save(beauty_perturb_path)
            
            
            

            protected_output_dir = "abspath/record/{}/".format(k)
            os.makedirs(protected_output_dir,exist_ok=True)
            protected_output_path = os.path.join(protected_output_dir,edit_image_name)
            
            protected_output_dir_copy = "abspath/record/{}/".format(k)
            os.makedirs(protected_output_dir_copy,exist_ok=True)
            protected_output_path_copy = os.path.join(protected_output_dir_copy,edit_image_name)
            
            protected_edit_main(perturb_path,protected_output_path_copy,instruction,seed)
            protected_edit_main(beauty_perturb_path,protected_output_path,instruction,seed)
 
    logger.info(f"END")
    end_time = time.time()
    elapsed_time = end_time-start_time
    print(f"程序总运行时间：{elapsed_time:.6f} 秒")
    print(f"程序平均运行时间：{elapsed_time/1000:.6f} 秒")

        
   
