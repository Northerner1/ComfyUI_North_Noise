import torch
import comfy.model_management
import comfy.sample
import comfy.sampler_helpers
import comfy.utils
import nodes
import folder_paths
from PIL import Image
import numpy as np

MAX_RESOLUTION = 8192

def prepare_mask(mask, shape, device="cpu"):
    mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(shape[2], shape[3]), mode="bilinear")
    mask = mask.expand((-1,shape[1],-1,-1))
    if mask.shape[0] < shape[0]:
        mask = mask.repeat((shape[0] -1) // mask.shape[0] + 1, 1, 1, 1)[:shape[0]]
    return mask.to(device)

class North_Noise:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 10, "min": 1, "max": 100}),
                "start_at_step": ("INT", {"default": 9, "min": 0, "max": 100}),
                "end_at_step": ("INT", {"default": 9, "min": 0, "max": 100}),
                "cfg": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "latent_image": ("LATENT", ),
                "noise_seed": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff}),
                "noise_strength": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                 "return_original_latent": ("BOOLEAN", {"default": False}),


            },
            "optional": {
              "vae": ("VAE",),
              "noise_mask": ("MASK", ),
            }
        }

    RETURN_TYPES = ("LATENT","IMAGE",)
    RETURN_NAMES = ("UNSAMPLED_LATENT","PREVIEW_IMAGE",)
    FUNCTION = "unsampler"

    CATEGORY = "sampling"

    def unsampler(self, model, steps, start_at_step, end_at_step, cfg, sampler_name, scheduler, positive, negative, latent_image, noise_seed, noise_strength,  noise_mask=None, vae=None, return_original_latent=False,):

        device = comfy.model_management.get_torch_device()
        latent = latent_image.copy()
        latent_image = latent["samples"]

        if start_at_step < 0: start_at_step = 0
        if end_at_step < 0: end_at_step = 0
        end_at_step = min(end_at_step, steps)
        start_at_step = min(start_at_step, end_at_step)

        torch.manual_seed(noise_seed)
        noise = torch.randn(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        noise = noise.to(device) * noise_strength

        if noise_mask is not None:
            noise_mask = prepare_mask(noise_mask, latent_image.shape, device)
            noise = noise * noise_mask

        latent_image = latent_image.to(device)


        conds0 = \
            {"positive": comfy.sampler_helpers.convert_cond(positive),
             "negative": comfy.sampler_helpers.convert_cond(negative)}

        conds = {}
        for k in conds0:
            conds[k] = list(map(lambda a: a.copy(), conds0[k]))

        models, inference_memory = comfy.sampler_helpers.get_additional_models(conds, model.model_dtype())
        comfy.model_management.load_models_gpu([model] + models, model.memory_required(noise.shape) + inference_memory)
        sampler = comfy.samplers.KSampler(model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=1.0, model_options=model.model_options)
        sigmas = sampler.sigmas[start_at_step:end_at_step+1].flip(0)
        sigmas = torch.cat([sigmas, torch.tensor([0.0001], device=device)])

        pbar = comfy.utils.ProgressBar(steps)
        def callback(step, x0, x, total_steps):
            pbar.update_absolute(step + 1, total_steps)
        
        samples = sampler.sample(noise, positive, negative, cfg=cfg, latent_image=latent_image, force_full_denoise=False, denoise_mask=noise_mask, sigmas=sigmas, start_step=0, last_step=len(sigmas)-1, callback=callback)
        comfy.sampler_helpers.cleanup_additional_models(models)
        
        if return_original_latent:
          samples = latent_image.cpu()

        if vae is not None:
           decoded_samples = vae.decode(samples)
        else:
           decoded_samples = samples *  model.model.latent_format.scale_factor

        out = latent.copy()
        out["samples"] = samples.cpu()
        return (out, decoded_samples.cpu(),)

NODE_CLASS_MAPPINGS = {
    "North_Noise": North_Noise,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "North_Noise": "North Noise",
}
