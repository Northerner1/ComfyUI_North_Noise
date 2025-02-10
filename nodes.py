import torch
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))
import comfy.model_management
import comfy.sample
import comfy.sampler_helpers
import comfy.samplers
import comfy.utils


class Unsampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "steps": ("INT", {"default": 10, "min": 1, "max": 10000}),
                    "noise_steps": ("INT", {"default": 10, "min": 0, "max": 10000}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "latent_image": ("LATENT", ),
                    "noise_strength": ("FLOAT", {"default": 0.60, "min": 0.0, "max": 200.0, "step": 0.01}),
                    }}

    RETURN_TYPES = ("LATENT", "FLOAT",)
    RETURN_NAMES = ("LATENT", "SIGMA",)
    FUNCTION = "unsampler"
    CATEGORY = "sampling"

    def unsampler(self, model, sampler_name, steps, noise_steps, scheduler, latent_image, noise_strength):
        device = comfy.model_management.get_torch_device()
        latent = latent_image
        latent_image_data = latent["samples"]
        noise_steps = min(noise_steps, steps)

        comfy.model_management.load_model_gpu(model)
        sampler = comfy.samplers.KSampler(model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=1.0, model_options=model.model_options)
        sigmas = sampler.sigmas.flip(0)
        current_latent = latent_image_data.to(device).clone()
        torch.manual_seed(0)
        current_sigma = 0.0
        pbar = comfy.utils.ProgressBar(noise_steps)

        for step in range(noise_steps):
            sigma = sigmas[step]
            pure_noise = torch.randn_like(current_latent) * sigma * noise_strength
            current_latent = current_latent + pure_noise
            current_sigma = sigma
            pbar.update_absolute(step + 1, noise_steps)

        out = latent.copy()
        out["samples"] = current_latent.cpu()
        return (out, current_sigma.cpu().numpy(), )

NODE_CLASS_MAPPINGS = {
    "North_Unsampler": Unsampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "North_Unsampler": "Unsampler",
}
