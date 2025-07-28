## Imaging  library
from PIL import Image
from torchvision import transforms as tfms
## Basic libraries
import numpy as np
import matplotlib.pyplot as plt
## Loading a VAE model
from diffusers import AutoencoderKL
import torch

sd_path = r'abspath/git_repo/instruct-pix2pix-main/checkpoints'
vae = AutoencoderKL.from_pretrained(sd_path,subfolder="vae",
                                   local_files_only=True,
                                   torch_dtype=torch.float16).to("cuda")


def pil_to_latents(image,vae):
    init_image = tfms.ToTensor()(image).unsqueeze(0) * 2.0 - 1.0
    init_image = init_image.to(device="cuda",dtype=torch.float16)
    init_latent_dist = vae.encode(init_image).latent_dist.sample() * 0.18215
    return init_latent_dist

print(f"VAE={vae}")
#img_path = r'/home/VAEs/Scarlet-Macaw-2.jpg'
img_path = r'abspath/Data/Instruction-Dataset/0000657/1382322226_0.jpg'
img = Image.open(img_path).convert("RGB").resize((512,512))
print(f"Dimension of this image : {np.array(img).shape}")  
# plt.imshow(img)
# plt.show()

latent_img = pil_to_latents(img,vae)
print(f"Dimension of this latent representation: {latent_img.shape}")

# visual
fig,axs = plt.subplots(1,4,figsize=(16,4))
for c in range(4):
   axs[c].imshow(latent_img[0][c].detach().cpu(),cmap='Greys')
plt.show()
plt.savefig("abspath/git_repo/instruct-pix2pix-main/test_content/visual.jpg")
plt.close()

def latent_to_pil(latents,vae):
    latents = (1/0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0,1)
    image = image.detach().cpu().permute(0,2,3,1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [ Image.fromarray(image) for image in images ]
    return pil_images

# decode
decoded_img = latent_to_pil(latent_img,vae)
plt.imshow(decoded_img[0])
plt.axis("off")
plt.show()
plt.savefig("abspath/git_repo/instruct-pix2pix-main/test_content/decode_img.jpg")
plt.close()

