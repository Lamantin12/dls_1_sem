#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL
from torchvision import transforms
import torch
import torch.nn as nn
import model.models as models
import io
torch.manual_seed(42)

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def load_models(filepath):
    G_model_A_B = models.CycleGANGenerator()
    F_model_B_A = models.CycleGANGenerator()

    G_model_A_B.load_state_dict(torch.load(filepath+"G_AB_199.pth", map_location=torch.device('cpu')))
    F_model_B_A.load_state_dict(torch.load(filepath+"G_BA_199.pth", map_location=torch.device('cpu')))

    G_model_A_B.eval()
    F_model_B_A.eval()
    return G_model_A_B, F_model_B_A


def display_in_out_rec(A_B_model, B_A_model, image, mode):
    """
    Displaying images like in article
    """
    fig, ax = plt.subplots(1, 3, figsize=(15,8))
    if mode == 'A2B':
        ax[0].set_title("Input $x$")
        ax[1].set_title("Output $G(x)$")
        ax[2].set_title("Reconstruction $F(G(x))$")
        out = A_B_model(image).detach().cpu()
        rec = B_A_model(out).detach().cpu()
    else:
        ax[0].set_title("Input $y$")
        ax[1].set_title("Output $F(y)$")
        ax[2].set_title("Reconstruction $G(F(y))$")
        out = B_A_model(image).detach().cpu()
        rec = A_B_model(out).detach().cpu()

    ax[0].imshow(transforms.ToPILImage()(image*0.5 + 0.5))
    ax[0].axis('off')
    ax[1].imshow(transforms.ToPILImage()(out*0.5 + 0.5))
    ax[1].axis('off')
    ax[2].imshow(transforms.ToPILImage()(rec*0.5 + 0.5))
    ax[2].axis('off')
    
    plt.show()
    return image, out, rec

def process_image_from_filepath(A_B_model, B_A_model, filepath, mode):
    image = PIL.Image.open(filepath)
    im, out, _ = display_in_out_rec(A_B_model, B_A_model, transform(image), mode)
    fig, ax = plt.subplots(1, 1)
    ax.imshow(transforms.ToPILImage()(im*0.5 + 0.5))
    ax.axis('off')
    fig, ax = plt.subplots(1, 1)
    ax.imshow(transforms.ToPILImage()(out*0.5 + 0.5))
    ax.axis('off')
    plt.savefig(fr"./images/results/{filepath.split(r'/')[-1]}")


def im_proc(A_B_model, B_A_model, image, mode):
    if mode == 'A2B':
        out = A_B_model(image).detach().cpu()
        rec = B_A_model(out).detach().cpu()
    else:
        out = B_A_model(image).detach().cpu()
        rec = A_B_model(out).detach().cpu()
    return image, out, rec

def process_image_from_image(A_B_model, B_A_model, image, mode):
    image = PIL.Image.open(io.BytesIO(image.read()))
    im, out, _ = display_in_out_rec(A_B_model, B_A_model, transform(image), mode)
    return transforms.ToPILImage()(out*0.5 + 0.5)
    fig, ax = plt.subplots(1, 1)
    ax.imshow(transforms.ToPILImage()(im*0.5 + 0.5))
    ax.axis('off')
    fig, ax = plt.subplots(1, 1)
    ax.imshow(transforms.ToPILImage()(out*0.5 + 0.5))
    ax.axis('off')
    

def main():
    model_A_B, model_B_A = load_models(r"./weights/")
    process_image_from_filepath(model_A_B, model_B_A, r"./images/pics/sunset.jpg", "B2A")
    process_image_from_filepath(model_A_B, model_B_A, r"./images/pics/sunset_spb.jpg", "B2A")
    process_image_from_filepath(model_A_B, model_B_A, r"./images/pics/dog.jpg", "B2A")
    process_image_from_filepath(model_A_B, model_B_A, r"./images/pics/1.jpg", "B2A")
    process_image_from_filepath(model_A_B, model_B_A, r"./images/pics/2.jpg", "B2A")
    process_image_from_filepath(model_A_B, model_B_A, r"./images/pics/3.jpg", "B2A")
    process_image_from_filepath(model_A_B, model_B_A, r"./images/pics/4.jpg", "B2A")
    process_image_from_filepath(model_A_B, model_B_A, r"./images/pics/5.jpg", "B2A")
    process_image_from_filepath(model_A_B, model_B_A, r"./images/pics/6.jpg", "B2A")
    process_image_from_filepath(model_A_B, model_B_A, r"./images/pics/7.jpg", "B2A")


if __name__=='__main__':
    main()
# %%
