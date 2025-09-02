import random
import time
import copy

from torch.utils.data import DataLoader
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import RandomSampler
import numpy as np
import torch.optim as optim

from metalDataSet import MetalPlateDataset
from unet import SkipGANomaly


def collate_and_augment(samples):
    # samples: list of (img,)  -> make it list of img
    imgs = [sample[0] for sample in samples]
    labels = [sample[1] for sample in samples]
    means = [sample[2] for sample in samples]

    batch = torch.stack(imgs, dim=0)  # [B,C,H,W]

    # per-sample random aug (simple example)
    out = []
    for x in batch:
        random_number = random.random()
        if random_number < 0.2:
            x = TF.hflip(x)

        elif random_number < 0.7:
            x = TF.vflip(x)

        x = TF.rotate(x, angle=random.uniform(-15, 15),
                      interpolation=TF.InterpolationMode.BILINEAR)

        out.append(x)

    return torch.stack(out, dim=0), labels, means


def collate_and_augment_(samples):
    # samples: list of (img,)  -> make it list of img
    imgs = [sample[0] for sample in samples]
    labels = [sample[1] for sample in samples]
    means = [sample[2] for sample in samples]

    batch = torch.stack(imgs, dim=0)  # [B,C,H,W]

    # per-sample random aug (simple example)
    out = []
    for x in batch:
        out.append(x)

    return torch.stack(out, dim=0), labels, means



import matplotlib.pyplot as plt

# Assuming you have batched_images with shape [B, C, H, W]
def show_sample(batched_images, reconstructed, sample_idx=0, e=1):
    sample = batched_images[sample_idx]  # Shape: [C, H, W]
    sample = sample.cpu().numpy()

    reconstructed = reconstructed[sample_idx]
    reconstructed = reconstructed.detach().cpu().numpy()

    print(reconstructed.min(), reconstructed.max())
    print(sample.min(), sample.max())

    # Convert from CHW to HWC for matplotlib
    if sample.shape[0] in (1, 3):  # if channels are first
        sample = np.transpose(sample, (1, 2, 0))  # CHW -> HWC
        reconstructed = np.transpose(reconstructed, (1, 2, 0))  # CHW -> HWC
    
    # Handle grayscale
    if sample.shape[2] == 1:
        sample = sample.squeeze(2)  # Remove single channel dimension
        plt.subplot(1, 2, 1)
        plt.imshow(reconstructed)

        plt.subplot(1, 2, 2)
        plt.imshow(sample)

        plt.savefig('./fig{}.png'.format(e))

    else:
        # For RGB, ensure values are in [0,1] or [0,255]
        if sample.max() <= 1.0:
            plt.subplot(1, 2, 1)
            plt.imshow(reconstructed)

            plt.subplot(1, 2, 2)
            plt.imshow(sample)

            plt.savefig('./fig{}.png'.format(e))

        else:
            plt.subplot(1, 2, 1)
            plt.imshow(reconstructed)

            plt.subplot(1, 2, 2)
            plt.imshow(sample)

            plt.savefig('./fig{}.png'.format(e))
    
    # plt.axis('off')
    # plt.title(f'Sample {sample_idx}')
    # plt.show()



#### main function!!! #######

if __name__ == "__main__":
    train_ds = MetalPlateDataset("./metal_plate/train/good/")
    model = SkipGANomaly(in_nc=3, out_nc=3, feat_dim=100,
                        lambda_adv=0.1, lambda_con=4.0, lambda_lat=0.1, use_bn=True, norm='bn').to('cuda')


    sampler = RandomSampler(train_ds, num_samples=len(train_ds))
    dataloader = DataLoader(train_ds, batch_size=4, sampler=sampler, collate_fn=collate_and_augment)

    # Define optimizers for the generator and discriminator
    g_optimizer = optim.Adam(model.G.parameters(), lr=0.0001)
    d_optimizer = optim.Adam(model.D.parameters(), lr=0.0001)

    # Define the number of training epochs
    num_epochs = 100 # You can adjust this

    print(f"Starting training for {num_epochs} epochs...")

    samples = []
    model.train()
    best_g_loss = 10
    best_model = None
    for epoch in range(num_epochs):
        model.train() # Set the model to training mode
        epoch_start_time = time.time()

        running_g_loss = 0.0
        running_d_loss = 0.0
        running_l_adv = 0.0
        running_l_con = 0.0
        running_l_lat = 0.0

        for i, (batch_images, ids, means) in enumerate(dataloader):
            batch_images = batch_images.to('cuda').float()
            # --- Train Discriminator ---
            d_optimizer.zero_grad()
            d_loss = model.d_loss(batch_images)
            d_loss.backward()
            d_optimizer.step()
            running_d_loss += d_loss.item()

            # --- Train Generator ---
            g_optimizer.zero_grad()
            g_loss, logs, reconstructed = model.g_loss(batch_images)

            print(f"\r Epoch: {epoch},  Batch number: {i}, gloss: {g_loss.item()}", end="", flush=True)
            g_loss.backward()
            g_optimizer.step()
            running_g_loss += g_loss.item()
            running_l_adv += logs['L_adv'].item()
            running_l_con += logs['L_con'].item()
            running_l_lat += logs['L_lat'].item()

            if g_loss.item() < best_g_loss:
                best_g_loss = g_loss.item()
                best_model = copy.deepcopy(model)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        freq = 1
        avg_d_loss = running_d_loss / len(dataloader)
        avg_g_loss = running_g_loss / len(dataloader)
        avg_l_adv = running_l_adv / len(dataloader)
        avg_l_con = running_l_con / len(dataloader)
        avg_l_lat = running_l_lat / len(dataloader)

        print(f"Epoch [{epoch+1}/{num_epochs}] completed in {epoch_duration} seconds. "
            f"Avg D_Loss: {avg_d_loss}, Avg G_Loss: {avg_g_loss}, "
            f"Avg L_adv: {avg_l_adv}, Avg L_con: {avg_l_con}, Avg L_lat: {avg_l_lat}")


    print("Training finished.")
    model = best_model

    sampler = RandomSampler(train_ds, num_samples=len(train_ds))
    dataloader = DataLoader(train_ds, batch_size=4, collate_fn=collate_and_augment_)

    model.eval()
    thresholds = {}
    for alpha in range(1, 10):
        print(alpha)
        scores = []
        with torch.no_grad():
            for i, (batch_images, ids, means) in enumerate(dataloader):
                batch_images = batch_images.to('cuda').float()
                anomaly_score, x_hat = model.anomaly_score(batch_images, alpha=float(alpha/10))

                for i in range(4):
                    try:
                        scores.append(anomaly_score[i].item())

                    except:
                        continue

        threshold = np.percentile(scores, 95)
        thresholds[alpha] = threshold

    checkpoint = {
        "G_state_dict": model.G.state_dict(),
        "D_state_dict": model.D.state_dict(),
        'd_optimizer': d_optimizer.state_dict(),
        'g_optimizer': g_optimizer.state_dict(),
        'thresholds': thresholds,
        'means': means[0],
    }

    torch.save(checkpoint, "skipganomaly_phase1_a.pth")


    # print("Training finished.")
    # model = best_model

    # sampler = RandomSampler(train_ds, num_samples=len(train_ds))
    # dataloader = DataLoader(train_ds, batch_size=4, sampler=sampler, collate_fn=collate_and_augment)

    # model.eval()
    # thresholds = {}
    # for alpha in range(1, 10):
    #     print(alpha)
    #     scores = []
    #     with torch.no_grad():
    #         for i, (batch_images, ids, means) in enumerate(dataloader):
    #             batch_images = batch_images.to('cuda').float()
    #             anomaly_score, x_hat = model.anomaly_score(batch_images, alpha=float(alpha/10))

    #             for i in range(4):
    #                 try:
    #                     scores.append(anomaly_score[i].item())

    #                 except:
    #                     continue

    #     threshold = np.percentile(scores, 95)
    #     thresholds[alpha] = threshold

    # checkpoint = {
    #     "G_state_dict": model.G.state_dict(),
    #     "D_state_dict": model.D.state_dict(),
    #     'd_optimizer': d_optimizer.state_dict(),
    #     'g_optimizer': g_optimizer.state_dict(),
    #     'thresholds': thresholds,
    #     'means': means[0],
    # }

    # torch.save(checkpoint, "skipganomaly_phase1_b.pth")