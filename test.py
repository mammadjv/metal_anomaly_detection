from torch.utils.data import DataLoader
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt

from metalDataSet import MetalPlateDataset
from unet import SkipGANomaly


# Assuming you have batched_images with shape [B, C, H, W]
def show_sample(batched_images, reconstructed, sample_idx=0, e=1):
    sample = batched_images[sample_idx]  # Shape: [C, H, W]
    sample = sample.cpu().numpy()

    reconstructed = reconstructed[sample_idx]
    reconstructed = reconstructed.detach().cpu().numpy()
    
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


#### main function!!! #######
if __name__ == "__main__":
    model_paths = ['skipganomaly_phase1_a.pth']
    for model_path in model_paths:
        model = SkipGANomaly(in_nc=3, out_nc=3, feat_dim=100,
                        lambda_adv=0.1, lambda_con=4.0, lambda_lat=0.1, use_bn=True, norm='bn')

        ### load the model from checkpoint
        checkpoint = torch.load(model_path, map_location="cuda")
        model.G.load_state_dict(checkpoint["G_state_dict"])
        model.D.load_state_dict(checkpoint["D_state_dict"])
        model = model.to('cuda')
        thresholds = checkpoint['thresholds'] ## computed thresholds on the train data
        mean = checkpoint['means'].cpu()  ## train data mean to be subtracted fro testdata
        test_ds = MetalPlateDataset("./metal_plate/test", dataset_mean=mean)

        test_dataloader = DataLoader(test_ds, batch_size=1, collate_fn=collate_and_augment_)
        model.eval()
        
        for alpha in range(1, 10):
            test_scores = []
            labels = []
            with torch.no_grad():
                for i, (batch_images, ids, mean) in enumerate(test_dataloader):
                    batch_images = batch_images.to('cuda').float()
                    labels.append(ids[0])
                    ## compute the anomaly score
                    score, x_hat = model.anomaly_score(batch_images, alpha=float(alpha/10))
                    
                    ## higher score than the threshold means anomaly
                    if score.item() > thresholds[alpha]:
                        test_scores.append(0)

                    else:
                        test_scores.append(1)

                precision = precision_score(labels, test_scores)
                recall = recall_score(labels, test_scores)
                print('precision=', precision, '; recall=', recall)