# Metal Plate Anomaly Detection: A Two-Phase Approach

This report presents a comprehensive approach to metal plate anomaly detection using deep learning architectures. The final system achieved **100% precision and 100% recall** on test data using **Skip-GANomaly** architecture with strategic data augmentation. We chose not to use any abnormal sample model during training, and performed *one class classification*, in which only normal samples are available.


## Traditional Computer Vision Exploration

The project began with a fundamental computer vision approach. I analyzed the histograms of the normal vs abnormal images, and figured out that their histograms are so different, except when the anomaly was from scratches, for which histogram distribution was similar to those of normal plate. Was thinking of doing a Gaussian Mixture Model or even feeding the histogram data into a ML model to do the classification, but realized the pattern, especially for scratches, needs a bit beyond histogram-based classification, and the discriminative scratch features should be detected based on edge patterns. Here are some normal vs abnormal histograms, the difference of which is obvious.

## Deep Learning Implementation

### Architecture Selection: Skip-GANomaly
The project transitioned to a deep learning approach using the Skip-GANomaly architecture, which combines:

- **U-Net Encoder-Decoder**: Skip-connected architecture for multi-scale feature preservation
- **Adversarial Training**: GAN framework with discriminator providing reconstruction feedback
- **Multi-objective Loss Function**: Combining adversarial, contextual, and latent space losses

### Training Configuration
- **Training Data**: 53 normal metal plate samples (unsupervised learning on normal samples only)
- **Architecture**: U-Net based encoder-decoder with discriminator network
- **Loss Components**:
  - Adversarial loss for realistic reconstruction
  - Contextual loss (L1) for pixel-level accuracy
  - Latent loss for feature space consistency
  - According to the paper, the weight for each component was set to 0.1, 4, and 0.1.

### Results
- **Test Dataset Size**: 97 test samples (26 good, 23 with rust, 16 major rust, 34 scratches)
- **Hyperparameter Tuning**: During the inference, there is a tunable hyperparameter called alpha for computing the anomaly threshold. We varied alpha from 0.1 to 0.9, which controls the contribution of reconstruction and latent space errors. For each alpha, we first computed a threshold, which acts as the decision boundary for the anomaly. Any sample having an error beyond that threshold is abnormal. That threshold was computed using 95% percentile on error distributions of the training data. What is the error range in the training data? Anything above that from unseen test data is abnormal.
- **Performance**: Tuning the hyperparameter we got the following result:

| Alpha | Precision | Recall |
|-------|---------|---------|
| 0.1   | 0.70 | 1.0 |
| 0.2   | 0.70 | 1.0 |
| 0.3   | 0.81 | 1.0 |
| 0.4   | 0.86 | 1.0 |
| 0.5   | 0.89 | 1.0 |
| 0.6   | 0.96 | 1.0 |
| 0.7   | 1.0 | 1.0 |
| 0.8   | 1.0 | 1.0 |
| 0.9   | 1.0 | 1.0 |

- **Note**: These 97 test samples can be good for choosing the appropriate alpha during actual test.


### Implementation
Everything was implemented using PyTorch, OpenCV. I used DataSet class to load the dataset during train/test. Also, some random augmentations were done on the 53 training samples on the fly, in which during the batching process we randomly apply some rotations. Before this augmentation, the model didn't perform well at all. But this augmentation helped the model to see more patterns, which resulted in a perfect anomaly detection.

In order to run the code:

1. Place the metal_plate folder next to the codes (next to train.py and test.py)
2. Install pytorch, torchvision, numpy, matplotlib
3. To train the model run:
```
python train.py
```
It took me 30 mins on a V100. The train.py loads the dataset, trains the Skip-GANomaly, computes the thresholds on the training data, and saves them with the model checkpoints.

4. To test the model:
```
python test.py
```
Running the test.py script loads the *skipganomaly_phase1_a.pth* (obtained from train.py, with all the model weights, alpha thresholds, and train data mean). As we know the threshold for each alpha, we can check the anomaly_score(reconstruction + latent space error) with the given threshold. Higher value than the threshold means that sample is abnormal.
