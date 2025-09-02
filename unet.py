import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(out_ch, norm_type='bn'):
    if norm_type == 'bn':
        return nn.BatchNorm2d(out_ch)
    elif norm_type == 'in':
        return nn.InstanceNorm2d(out_ch, affine=True)
    else:
        return nn.Identity() # No normalization


class ConvBlock(nn.Module):
    """Encoder block with downsampling and sequential Conv+BN+Act layers."""
    def __init__(self, in_ch, out_ch, use_bn=True, act="relu", norm="bn"):
        super().__init__()
        layers = [
            # Initial downsampling layer
            nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=not use_bn),
            _make_norm(out_ch, norm if use_bn else 'none'),
            nn.LeakyReLU(0.2, inplace=True) if act=="lrelu" else nn.ReLU(inplace=True)
        ]

        # Add at least three more Conv+BN+Act layers as requested
        for _ in range(3):
            layers += [
                nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=not use_bn),
                _make_norm(out_ch, norm if use_bn else 'none'),
                nn.LeakyReLU(0.2, inplace=True) if act=="lrelu" else nn.ReLU(inplace=True)
            ]

        self.main = nn.Sequential(*layers)

    def forward(self, x): return self.main(x)


class ResizeConvBlock(nn.Module):
    """Upsample (nearest/bilinear) + sequential Conv+BN+ReLU layers."""
    def __init__(self, in_ch, out_ch, use_bn=True, scale=2, mode='nearest', norm="bn"):
        super().__init__()
        # build upsample with valid kwargs
        if mode in ('linear', 'bilinear', 'bicubic', 'trilinear'):
            self.upsample = nn.Upsample(scale_factor=scale, mode=mode, align_corners=False)
        else:
            self.upsample = nn.Upsample(scale_factor=scale, mode=mode)

        # Sequential Conv+BN+ReLU layers (at least one as part of the block)
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=not use_bn),
            _make_norm(out_ch, norm if use_bn else 'none'),
            nn.ReLU(inplace=True) # Decoder typically uses ReLU
        ]

        self.post = nn.Sequential(*layers)

    def forward(self, x):
        return self.post(self.upsample(x))


class Generator(nn.Module):
    """
    UNet-style generator with resize-convolutions in the decoder.
    Input size should be a multiple of 32. Outputs are in [-1, 1] (tanh).
    """
    def __init__(self, in_nc=3, out_nc=3, up_mode='nearest', use_bn=True, norm='bn'):
        super().__init__()

        # ------- Encoder -------
        self.e1 = ConvBlock(in_nc,   64, use_bn=use_bn, norm=norm)
        self.e2 = ConvBlock(64,     128, use_bn=use_bn, norm=norm)
        self.e3 = ConvBlock(128,    256, use_bn=use_bn, norm=norm)
        self.e4 = ConvBlock(256,    512, use_bn=use_bn, norm=norm)
        self.e5 = ConvBlock(512,    512, use_bn=use_bn, norm=norm)  # bottleneck

        # ------- Decoder
        self.d1 = ResizeConvBlock(512,      512, use_bn=use_bn, mode=up_mode, norm=norm)       # -> concat with e4 (512) => 1024
        self.d2 = ResizeConvBlock(512+512,  256, use_bn=use_bn, mode=up_mode, norm=norm)       # -> concat with e3 (256) => 512
        self.d3 = ResizeConvBlock(256+256,  128, use_bn=use_bn, mode=up_mode, norm=norm)       # -> concat with e2 (128) => 256
        self.d4 = ResizeConvBlock(128+128,   64, use_bn=use_bn, mode=up_mode, norm=norm)       # -> concat with e1 (64)  => 128

        # final upsample + conv to out channels (no BN)
        if up_mode in ('linear', 'bilinear', 'bicubic', 'trilinear'):
            self.final_up = nn.Upsample(scale_factor=2, mode=up_mode, align_corners=False)
        else:
            self.final_up = nn.Upsample(scale_factor=2, mode=up_mode)
        self.final_conv = nn.Conv2d(64+64, out_nc, kernel_size=3, stride=1, padding=1)
        self.out_act = nn.Tanh()

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d,)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        # Encode
        e1 = self.e1(x)   # 64
        e2 = self.e2(e1)  # 128
        e3 = self.e3(e2)  # 256
        e4 = self.e4(e3)  # 512
        e5 = self.e5(e4)  # 512 (bottleneck)

        # Decode with skip concatenations
        d1 = self.d1(e5)                 # 512
        d1 = torch.cat([d1, e4], dim=1)  # 1024

        d2 = self.d2(d1)                 # 256
        d2 = torch.cat([d2, e3], dim=1)  # 512

        d3 = self.d3(d2)                 # 128
        d3 = torch.cat([d3, e2], dim=1)  # 256

        d4 = self.d4(d3)                 # 64
        d4 = torch.cat([d4, e1], dim=1)  # 128

        out = self.final_conv(self.final_up(d4))
        return self.out_act(out)


class Discriminator(nn.Module):
    def __init__(self, in_nc=3, feat_dim=100, use_bn=True, norm='bn'):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(in_nc, 64, 4, 2, 1, bias=not use_bn),
            _make_norm(64, norm if use_bn else 'none'),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=not use_bn),
            _make_norm(128, norm if use_bn else 'none'),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=not use_bn),
            _make_norm(256, norm if use_bn else 'none'),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=not use_bn),
            _make_norm(512, norm if use_bn else 'none'),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # robust to different input sizes
        self.fc_feat = nn.Linear(512, feat_dim)
        self.fc_cls  = nn.Linear(512, 1)        # real/fake logit

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.trunk(x)              # B x 512 x H' x W'
        g = self.avgpool(h).flatten(1) # B x 512
        feat = self.fc_feat(g)         # B x feat_dim
        logit = self.fc_cls(g).squeeze(1)  # B
        return logit, feat


class SkipGANomaly(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, feat_dim=100,
                 lambda_adv=1.0, lambda_con=50.0, lambda_lat=1.0, use_bn=True, norm='bn'):
        super().__init__()
        self.G = Generator(in_nc, out_nc, use_bn=use_bn, norm=norm)
        self.D = Discriminator(in_nc, feat_dim, use_bn=use_bn, norm=norm)

        self.lambda_adv = lambda_adv
        self.lambda_con = lambda_con
        self.lambda_lat = lambda_lat

        self.bce = nn.BCEWithLogitsLoss()
        self.l1  = nn.L1Loss()
        self.mse = nn.MSELoss()

    @torch.no_grad()
    def anomaly_score(self, x, alpha=0.5):
        """
        A(x) = alpha * ||x - G(x)||_1 + (1-alpha) * ||f_D(x) - f_D(G(x))||_2
        """
        x_hat = self.G(x)
        _, f_real = self.D(x)
        _, f_fake = self.D(x_hat)

        recon = F.l1_loss(x_hat, x, reduction='none').mean(dim=(1,2,3))
        latent = F.mse_loss(f_real, f_fake, reduction='none').mean(dim=1)
        return alpha * recon + (1 - alpha) * latent, x_hat

    def d_loss(self, x):
        with torch.no_grad():
            x_hat = self.G(x)  # avoid grads to G during D step

        logit_real, _ = self.D(x)
        logit_fake, _ = self.D(x_hat)

        y_real = torch.ones_like(logit_real)   # or 0.9 for smoothing
        y_fake = torch.zeros_like(logit_fake)

        loss = self.bce(logit_real, y_real) + self.bce(logit_fake, y_fake)
        return 0.5 * loss  # averaging; optional

    def g_loss(self, x):
        x_hat = self.G(x)

        logit_fake, f_fake = self.D(x_hat)

        # get real features w/o building D's graph
        with torch.no_grad():
            _, f_real = self.D(x)

        # adversarial (make fake look real)
        y_real = torch.ones_like(logit_fake)
        L_adv = self.bce(logit_fake, y_real)

        # contextual (L1 per paper)
        L_con = self.l1(x_hat, x)

        # latent (match D features)
        L_lat = self.mse(f_real, f_fake)

        loss = self.lambda_adv * L_adv + self.lambda_con * L_con + self.lambda_lat * L_lat

        logs = {"L_adv": L_adv.detach(), "L_con": L_con.detach(), "L_lat": L_lat.detach()}
        return loss, logs, x_hat


    def forward(self, x):
        return self.G(x)

