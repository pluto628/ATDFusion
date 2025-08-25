import torch
import torch.nn as nn
import torch.nn.functional as F

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_se=True):
        super().__init__()
        self.use_se = use_se
        self.same_shape = in_channels == out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        if use_se:
            self.se = SELayer(out_channels)

        self.shortcut = (
            nn.Identity() if self.same_shape else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        )

        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.conv(x)
        if self.use_se:
            out = self.se(out)
        out += self.shortcut(x)
        return self.relu(out)


def upsample_block(in_c, out_c):
    return nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(0.2, inplace=True)
    )


def make_detail_block(in_c1, in_c2, mid_c, out_c):
    return nn.Sequential(
        ResBlock(in_c1, mid_c),
        ResBlock(mid_c, out_c)
    )

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.G_generator = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 8x8
        self.res_block_8 = ResBlock(1024, 1024)

        self.upsample_8_16 = upsample_block(1024, 512)
        self.res_block_16 = ResBlock(1536, 512)

        self.upsample_16_32 = upsample_block(512, 256)
        self.res_block_32 = ResBlock(1280, 256)

        self.upsample_32_64 = upsample_block(256, 128)
        self.res_block_64 = ResBlock(1152, 256)

        self.res_block_detail64 = make_detail_block(256, 256, 256, 128)
        self.res_block_detail128 = make_detail_block(128, 128, 128, 64)
        self.res_block_detail256 = make_detail_block(64, 64, 64, 32)

        self.upsample_64_128 = upsample_block(384, 128)
        self.upsample_128_256 = upsample_block(192, 64)

        self.res_block_256 = ResBlock(96, 32)

        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 1, 3, padding=1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, data_MRI, features_modal1, features_modal2, detail1, detail2):
        G = self.G_generator(detail1[0])  # (B, 1, 256, 256)
        # 8x8
        G_8 = F.interpolate(G, size=(8, 8), mode='bilinear', align_corners=False)
        F_ct_8 = G_8 * features_modal1[3]
        F_mri_8 = (1 - G_8) * features_modal2[3]
        x = torch.cat([F_ct_8 + F_mri_8], dim=1)  #1024
        x = self.res_block_8(x)

        # 16x16
        x = self.upsample_8_16(x)
        G_16 = F.interpolate(G, size=(16, 16), mode='bilinear', align_corners=False)
        F_ct_16 = G_16 * features_modal1[2]
        F_mri_16 = (1 - G_16) * features_modal2[2]
        x = torch.cat([x, F_ct_16 + F_mri_16], dim=1) # 512+1024
        x = self.res_block_16(x) # 256

        # 32x32
        x = self.upsample_16_32(x)
        G_32 = F.interpolate(G, size=(32, 32), mode='bilinear', align_corners=False)
        F_ct_32 = G_32 * features_modal1[1]
        F_mri_32 = (1 - G_32) * features_modal2[1]
        x = torch.cat([x, F_ct_32 + F_mri_32], dim=1) # 256+1024
        x = self.res_block_32(x)

        # 64x64
        x = self.upsample_32_64(x)
        G_64 = F.interpolate(G, size=(64, 64), mode='bilinear', align_corners=False)
        F_ct_64 = G_64 * features_modal1[0]
        F_mri_64 = (1 - G_64) * features_modal2[0]
        x = torch.cat([x, F_ct_64 + F_mri_64], dim=1) # 128+1024->1152
        x = self.res_block_64(x) # 256

        G_64 = F.interpolate(G, size=(64, 64), mode='bilinear', align_corners=False)
        D_ct_64 = G_64 * detail1[2]
        D_mri_64 = (1 - G_64) * detail2[2]
        x1 = torch.cat([D_ct_64 + D_mri_64], dim=1)  #256
        x1 = self.res_block_detail64(x1) #128

        G_128 = F.interpolate(G, size=(128, 128), mode='bilinear', align_corners=False)
        D_ct_128 = G_128 * detail1[1]
        D_mri_128 = (1 - G_128) * detail2[1]
        x2 = torch.cat([D_ct_128 + D_mri_128], dim=1) #128
        x2 = self.res_block_detail128(x2) #64

        G_256 = F.interpolate(G, size=(256, 256), mode='bilinear', align_corners=False)
        D_ct_256 = G_256 * detail1[0]
        D_mri_256 = (1 - G_256) * detail2[0]
        x3 = torch.cat([D_ct_256 + D_mri_256], dim=1) #64
        x3 = self.res_block_detail256(x3) #32

        x = torch.cat([x, x1], dim=1) #256+128=384
        x = self.upsample_64_128(x) #128

        x = torch.cat([x, x2], dim=1) #128+64=192
        x = self.upsample_128_256(x) #64

        x = torch.cat([x, x3], dim=1) #64+32=96
        x = self.res_block_256(x) #32

        x = self.final_conv(x)
        x = x + (1 - G_256) * data_MRI
        return self.sigmoid(x), G


def test_decoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    features_1 = [
        torch.randn(1, 1024, 64, 64, device=device),
        torch.randn(1, 1024, 32, 32, device=device),
        torch.randn(1, 1024, 16, 16, device=device),
        torch.randn(1, 1024, 8, 8, device=device),
    ]

    features_2 = [
        torch.randn(1, 1024, 64, 64, device=device),
        torch.randn(1, 1024, 32, 32, device=device),
        torch.randn(1, 1024, 16, 16, device=device),
        torch.randn(1, 1024, 8, 8, device=device),
    ]
    features_3 = [
        torch.randn(1, 64, 256, 256, device=device),
        torch.randn(1, 128, 128, 128, device=device),
        torch.randn(1, 256, 64, 64, device=device),
    ]

    features_4 = [
        torch.randn(1, 64, 256, 256, device=device),
        torch.randn(1, 128, 128, 128, device=device),
        torch.randn(1, 256, 64, 64, device=device),
    ]

    data_MRI = torch.randn(1, 1, 256, 256, device=device)
    model = Decoder().to(device)

    output, _ = model(data_MRI, features_1, features_2, features_3, features_4)

    output = output.cpu()
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

    return output

if __name__ == "__main__":
    test_decoder()