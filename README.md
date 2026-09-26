# import torch
import torch.nn as nn

# -----------------------------
# Basic building blocks
# -----------------------------
class Conv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class C3k2(nn.Module):
    def __init__(self, c1, c2, n=1, k=3, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv(c1, self.c, 1, 1)
        self.m = nn.Sequential(*[Conv(self.c, self.c, k, 1) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        y = torch.cat([m(y1) for m in self.m], dim=1)
        return y


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2 * 4, c2, 1, 1)
        self.maxp = nn.MaxPool2d(3, 1, 1, ceil_mode=False)
        self.k = k

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.maxp(x)
        y2 = self.maxp(y1)
        y3 = self.maxp(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))


# -----------------------------
# YOLOv12-style head
# -----------------------------
class Detect(nn.Module):
    def __init__(self, nc=80, anchors=None, ch=()):
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(ch)
        self.na = 3
        self.grid = [torch.zeros(1)] * self.nl
        self.stride = torch.tensor([8.0, 16.0, 32.0])
        self.anchor_grid = anchors if anchors is not None else [torch.empty(0)] * self.nl
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)

    def forward(self, x):
        z = []
        for i, xi in enumerate(x):
            out = self.m[i](xi)
            bs, _, ny, nx = out.shape
            out = out.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            z.append(out)
        return z


# -----------------------------
# Full model
# -----------------------------
class YOLOv12(nn.Module):
    def __init__(self, nc=80):
        super().__init__()
        self.stem = Conv(3, 32, 3, 2, 1)

        self.backbone1 = C3k2(32, 64, n=3)
        self.backbone2 = C3k2(64, 128, n=6)
        self.backbone3 = C3k2(128, 256, n=6)
        self.backbone4 = C3k2(256, 512, n=3)

        self.sppf = SPPF(512, 512, k=5)

        self.neck1 = C3k2(512, 256, n=3)
        self.neck2 = C3k2(256, 128, n=3)
        self.neck3 = C3k2(128, 64, n=3)

        self.head = Detect(nc=nc, ch=[64, 128, 256])

    def forward(self, x):
        x = self.stem(x)

        x1 = self.backbone1(x)
        x2 = self.backbone2(x1)
        x3 = self.backbone3(x2)
        x4 = self.backbone4(x3)
        x4 = self.sppf(x4)

        p3 = self.neck1(x4)
        p2 = self.neck2(p3)
        p1 = self.neck3(p2)

        return self.head([p1, p2, p3])


if __name__ == "__main__":
    model = YOLOv12(nc=80)
    x = torch.randn(1, 3, 640, 640)
    y = model(x)
    print(type(y), len(y))
    for i, out in enumerate(y):
        print(f"output {i}: {tuple(out.shape)}")