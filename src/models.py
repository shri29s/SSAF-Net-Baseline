import torch
import torch.nn as nn
import torch.nn.functional as F

# Attention modules
class ChannelAttention(nn.Module):
    def __init__(self, channels, k=3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)

    def forward(self, x): # x (B, C, H, W)
        avg = x.mean(dim=[2, 3])
        mx = x.amax(dim=[2, 3])
        avg_out = self.conv(avg.unsqueeze(1)).squeeze(1)
        mx_out = self.conv(mx.unsqueeze(1)).squeeze(1)
        att = torch.sigmoid(avg_out + mx_out)
        return x * att[:, :, None, None]
    
class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=k, padding=k//2, bias=False)

    def forward(self, x): # x: (B, C, H, W)
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        att = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * att
    

# Encoders
class EncoderSpa(nn.Module):
    # Spatial encoder (3x3 convs + Channel attention)
    def __init__(self, L, P):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(L, 32, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(32),
                                   nn.LeakyReLU(0.2),
                                   nn.Dropout2d(0.1))
        self.cam1 = ChannelAttention(32)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(16), 
                                   nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(16, 4, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(4), 
                                   nn.LeakyReLU(0.2))
        self.cam3 = ChannelAttention(4)
        self.conv4 = nn.Conv2d(4, P, 1, bias=False)

    def forward(self, x):
        h = self.cam1(self.conv1(x))
        h = self.conv2(h)
        h = self.cam3(self.conv3(h))
        h = self.conv4(h)
        return F.softmax(h, dim=1)

class EncoderSpe(nn.Module):
    # Spectral encoder (1x1 convs + Spatial attention)
    def __init__(self, L, P):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(L, 32, 1, bias=False),
                                   nn.BatchNorm2d(32), 
                                   nn.LeakyReLU(0.2))
        self.sam1 = SpatialAttention()
        self.conv2 = nn.Sequential(nn.Conv2d(32, 16, 1, bias=False),
                                   nn.BatchNorm2d(16),
                                   nn.LeakyReLU(0.2))
        self.sam2 = SpatialAttention()
        self.conv3 = nn.Sequential(nn.Conv2d(16, 4, 1, bias=False),
                                   nn.BatchNorm2d(4), 
                                   nn.LeakyReLU(0.2))
        self.sam3 = SpatialAttention()
        self.conv4 = nn.Conv2d(4, P, 1, bias=False)

    def forward(self, x):
        h = self.sam1(self.conv1(x))
        h = self.sam2(self.conv2(h))
        h = self.sam3(self.conv3(h))
        h = self.conv4(h)
        return F.softmax(h, dim=1)
    
# EV-NET (VAE for per-pixel endmember variability)
class EVNet(nn.Module):
    # Endmember Variability Network
    # M_n = M0 @ psi_n + dM_n (Perturbed Prototype Model)
    def __init__(self, L, P, J=4, M0=None):
        super().__init__()
        self.L = L
        self.P = P
        self.J = J

        # Inference: pixel -> (mu_z, logvar_z)
        self.enc = nn.Sequential(
            nn.Linear(L, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 32), nn.LeakyReLU(0.2),
            nn.Linear(32, 16), nn.LeakyReLU(0.2),
        )
        self.mu = nn.Linear(16, J)
        self.logv = nn.Linear(16, J)

        # Generative: z -> psi (PxP scaling), dM (Lxp perturbation)
        self.dec_psi = nn.Sequential(
            nn.Linear(J, 32), nn.LeakyReLU(0.2),
            nn.Linear(32, P * P), nn.Softplus()
        )

        self.dec_dM = nn.Sequential(
            nn.Linear(J, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, L * P), nn.Sigmoid(),
        )

        if M0 is not None:
            self.register_buffer("M0", M0) # (L, P)
        else:
            self.register_buffer("M0", torch.eye(L, P))

    def reparameterise(self, mu, logv):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logv)
    
    def forward(self, y): # y: (N, L)
        N = y.shape[0]
        h = self.enc(y)
        mu = self.mu(h)
        logv = self.logv(h)
        z = self.reparameterise(mu, logv) # (N, J)

        psi = self.dec_psi(z).view(N, self.P, self.P) # (N, P, P)
        dM = self.dec_dM(z).view(N, self.L, self.P)

        M0_exp = self.M0.unsqueeze(0).expand(N, -1, -1)
        Mn = torch.bmm(M0_exp, psi) + dM
        Mn = torch.clamp(Mn, 0, 1)
        return Mn, mu, logv
    
class SSAFNet(nn.Module):
    def __init__(self, L, P, J=4, M0=None):
        super().__init__()
        self.L = L
        self.P = P
        self.enc_spa = EncoderSpa(L, P)
        self.enc_spe = EncoderSpe(L, P)
        self.ev_net = EVNet(L, P, J, M0)

    def decode(self, A, Mn):
        """
        A:  (B, P, H, W)
        Mn: (N, L, P), N = B*H*W
        Change to: (B, L, H, W)
        """
        B, _, H, W = A.shape
        N = B * H * W
        a = A.permute(0, 2, 3, 1).reshape(N, self.P, 1) # (N, P, 1)
        y_hat = torch.bmm(Mn, a).squeeze(-1) # (N, L)
        y_hat = y_hat.view(B, H, W, self.L).permute(0, 3, 1, 2) # (B, L, H, W)
        return y_hat
    
    def forward(self, x): # x: (B, L, H, W)
        B, _, H, W = x.shape
        N = B * H * W

        A1 = self.enc_spa(x)
        y_flat = x.permute(0, 2, 3, 1).reshape(N, self.L)
        Mn, mu, logv = self.ev_net(y_flat)
        Y1 = self.decode(A1, Mn)
        A2 = self.enc_spe(Y1)
        Y2 = self.decode(A2, Mn)

        return Y1, Y2, A1, A2, Mn, mu, logv