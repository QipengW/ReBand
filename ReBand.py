import torch
import torch.nn as nn
import torch.fft as fft
    
class ComplexConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        
        self.conv_r = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_i = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.act = nn.GELU()
    
    def forward(self, x):
        B, N, L, D = x.shape
        x_flat = x.reshape(B*N, L, D).permute(0, 2, 1)  # [B*N, D, L]
        
        real, imag = x_flat.real, x_flat.imag
        # 卷积： (real * conv_r) - (imag * conv_i) + i[(real * conv_i) + (imag * conv_r)]
        out_r = self.conv_r(real) - self.conv_i(imag)
        out_i = self.conv_r(imag) + self.conv_i(real)
        
        y = torch.stack([out_r, out_i], dim=-1)
        out = F.softshrink(y, lambd=0.01)
        out = torch.view_as_complex(out).permute(0, 2, 1)
        out_channels = out.shape[2]
        out = out.view(B, N, L, out_channels)
        return out

def create_centered_order():
    center = 0
    remaining = list(range(1, 49))
    left, right = [], []
    for i, freq in enumerate(remaining):
        if i % 2 == 0:
            left.insert(0, freq)
        else:
            right.append(freq)
    return left + [center] + right

def normalize_frequency_domain(F):
    energy = torch.abs(F) ** 2
    energy_mean = energy.mean(dim=[0, 1, 3])
    
    energy_mean = energy_mean.clamp(min=1e-12)
    
    gamma = 0.5
    scale = energy_mean.pow((1.0 - gamma) / 2.0)
    return F / scale.view(1, 1, -1, 1)

class MultiHeadTimeAttention(nn.Module):
    
    def __init__(self, d_model=32, num_heads=4, dropout=0.1):

        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
    def forward(self, x):

        B, N, T, D = x.shape
        
        x_reshaped = x.reshape(B * N, T, D)
        
        Q = self.W_q(x_reshaped)  # [B*N, T, D]
        K = self.W_k(x_reshaped)  # [B*N, T, D]
        V = self.W_v(x_reshaped)  # [B*N, T, D]
        
        Q = Q.view(B * N, T, self.num_heads, self.d_k).permute(0, 2, 1, 3)  # [B*N, H, T, d_k]
        K = K.view(B * N, T, self.num_heads, self.d_k).permute(0, 2, 1, 3)  # [B*N, H, T, d_k]
        V = V.view(B * N, T, self.num_heads, self.d_k).permute(0, 2, 1, 3)  # [B*N, H, T, d_k]
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B*N, H, T, T]
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, V)  # [B*N, H, T, d_k]
        
        context = context.permute(0, 2, 1, 3).contiguous()  # [B*N, T, H, d_k]
        context = context.view(B * N, T, self.d_model)  # [B*N, T, D]
        output = self.W_o(context)  # [B*N, T, D]
        
        output = output.view(B, N, T, D)
        return output

class LongTermDecoder(nn.Module):
    def __init__(self, input_dim=32, seq_len=96, hidden_size=128,future_steps=96):
        super().__init__()
        self.L = nn.Linear(96,360) ####pred_size
        self.L1 = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        x = x.permute(0,1,3,2)
        future_values = self.L(x)
        pre = self.L1(future_values.permute(0,1,3,2)).squeeze()
        return pre

class CenteredSpectralModel(nn.Module):
    def __init__(self, input_dim=32, future_steps=96):
        super().__init__()
        self.encoder = nn.Linear(1, 32)
        self.Temp = MultiHeadTimeAttention()
        self.Spal = MultiHeadTimeAttention()
#         self.complex_mlp = ComplexMLP(input_dim=input_dim, output_dim=input_dim)
        self.complex_mlp = ComplexConv1D(32, 16, kernel_size=3)
        self.complex_mlp1 = ComplexConv1D(16, 32, kernel_size=3)   
        self.fuse = nn.Linear(3,1)
        self.decoder = LongTermDecoder(input_dim=input_dim, future_steps=future_steps)
        
        centered_order = create_centered_order()
        self.register_buffer('forward_indices', torch.tensor(centered_order, dtype=torch.long))
        
        # inverse_map[original_index] = position_in_centered_order
        inverse_map = torch.empty(49, dtype=torch.long)
        for new_pos, orig_idx in enumerate(centered_order):
            inverse_map[orig_idx] = new_pos
        self.register_buffer('inverse_map', inverse_map)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, T = x.shape
        x_feat = self.encoder(x.unsqueeze(-1))  # [B, N, T, D]
        Temp = self.Temp(x_feat)
        Spal = self.Spal(x_feat.permute(0,2,1,3)).permute(0,2,1,3)
        F_standard = fft.rfft(x_feat, dim=2)    # [B, N, 49, D]
        F_centered = F_standard[:, :, self.forward_indices, :]  # [B, N, 49, D]
        F_centered_norm = normalize_frequency_domain(F_centered)
        F_centered_enhanced = self.complex_mlp1(self.complex_mlp(F_centered_norm))     # [B, N, 49, D]
        F_standard_enhanced = torch.zeros_like(F_centered_enhanced)
        F_standard_enhanced = F_centered_enhanced[:, :, self.inverse_map, :]
        x_time = fft.irfft(F_standard_enhanced, n=T, dim=2)  # [B, N, T, D]
        ############
        final = self.fuse(torch.stack((Temp,Spal,x_time),dim=4)).squeeze()
        pred = self.decoder(final)
        return pred
