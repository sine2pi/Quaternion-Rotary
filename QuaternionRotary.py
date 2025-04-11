
import math
import torch
import torch.nn as nn
from einops import rearrange

class RotaryEmbedding(nn.Module):

    def __init__( self, dim, theta = 10000, num_freqs = 1, learned_freq = True, theta_rescale_factor = 1., 
                 use_quaternion = False, rot_scale = 1.0, rot_count = 1, use_projection = False, proj_dim = 3, 
                 proj_scale = 0.1, ): 
        super().__init__()
        theta *= theta_rescale_factor ** (dim / (dim - 2))
        self.freqs = nn.Parameter(torch.arange(0, num_freqs) * (2 * math.pi / theta), requires_grad=learned_freq)
        self.register_buffer('dummy', torch.tensor(0), persistent=False)
        self.use_quaternion = use_quaternion
        self.use_projection = use_projection
        self.proj_dim = proj_dim
        self.proj_scale = proj_scale
        
        if use_quaternion:
            self.dparam = nn.Parameter(torch.zeros(1))
            self.rscale = rot_scale
            self.rot = rot_count
            self.tscale = 1.0
            pairs = []
            for i in range(0, dim-1, 2):
                pairs.append(torch.tensor([i, i+1]))
            self.pairs = nn.Parameter(torch.stack(pairs), requires_grad=False)
            self.thetas = nn.Parameter(torch.ones(len(self.pairs)) * (2 * math.pi / len(self.pairs)), 
                                      requires_grad=False)
            if use_projection:
                self.proj_down = None
                self.proj_up = None

    @property
    def device(self):
        return self.dummy.device

    def q_rotation(self, x, theta, u, v=None):
        eps = 1e-8
        u_norm = torch.norm(u, p=2)
        u = u / (u_norm + eps)
        w = torch.cos(theta / 2)
        vec = torch.sin(theta / 2) * u
        x_shape = x.shape
        x = x.reshape(-1, 3)
        uv_cross = torch.cross(u.unsqueeze(0), x)
        uuv_cross = torch.cross(u.unsqueeze(0), uv_cross)
        x_rot = x + torch.clamp(2 * (w * uv_cross + uuv_cross), min=-10.0, max=10.0)
        return x_rot.reshape(*x_shape)

    def rotation_matrix(self, dims, i, j, theta):
        G = torch.eye(dims, device=theta.device)
        c, s = torch.cos(theta), torch.sin(theta)
        G[i, i], G[j, j] = c, c
        G[i, j], G[j, i] = -s, s
        if dims == 3:
            u = torch.eye(dims, device=theta.device)[i]
            v = torch.eye(dims, device=theta.device)[j]
            if theta < 0: 
                Q = self.q_rotation(torch.eye(dims, device=theta.device), theta=abs(theta), u=u, v=v)
            else:
                Q = self.q_rotation(torch.eye(dims, device=theta.device), theta=theta, u=u, v=v)
            G = (G + Q) / 2
        return G

    def rotations(self, x):
        direction = torch.sigmoid(self.dparam) * 2 - 1
        rotate = int(round(self.rscale * self.rot))
        head_dim = x.shape[-1]
        for k in range(min(rotate, len(self.pairs))):
            i, j = self.pairs[k].long()
            if i >= head_dim or j >= head_dim:
                continue
            theta = direction * self.thetas[k] * self.tscale
            G = self.rotation_matrix(dims=head_dim, i=i.item(), j=j.item(), theta=theta)
            x_shape = x.shape
            x = x.reshape(-1, head_dim)
            x = x @ G
            x = x.reshape(*x_shape)
        return x

    def _ensure_projection(self, x):
        if self.proj_down is None or self.proj_down.weight.device != x.device:
            head_dim = x.shape[-1] 
            self.proj_down = Linear(head_dim, self.proj_dim, bias=False).to(x.device)
            self.proj_up = Linear(self.proj_dim, head_dim, bias=False).to(x.device)
            with torch.no_grad():
                nn.init.orthogonal_(self.proj_down.weight, gain=self.proj_scale)
                nn.init.orthogonal_(self.proj_up.weight, gain=self.proj_scale)
                U, S, V = torch.svd(self.proj_down.weight)
                S_inv = 1.0 / (S + 1e-6) 
                S_inv = torch.clamp(S_inv, max=10.0)
                pseudo_inv = V @ torch.diag(S_inv) @ U.t()
                self.proj_up.weight.copy_(pseudo_inv * self.proj_scale)

    def project_and_rotate(self, x):
        orig_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        with torch.no_grad():
            x_norm = torch.norm(x_flat, dim=1, keepdim=True)
            if torch.max(x_norm) > 1e3:
                x_flat = x_flat * (1e3 / torch.max(x_norm))
        if x.shape[-1] > 3 and self.use_projection:
            self._ensure_projection(x)
            x_3d = self.proj_down(x_flat)
            if torch.isnan(x_3d).any():
                return x.reshape(*orig_shape)
            x_3d_rot = self.rotations(x_3d)
            if torch.isnan(x_3d_rot).any():
                x_rot = self.proj_up(x_3d)
            else:
                x_rot = self.proj_up(x_3d_rot)
            alpha = 0.9
            x_rot = alpha * x_rot + (1-alpha) * x_flat
            if torch.isnan(x_rot).any():
                return x.reshape(*orig_shape)
        else:
            x_rot = self.rotations(x_flat)
        return x_rot.reshape(*orig_shape)

    def apply_rotary(self, freqs, t, start_index=0, scale=1., seq_dim=-2, freqs_seq_dim=None):
        dtype = t.dtype
        
        def _exists(val):
            return val is not None
        
        def _slice_at_dim(tensor, dim_slice, dim):
            dim += (tensor.ndim if dim < 0 else 0)
            colons = [slice(None)] * tensor.ndim
            colons[dim] = dim_slice
            return tensor[tuple(colons)]
        
        def _rotate_half(x):
            x = rearrange(x, '... (d r) -> ... d r', r=2)
            x1, x2 = x.unbind(dim=-1)
            x = torch.stack((-x2, x1), dim=-1)
            return rearrange(x, '... d r -> ... (d r)')
        
        if not _exists(freqs_seq_dim):
            if freqs.ndim == 2 or t.ndim == 3:
                freqs_seq_dim = 0
                
        if t.ndim == 3 or _exists(freqs_seq_dim):
            ctx = t.shape[seq_dim]
            freqs = _slice_at_dim(freqs, slice(-ctx, None), dim=freqs_seq_dim)
        rot_dim = freqs.shape[-1]
        end_index = start_index + rot_dim
        
        assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
        t_left = t[..., :start_index]
        t_middle = t[..., start_index:end_index]
        t_right = t[..., end_index:]
        t_transformed = (t_middle * freqs.cos() * scale) + (_rotate_half(t_middle) * freqs.sin() * scale)
        out = torch.cat((t_left, t_transformed, t_right), dim=-1)
        return out.type(dtype)

    def rotate_(self, t, seq_dim=None, offset=0, scale=None):
        if self.use_quaternion:
            if self.use_projection and t.shape[-1] > 3:
                return self.project_and_rotate(t)
            else:
                return self.rotations(t)
        else:
            ctx = t.shape[2]
            device, dtype = t.device, t.dtype
            seq = torch.arange(ctx, device=device, dtype=dtype) + offset
            freqs = self.forward(seq)
            scale = scale if scale is not None else 1.0
            return self.apply_rotary(freqs, t, scale=scale, seq_dim=2)
    
    def learned_rotations(self, rotations, t, start_index = 0, freq_ranges = None):
        if exists(freq_ranges):
            rotations = einsum('..., f -> ... f', rotations, freq_ranges)
            rotations = rearrange(rotations, '... r f -> ... (r f)')
        rotations = repeat(rotations, '... n -> ... (n r)', r = 2)
        return self.apply_rotary(rotations, t, start_index = start_index)

    def forward(self, t):
        freqs = self.freqs
        freqs = torch.einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = torch.repeat_interleave(freqs, 2, dim=-1)
        return freqs
    
class CompactRotation:
    def __init__(self, dim, rot_pairs=None, rot_scale=1.0, rot_count=1):
        self.scale = rot_scale
        self.count = rot_count

        if rot_pairs is None:
            pairs = []
            for i in range(0, dim-1, 2):
                pairs.append((i, i+1))
            self.pairs = pairs
        else:
            self.pairs = rot_pairs
            
        self.thetas = [2 * math.pi / len(self.pairs)] * len(self.pairs)
        self.direction = 1.0  # Fixed direction instead of learned
        
    def __call__(self, x):
        """Apply rotations to input tensor"""
        return self.rotate(x)
        
    def rotate(self, x):
        rotate_steps = min(int(round(self.scale * self.count)), len(self.pairs))
        head_dim = x.shape[-1]
        
        for k in range(rotate_steps):
            i, j = self.pairs[k]
            if i >= head_dim or j >= head_dim:
                continue
                
            # Create rotation matrix
            theta = self.direction * self.thetas[k]
            device = x.device
            G = torch.eye(head_dim, device=device)
            c, s = torch.cos(theta), torch.sin(theta)
            G[i, i], G[j, j] = c, c
            G[i, j], G[j, i] = -s, s
            
            # Apply rotation
            x_shape = x.shape
            x = x.reshape(-1, head_dim)
            x = x @ G
            x = x.reshape(*x_shape)
            
        return x
    
    @staticmethod
    def q_rotate(x, theta, axis_idx, dims=3):
        """Quaternion rotation in 3D space (simplified)"""
        device = x.device
        u = torch.zeros(dims, device=device)
        u[axis_idx] = 1.0
        
        x_shape = x.shape
        x = x.reshape(-1, dims)
        
        # Quaternion rotation formula
        uv_cross = torch.cross(u.unsqueeze(0), x)
        uuv_cross = torch.cross(u.unsqueeze(0), uv_cross)
        w = torch.cos(theta / 2)
        x_rot = x + 2 * (w * uv_cross + uuv_cross)
        
        return x_rot.reshape(*x_shape)
    

class LSrotations: #learned smaller compact verion

    def __init__(self, dim, rot_pairs=None, rot_scale=1.0, rot_count=1, learned_freq=False):
        self.dim = dim
        self.scale = rot_scale
        self.count = rot_count
        
        # Set up rotation pairs and angles
        if rot_pairs is None:
            pairs = []
            for i in range(0, dim-1, 2):
                pairs.append((i, i+1))
            self.pairs = pairs
        else:
            self.pairs = rot_pairs
            
        self.thetas = [2 * math.pi / len(self.pairs)] * len(self.pairs)
        self.direction = 1.0
        
        # For learned rotations
        if learned_freq:
            self.freqs = nn.Parameter(torch.ones(dim // 2))
        
    def __call__(self, x):
        return self.rotate(x)
        
    def rotate(self, x):
        rotate_steps = min(int(round(self.scale * self.count)), len(self.pairs))
        head_dim = x.shape[-1]
        
        for k in range(rotate_steps):
            i, j = self.pairs[k]
            if i >= head_dim or j >= head_dim:
                continue
                
            # Create rotation matrix
            theta_value = self.direction * self.thetas[k]
            device = x.device
            G = torch.eye(head_dim, device=device)
            # Convert theta to a tensor before using cos/sin
            theta = torch.tensor(theta_value, device=device)
            c, s = torch.cos(theta), torch.sin(theta)
            G[i, i], G[j, j] = c, c
            G[i, j], G[j, i] = -s, s
            
            # Apply rotation
            x_shape = x.shape
            x = x.reshape(-1, head_dim)
            x = x @ G
            x = x.reshape(*x_shape)
            
        return x
    
    def _exists(self, val):
        return val is not None
    
    def _rotate_half(self, x):
        x = x.view(*x.shape[:-1], -1, 2)
        x1, x2 = x[..., 0], x[..., 1]
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary(self, freqs, t, start_index=0, scale=1.0):
        """Apply rotary embeddings to input tensor"""
        dtype = t.dtype
        rot_dim = freqs.shape[-1]
        end_index = start_index + rot_dim
        
        assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not sufficient for rotation positions {rot_dim}'
        
        t_left = t[..., :start_index]
        t_middle = t[..., start_index:end_index]
        t_right = t[..., end_index:]
        
        t_transformed = (t_middle * freqs.cos() * scale) + (self._rotate_half(t_middle) * freqs.sin() * scale)
        return torch.cat((t_left, t_transformed, t_right), dim=-1).type(dtype)
    
    def learned_rotations(self, rotations, t, start_index=0, freq_ranges=None):
        """Apply learned rotations to the input tensor"""
        if self._exists(freq_ranges):
            # Apply frequency ranges if provided
            rotations = torch.einsum('..., f -> ... f', rotations, freq_ranges)
            rotations = rotations.view(*rotations.shape[:-2], -1)
            
        # Double each rotation for sin/cos pairs
        shape = list(rotations.shape)
        shape[-1] *= 2
        expanded = torch.zeros(shape, device=rotations.device)
        expanded[..., ::2] = rotations
        expanded[..., 1::2] = rotations
        
        return self.apply_rotary(expanded, t, start_index=start_index)
    
    @staticmethod
    def sinusoids(length, channels, max_timescale=10000):
        """Generate sinusoidal position embeddings"""
        assert channels % 2 == 0
        log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
        scaled_time = torch.arange(length).unsqueeze(1) * inv_timescales.unsqueeze(0)
        return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
    
# # Basic rotation
# rotations = CompactRotations(dim=256)
# x_rotated = rotations(x)

# # Or with learned rotations
# rotations = CompactRotations(dim=256, learned_freq=True)
# pos_freqs = rotations.sinusoids(seq_length, channels=128)
# x_rotated = rotations.learned_rotations(pos_freqs, x)

# class QuaternionRotary(nn.Module):

#     def __init__(
#         self,
#         dim,
#         theta = 10000,
#         num_freqs = 1,
#         learned_freq = False,
#         theta_rescale_factor = 1.0,
#         use_quaternion = False,
#         rot_scale = 1.0,
#         rot_count = 1,
#         use_projection = True,
#         proj_dim = 3,
#         proj_scale = 0.1,
#     ):
#         super().__init__()
        
#         theta *= theta_rescale_factor ** (dim / (dim - 2))
#         self.freqs = nn.Parameter(torch.arange(0, num_freqs) * (2 * math.pi / theta), requires_grad=learned_freq)
        
#         self.register_buffer('dummy', torch.tensor(0), persistent=False)
        
#         self.use_quaternion = use_quaternion
#         self.use_projection = use_projection
#         self.proj_dim = proj_dim
#         self.proj_scale = proj_scale
        
#         if use_quaternion:
#             self.dparam = nn.Parameter(torch.zeros(1))
#             self.rscale = rot_scale
#             self.rot = rot_count
#             self.tscale = 1.0
            
#             pairs = []
#             for i in range(0, dim-1, 2):
#                 pairs.append(torch.tensor([i, i+1]))
#             self.pairs = nn.Parameter(torch.stack(pairs), requires_grad=False)
            
#             self.thetas = nn.Parameter(torch.ones(len(self.pairs)) * (2 * math.pi / len(self.pairs)), 
#                                       requires_grad=False)
            
#             if use_projection:
#                 self.proj_down = None
#                 self.proj_up = None

#     @property
#     def device(self):
#         return self.dummy.device

#     def q_rotation(self, x, theta, u, v=None):
#         """Quaternion rotation for 3D vectors."""
#         eps = 1e-6
#         u_norm = torch.norm(u, p=2)
#         u = u / (u_norm + eps)
        
#         w = torch.cos(theta / 2)
#         vec = torch.sin(theta / 2) * u
        
#         x_shape = x.shape
#         x = x.reshape(-1, 3)
        
#         uv_cross = torch.cross(u.unsqueeze(0), x)
#         uuv_cross = torch.cross(u.unsqueeze(0), uv_cross)
        
#         x_rot = x + torch.clamp(2 * (w * uv_cross + uuv_cross), min=-10.0, max=10.0)
        
#         return x_rot.reshape(*x_shape)

#     def rotation_matrix(self, dims, i, j, theta):
#         """Create a rotation matrix for dimensions i,j with angle theta."""
#         G = torch.eye(dims, device=theta.device)
#         c, s = torch.cos(theta), torch.sin(theta)
#         G[i, i], G[j, j] = c, c
#         G[i, j], G[j, i] = -s, s
        
#         if dims == 3:
#             u = torch.eye(dims, device=theta.device)[i]
#             v = torch.eye(dims, device=theta.device)[j]
#             if theta < 0: 
#                 Q = self.q_rotation(torch.eye(dims, device=theta.device), theta=abs(theta), u=u, v=v)
#             else:
#                 Q = self.q_rotation(torch.eye(dims, device=theta.device), theta=theta, u=u, v=v)
#             G = (G + Q) / 2
            
#         return G

#     def rotations(self, x):

#         direction = torch.sigmoid(self.dparam) * 2 - 1
#         rotate = int(round(self.rscale * self.rot))
        
#         head_dim = x.shape[-1]
        
#         for k in range(min(rotate, len(self.pairs))):
#             i, j = self.pairs[k].long()
#             if i >= head_dim or j >= head_dim:
#                 continue
            
#             theta = direction * self.thetas[k] * self.tscale
#             G = self.rotation_matrix(dims=head_dim, i=i.item(), j=j.item(), theta=theta)
            
#             x_shape = x.shape
#             x = x.reshape(-1, head_dim)
#             x = x @ G
#             x = x.reshape(*x_shape)
        
#         return x

#     def _ensure_projection_layers(self, x):
#         if self.proj_down is None or self.proj_down.weight.device != x.device:
#             head_dim = x.shape[-1] 
            
#             self.proj_down = nn.Linear(head_dim, self.proj_dim, bias=False).to(x.device)
#             self.proj_up = nn.Linear(self.proj_dim, head_dim, bias=False).to(x.device)
            
#             with torch.no_grad():
#                 nn.init.orthogonal_(self.proj_down.weight, gain=self.proj_scale)
#                 nn.init.orthogonal_(self.proj_up.weight, gain=self.proj_scale)
                
#                 U, S, V = torch.svd(self.proj_down.weight)
#                 S_inv = 1.0 / (S + 1e-6) 
#                 S_inv = torch.clamp(S_inv, max=10.0)
#                 pseudo_inv = V @ torch.diag(S_inv) @ U.t()
#                 self.proj_up.weight.copy_(pseudo_inv * self.proj_scale)

#     def project_and_rotate(self, x):
#         orig_shape = x.shape
#         x_flat = x.reshape(-1, x.shape[-1])
        
#         with torch.no_grad():
#             x_norm = torch.norm(x_flat, dim=1, keepdim=True)
#             if torch.max(x_norm) > 1e3:
#                 x_flat = x_flat * (1e3 / torch.max(x_norm))
        
#         if x.shape[-1] > 3 and self.use_projection:
#             self._ensure_projection_layers(x)
            
#             x_3d = self.proj_down(x_flat)
            
#             if torch.isnan(x_3d).any():
#                 return x.reshape(*orig_shape)
            
#             x_3d_rot = self.rotations(x_3d)
            
#             if torch.isnan(x_3d_rot).any():
#                 x_rot = self.proj_up(x_3d)
#             else:
#                 x_rot = self.proj_up(x_3d_rot)
                
#             alpha = 0.9
#             x_rot = alpha * x_rot + (1-alpha) * x_flat
            
#             if torch.isnan(x_rot).any():
#                 return x.reshape(*orig_shape)
                
#         else:
#             x_rot = self.rotations(x_flat)
        
#         return x_rot.reshape(*orig_shape)

#     def apply_rotary_emb(self, freqs, t, start_index=0, scale=1., seq_dim=-2, freqs_seq_dim=None):
#         dtype = t.dtype
        
#         def _exists(val):
#             return val is not None
        
#         def _slice_at_dim(tensor, dim_slice, dim):
#             dim += (tensor.ndim if dim < 0 else 0)
#             colons = [slice(None)] * tensor.ndim
#             colons[dim] = dim_slice
#             return tensor[tuple(colons)]
        
#         def _rotate_half(x):
#             x = rearrange(x, '... (d r) -> ... d r', r=2)
#             x1, x2 = x.unbind(dim=-1)
#             x = torch.stack((-x2, x1), dim=-1)
#             return rearrange(x, '... d r -> ... (d r)')
        
#         if not _exists(freqs_seq_dim):
#             if freqs.ndim == 2 or t.ndim == 3:
#                 freqs_seq_dim = 0
                
#         if t.ndim == 3 or _exists(freqs_seq_dim):
#             seq_len = t.shape[seq_dim]
#             freqs = _slice_at_dim(freqs, slice(-seq_len, None), dim=freqs_seq_dim)
            
#         rot_dim = freqs.shape[-1]
#         end_index = start_index + rot_dim
        
#         assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
        
#         t_left = t[..., :start_index]
#         t_middle = t[..., start_index:end_index]
#         t_right = t[..., end_index:]
        
#         t_transformed = (t_middle * freqs.cos() * scale) + (_rotate_half(t_middle) * freqs.sin() * scale)
#         out = torch.cat((t_left, t_transformed, t_right), dim=-1)
        
#         return out.type(dtype)

#     def rotate_queries_or_keys(self, t, seq_dim=None, offset=0, scale=None):
#         if self.use_quaternion:
#             if self.use_projection and t.shape[-1] > 3:
#                 return self.project_and_rotate(t)
#             else:
#                 return self.rotations(t)
#         else:
#             seq_len = t.shape[2]
#             device, dtype = t.device, t.dtype
            
#             seq = torch.arange(seq_len, device=device, dtype=dtype) + offset
            
#             freqs = self.forward(seq)
            
#             scale = scale if scale is not None else 1.0
#             return self.apply_rotary_emb(freqs, t, scale=scale, seq_dim=2)
    
#     def forward(self, t):
#         freqs = self.freqs
#         freqs = torch.einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
#         freqs = torch.repeat_interleave(freqs, 2, dim=-1)
#         return freqs

