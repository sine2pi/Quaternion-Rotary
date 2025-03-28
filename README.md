Basic usage 

        self.rotary1 = RotaryEmbedding(
            dim=dims//head,
            theta=10000,
            use_quaternion=True,
            use_projection=True, <- if you need more than 3 dimensions
            rot_scale=4.0,
            rot_count=1
        )

        self.rotary2 = RotaryEmbedding(
            dim=dims//head,
            theta=-6000, <-change direction
            use_quaternion=False, <- falls back to regular RoPE
            use_projection=False,
            rot_scale=1.0,
            rot_count=4
        )

      q = self.rotary1.rotate_queries_or_keys(q)
      k = self.rotary2.rotate_queries_or_keys(k)
      ... as many as you need
        see code for more options


 use_projection = True
 
    The code projects to a new 3D dimension using a learned linear projection
    Applies true quaternion rotations in this 3D space
    Projects back to the original dimension using another learned projection
    The projections are initialized as pseudo-inverses of each other

 use_projection = False
 
    Directly applies rotations in the original dimension
    For dimensions > 3, it falls back to using multiple 2D Givens rotations
    It does NOT use quaternion rotation except when exactly in 3D
    Each pair of dimensions gets rotated separately in 2D planes


```python
import math
import torch
import torch.nn as nn
from einops import rearrange

class QuaternionRotary(nn.Module):

    def __init__(
        self,
        dim,
        theta = 10000,
        num_freqs = 1,
        learned_freq = False,
        theta_rescale_factor = 1.0,
        use_quaternion = False,
        rot_scale = 1.0,
        rot_count = 1,
        use_projection = True,
        proj_dim = 3,
        proj_scale = 0.1,
    ):
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
        """Quaternion rotation for 3D vectors."""
        eps = 1e-6
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
        """Create a rotation matrix for dimensions i,j with angle theta."""
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

    def _ensure_projection_layers(self, x):
        if self.proj_down is None or self.proj_down.weight.device != x.device:
            head_dim = x.shape[-1] 
            
            self.proj_down = nn.Linear(head_dim, self.proj_dim, bias=False).to(x.device)
            self.proj_up = nn.Linear(self.proj_dim, head_dim, bias=False).to(x.device)
            
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
            self._ensure_projection_layers(x)
            
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

    def apply_rotary_emb(self, freqs, t, start_index=0, scale=1., seq_dim=-2, freqs_seq_dim=None):
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
            seq_len = t.shape[seq_dim]
            freqs = _slice_at_dim(freqs, slice(-seq_len, None), dim=freqs_seq_dim)
            
        rot_dim = freqs.shape[-1]
        end_index = start_index + rot_dim
        
        assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
        
        t_left = t[..., :start_index]
        t_middle = t[..., start_index:end_index]
        t_right = t[..., end_index:]
        
        t_transformed = (t_middle * freqs.cos() * scale) + (_rotate_half(t_middle) * freqs.sin() * scale)
        out = torch.cat((t_left, t_transformed, t_right), dim=-1)
        
        return out.type(dtype)

    def rotate_queries_or_keys(self, t, seq_dim=None, offset=0, scale=None):
        if self.use_quaternion:
            if self.use_projection and t.shape[-1] > 3:
                return self.project_and_rotate(t)
            else:
                return self.rotations(t)
        else:
            seq_len = t.shape[2]
            device, dtype = t.device, t.dtype
            
            seq = torch.arange(seq_len, device=device, dtype=dtype) + offset
            
            freqs = self.forward(seq)
            
            scale = scale if scale is not None else 1.0
            return self.apply_rotary_emb(freqs, t, scale=scale, seq_dim=2)
    
    def forward(self, t):
        freqs = self.freqs
        freqs = torch.einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = torch.repeat_interleave(freqs, 2, dim=-1)
        return freqs
```
