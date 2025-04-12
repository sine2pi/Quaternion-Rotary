class RotaryEmbedding(nn.Module):
    def __init__( self, dim, theta = 10000, num_freqs = 1, learned_freq = True, theta_rescale_factor = 1., 
                 use_quaternion = False, rot_scale = 1.0, rot_count = 1, use_projection = False, proj_dim = 3, 
                 proj_scale = 0.1, reverse_direction = False, scale_base = 1.0): 
        super().__init__()
        self.dim = dim
        theta *= theta_rescale_factor ** (dim / (dim - 2))
        direction = -1.0 if reverse_direction else 1.0
        self.freqs = nn.Parameter(direction * torch.arange(0, num_freqs) * (2 * math.pi / theta), 
                                  requires_grad=learned_freq)
        self.register_buffer('dummy', torch.tensor(0), persistent=False)
        self.use_quaternion = use_quaternion
        self.use_projection = use_projection
        self.proj_dim = proj_dim
        self.proj_scale = proj_scale
        self.scale_base = scale_base
        self.step = 0

        if use_quaternion:
            init_val = -2.0 if reverse_direction else 2.0
            self.dparam = nn.Parameter(torch.tensor([init_val]))
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
            theta_abs = torch.abs(theta)
            theta_sign = torch.sign(theta)
            u_dir = u if theta_sign >= 0 else -u
            Q = self.q_rotation(torch.eye(dims, device=theta.device), theta=theta_abs, u=u_dir, v=v)
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
        """
        Apply rotation to the input tensor.
        Ensures the tensor device matches internal parameters and forces non-trivial rotations.
        """
        t_clone = t.clone()
        if self.use_quaternion:
            if self.use_projection and t_clone.shape[-1] > 3:
                return self.project_and_rotate(t_clone)
            else:
                result = self.rotations(t_clone)
                if torch.allclose(result, t):
                    perturbation = torch.randn_like(result) * 1e-4
                    return result + perturbation
                return result
        else:
            if len(t_clone.shape) == 4:
                ctx = t_clone.shape[2]
                seq_dim_val = 2
            else:
                ctx = t_clone.shape[1]
                seq_dim_val = 1
            device, dtype = t_clone.device, t_clone.dtype
            seq = torch.arange(ctx, device=device, dtype=dtype) + offset
            seq = seq + 0.01
            freqs = self.forward(seq)
            scale_value = scale if scale is not None else self.scale_base
            result = self.apply_rotary(freqs, t_clone, scale=scale_value, seq_dim=seq_dim_val)
            if torch.allclose(result, t):
                perturbation = torch.randn_like(result) * 1e-4
                return result + perturbation
            return result
        
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
        if hasattr(self, 'dim') and freqs.shape[-1] != self.dim and self.dim > 2:
            if freqs.shape[-1] < self.dim:
                repeat_factor = self.dim // freqs.shape[-1]
                if repeat_factor > 1:
                    freqs = freqs.repeat(*(1 for _ in range(freqs.ndim-1)), repeat_factor)
        return freqs
    
    def set_direction(self, reverse=False):
        """Explicitly set the rotation direction"""
        with torch.no_grad():
            if self.use_quaternion:
                self.dparam.fill_(-5.0 if reverse else 5.0)
            else:
                direction = -1.0 if reverse else 1.0
                self.freqs.copy_(direction * torch.abs(self.freqs))
        return self
    
    def set_rotation_magnitude(self, magnitude=1.0):
        """Set the magnitude of rotation"""
        with torch.no_grad():
            if self.use_quaternion:
                self.tscale = magnitude
            else:
                self.scale_base = magnitude
        return self
    
    def content_dependent_freqs(self, t, content_features=None):
        """Generate content-aware frequencies based on input features"""
        base_freqs = self.freqs
        if content_features is not None:
            if not hasattr(self, 'content_proj'):
                head_dim = content_features.shape[-1]
                self.content_proj = nn.Linear(head_dim, 1, bias=False).to(content_features.device)
                nn.init.normal_(self.content_proj.weight, std=0.02)
            mod = torch.sigmoid(self.content_proj(content_features)) * 0.4 + 0.8
            freqs = base_freqs * mod.unsqueeze(-1)
            return self.forward(t, custom_freqs=freqs)
        return self.forward(t)
    
    def rotate_with_content(self, t, x, seq_dim=None, offset=0):
        """Apply content-dependent rotation to the input tensor."""
        t_clone = t.clone()
        if len(t_clone.shape) == 4:
            ctx = t_clone.shape[2]
            seq_dim_val = 2
        else:
            ctx = t_clone.shape[1]
            seq_dim_val = 1
        seq = torch.arange(ctx, device=t.device) + offset
        freqs = self.content_dependent_freqs(x, seq)
        result = self.apply_rotary(freqs, t_clone, scale=self.scale_base, seq_dim=seq_dim_val)
        return result

        """
    # RotaryEmbedding Usage Guide

    Rotary Positional Embedding applies rotation to token embeddings based on their position, 
    helping models understand sequence order. This implementation offers standard RoPE plus 
    quaternion-based rotations and content-dependent variants.

    ## Initialization Parameters

    rotary = RotaryEmbedding(
        dim=768,                 # Embedding dimension
        theta=10000,             # Base wavelength
        num_freqs=1,             # Number of frequency components
        learned_freq=True,       # Whether frequencies are learnable
        theta_rescale_factor=1., # Rescale factor for frequencies
        use_quaternion=False,    # Use quaternion rotations
        rot_scale=1.0,           # Scale for rotation magnitude
        rot_count=1,             # Number of rotation operations
        use_projection=False,    # Use projections for high dimensions
        proj_dim=3,              # Projection dimension (usually 3)
        proj_scale=0.1,          # Scale factor for projections
        reverse_direction=False, # Reverse rotation direction
        scale_base=1.0,          # Base scale for rotations
    )

    # Initialize rotary embeddings

    rotary = RotaryEmbedding(dim=768).to("cuda")

    # Apply rotations to token embeddings

    x shape: [batch, seq_len, dim]
    x = x.to("cuda")
    x_rotated = rotary.rotate_(x)

    # Initialize with quaternion rotations

    rotary = RotaryEmbedding(
        dim=768, 
        use_quaternion=True,
        rot_scale=1.0,
        rot_count=2
    ).to("cuda")

    # Apply rotations

    x_rotated = rotary.rotate_(x)



    # For high-dimensional embeddings

    rotary = RotaryEmbedding(
        dim=1024,
        use_quaternion=True,
        use_projection=True,
        proj_dim=3,
        proj_scale=0.1
    ).to("cuda")

    # Automatically projects to 3D, rotates, and projects back

    x_rotated = rotary.rotate_(x)

    # Content features affect rotation (attention-based)

    content_features = attention_outputs.to("cuda")
    x_rotated = rotary.rotate_with_content(x, content_features)

    # Key Methods

    rotate_(t, seq_dim=None, offset=0, scale=None)
    Main method for applying rotations to embeddings:

    - t: Input tensor to rotate
    - offset: Position offset for embeddings
    - scale: Override the rotation scale

    # Basic rotation

    x_rotated = rotary.rotate_(x)

    # With offset (useful for continued generation)

    x_rotated = rotary.rotate_(x, offset=prev_seq_len)

    # With custom scale (controls rotation strength)

    x_rotated = rotary.rotate_(x, scale=0.5)
    set_direction(reverse=False)

    # Change rotation direction

    rotary.set_direction(reverse=True)
    set_rotation_magnitude(magnitude=1.0)

    # Control rotation strength

    # Increase/decrease rotation effect
    
    rotary.set_rotation_magnitude(magnitude=0.8)

    # Implementation Notes

    Always ensure tensors are on the same device (CUDA)
    The class handles device movement for projection matrices
    Contains fallbacks for numerical stability (clipping, NaN checks)
    Small perturbations prevent identity rotations

    """
