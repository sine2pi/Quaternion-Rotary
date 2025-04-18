

    # RotaryEmbedding Usage Guide
    # v.01
    
    Rotary Positional Embedding applies rotation to token embeddings based on their position, 
    helping models understand sequence order. This implementation offers standard RoPE plus 
    quaternion-based rotations and content-dependent variants.

    Update: Now has more stuff.

    




## Initialization Parameters

<img width="683" alt="rot" src="https://github.com/user-attachments/assets/64726baf-8782-4174-a3fd-4cfda78280e5" />

    With freqs=10000 (which becomes theta_scaled), your self.freqs would contain approximately these values:
    
    Starting around 6.28 (2π) for the first element
    Following a geometric decay pattern
    Middle values around 1.0-0.1
    Ending near 0.0006 for the last element
    This creates a frequency spectrum where:
    
    Lower indices (early in the sequence) get high-frequency signals (~6.28)
    Higher indices (deeper in the embedding) get very low-frequency signals (~0.0006)
    This pattern helps the network distinguish between positions at different scales - nearby tokens create strong oscillations in early dimensions, while distant tokens are captured by the slow-changing later dimensions.

```python
    rotary = RotaryEmbedding(
        dim=768,                 # Embedding dimension
        theta=1,                 # Number of frequency components
        num_freqs=10000,         # Base wavelength
        learned_freq=True,       # Whether frequencies are learnable
        theta_rescale_factor=1., # Rescale factor for frequencies
        use_quaternion=False,    # Use quaternion rotation
        ~ rot_scale=1.0,           # Scale for rotation magnitude~ ~
        ~ rot_count=1,             # Number of rotation operations ~   Now rotates 360 degrees based on a learnable parameter
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

# Main method for applying rotations to embeddings:

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

# Increase/decrease rotation effect
    
    rotary.set_rotation_magnitude(magnitude=0.8)

# Implementation Notes

    # Always ensure tensors are on the same device (CUDA)
    # The class handles device movement for projection matrices
    # Contains fallbacks for numerical stability (clipping, NaN checks)
    # Small perturbations prevent identity rotations

```
