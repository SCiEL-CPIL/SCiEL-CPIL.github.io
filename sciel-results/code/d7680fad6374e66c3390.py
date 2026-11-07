"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Uses a Radial Basis Function (RBF) network with learnable centers.
    
    The 10 parameters are split into:
    - p0..p4: Centers of 5 Gaussian basis functions (mapped to [0.1, 6.0])
    - p5..p9: Amplitudes of the 5 Gaussian basis functions
    
    This approach allows the optimizer to place the 'bumps' where they are most
    useful for approximating the target function, rather than being constrained
    to fixed evenly-spaced locations. The widths are fixed to ensure stability
    and prevent the optimizer from collapsing centers to points (which would
    cause ill-conditioning).
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Ensure params is a 1D array
    p = np.asarray(params, dtype=np.float64)
    
    # Handle potential shape issues defensively
    if p.shape != (10,):
        p = p.flatten()
        if len(p) < 10:
             # Pad with zeros if somehow truncated (shouldn't happen with BFGS)
             p = np.pad(p, (0, 10-len(p)))
        else:
             p = p[:10]
             
    # Split parameters
    # First 5: Centers (raw values, will be mapped)
    # Last 5: Amplitudes
    raw_centers = p[0:5]
    amplitudes = p[5:10]
    
    # Map raw centers to the domain [0.1, 6.0]
    # Using a sigmoid-like mapping or linear scaling?
    # Linear scaling from raw values can be unstable if raw values explode.
    # Let's use a soft mapping to ensure centers stay roughly in domain, 
    # but allow some flexibility.
    # Actually, simple linear scaling is often better for BFGS if initialized well.
    # The evaluator starts params at all ones.
    # If raw_centers = 1, we want centers spread out.
    # Let's define a fixed set of "ideal" centers and let params shift them slightly?
    # No, the prompt says "different strategy". Learnable centers is the key.
    
    # Let's map raw_centers to [0.1, 6.0] using a linear transformation that is
    # stable. Since BFGS starts at 1, let's assume raw values near 1 should map to
    # reasonable positions.
    # A robust way: centers = 0.1 + 5.9 * sigmoid(raw_centers) ?
    # Or simpler: centers = 0.1 + 5.9 * (0.5 + 0.5 * tanh(raw_centers))
    # This keeps them in [0.1, 6.0] and is smooth.
    
    centers = 0.1 + 5.9 * (0.5 + 0.5 * np.tanh(raw_centers))
    
    # Fixed widths. With 5 centers in a range of ~6, a width of ~1.2 ensures overlap.
    sigma = 1.2
    
    # Compute basis functions
    # x shape (N,), centers shape (5,)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian basis
    basis = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Linear combination
    # basis shape (N, 5), amplitudes shape (5,)
    y = basis @ amplitudes
    
    # Ensure output is 1D array
    return y
# EVOLVE-BLOCK-END
