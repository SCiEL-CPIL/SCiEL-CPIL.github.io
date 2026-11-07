"""Non-linear parametric model: Mixed Anisotropic/Isotropic Gaussians with Linear Trend."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a mixed basis of:
    1. A global offset.
    2. An anisotropic Gaussian (independent centers for x, y, z).
    3. An isotropic Gaussian (shared center for x, y, z).
    4. A linear trend term L * (x + y + z).
    
    Model:
    f(x, y, z) = c0 
                 + A1 * exp(-w1 * ((x-c1x)^2 + (y-c1y)^2 + (z-c1z)^2))
                 + A2 * exp(-w2 * ((x-c2)^2 + (y-c2)^2 + (z-c2)^2))
                 + L * (x + y + z)
    
    Parameters (length 10):
    0: c0 (Offset)
    1: A1 (Amplitude Aniso)
    2: c1x (Center X Aniso, tanh mapped)
    3: c1y (Center Y Aniso, tanh mapped)
    4: c1z (Center Z Aniso, tanh mapped)
    5: w1 (Width Aniso, softplus)
    6: A2 (Amplitude Iso)
    7: c2 (Center Iso, tanh mapped)
    8: w2 (Width Iso, softplus)
    9: L (Linear Coefficient)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    
    # Ensure params is a flat array of length 10
    params = np.asarray(params, dtype=np.float64).ravel()
    if params.size != 10:
        params = np.ones(10)

    # Extract parameters
    c0 = params[0]
    
    # Term 1: Anisotropic Gaussian
    A1 = params[1]
    c1x = 1.25 + 0.75 * np.tanh(params[2])
    c1y = 1.25 + 0.75 * np.tanh(params[3])
    c1z = 1.25 + 0.75 * np.tanh(params[4])
    
    w1_raw = params[5]
    w1_abs = np.abs(w1_raw)
    w1_pos = np.maximum(w1_raw, 0) + np.log1p(np.exp(-w1_abs))
    w1_pos = np.clip(w1_pos, 0.1, 10.0)
    
    dx1 = x - c1x
    dy1 = y - c1y
    dz1 = z - c1z
    dist_sq1 = dx1*dx1 + dy1*dy1 + dz1*dz1
    term1 = A1 * np.exp(-w1_pos * dist_sq1)
    
    # Term 2: Isotropic Gaussian
    A2 = params[6]
    c2 = 1.25 + 0.75 * np.tanh(params[7])
    
    w2_raw = params[8]
    w2_abs = np.abs(w2_raw)
    w2_pos = np.maximum(w2_raw, 0) + np.log1p(np.exp(-w2_abs))
    w2_pos = np.clip(w2_pos, 0.1, 10.0)
    
    dx2 = x - c2
    dy2 = y - c2
    dz2 = z - c2
    dist_sq2 = dx2*dx2 + dy2*dy2 + dz2*dz2
    term2 = A2 * np.exp(-w2_pos * dist_sq2)
    
    # Term 3: Linear Trend
    L = params[9]
    term3 = L * (x + y + z)
    
    prediction = c0 + term1 + term2 + term3
    
    # Ensure finite output
    if not np.all(np.isfinite(prediction)):
        return np.full_like(x, np.nan)
        
    return prediction
# EVOLVE-BLOCK-END
