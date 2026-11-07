import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    p = params.flatten()
    
    if len(p) < 10:
        return p[0]*x + p[1]*y + p[2]*z + p[3]
        
    # Center inputs to improve numerical stability for BFGS
    # Mean of [0.5, 2] is 1.25
    cx = x - 1.25
    cy = y - 1.25
    cz = z - 1.25
    
    # Numerator: Linear combination of 1, x, y, z, xy, xz, yz (using centered vars)
    num = (p[0] + 
           p[1] * cx + 
           p[2] * cy + 
           p[3] * cz + 
           p[4] * (cx * cy) + 
           p[5] * (cx * cz) + 
           p[6] * (cy * cz))
           
    # Denominator: 1 + scaled linear combination of x, y, z (using centered vars)
    # Scaling by 0.5 keeps the denominator closer to 1, improving stability
    den = 1.0 + 0.5 * (p[7] * cx + p[8] * cy + p[9] * cz)
    
    # Robust safety check: clamp denominator away from zero
    # This ensures the function is well-defined and gradients are finite
    den_safe = np.sign(den) * np.maximum(np.abs(den), 1e-4)
    # Handle the case where den is exactly 0 (sign is 0)
    den_safe = np.where(den_safe == 0, 1e-4, den_safe)
    
    return num / den_safe
