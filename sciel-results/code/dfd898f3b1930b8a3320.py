import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Adaptive Radial Basis Function (RBF) Network.
    
    Unlike the fixed linear basis, this model uses 5 RBF centers with learned
    positions and widths. The parameters are:
    p[0:5]   : Centers (c1, c2, c3, c4, c5)
    p[5:10]  : Widths (w1, w2, w3, w4, w5) - used as 1/width in Gaussian exp
    
    The output is the sum of Gaussians:
    y = sum_{i=1}^{5} exp(-((x - c_i)^2) * w_i^2)
    
    Note: Since the coefficients are fixed to 1, this is a non-linear model 
    in parameters. BFGS will optimize the geometry (centers/widths) to 
    approximate the target. To allow for amplitude scaling and offset, 
    we rely on the fact that BFGS can adjust widths to effectively scale 
    the contribution (narrower = higher peak) and centers to align with 
    features. However, pure sum of unit-height Gaussians is limited.
    
    Wait, the task says params length is 10. If I use 5 centers and 5 widths, 
    I have no amplitude or bias. 
    A better parameterization for 10 params with RBFs:
    4 RBFs with (center, width) pairs = 8 params.
    1 global amplitude scale.
    1 bias.
    
    Let's use 4 RBFs:
    p[0], p[1]: c1, w1
    p[2], p[3]: c2, w2
    p[4], p[5]: c3, w3
    p[6], p[7]: c4, w4
    p[8]: global_scale
    p[9]: bias
    
    Model:
    y = p[8] * (exp(-((x-c1)^2)*w1^2) + ... + exp(-((x-c4)^2)*w4^2)) + p[9]
    
    This allows the model to fit peaks and valleys by adjusting centers and 
    sharpness, scaled globally. This is materially different from linear 
    basis expansion as it is non-linear in the parameters (centers/widths).
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
    
    p = np.asarray(params, dtype=np.float64)
    if p.ndim != 1 or p.shape[0] != 10:
        return np.ones_like(x) * (p[0] if p.size > 0 else 0.0)
    
    # Extract parameters
    c1, w1 = p[0], p[1]
    c2, w2 = p[2], p[3]
    c3, w3 = p[4], p[5]
    c4, w4 = p[6], p[7]
    scale = p[8]
    bias = p[9]
    
    # Ensure widths are positive to avoid numerical issues in exp
    # Using abs + small epsilon to keep gradient flow smooth-ish
    w1 = np.abs(w1) + 1e-6
    w2 = np.abs(w2) + 1e-6
    w3 = np.abs(w3) + 1e-6
    w4 = np.abs(w4) + 1e-6
    
    # Compute RBF terms
    # Gaussian: exp(-((x - c)^2) * w^2)
    # Note: If w is large, the peak is narrow. If w is small, the peak is wide.
    # BFGS starting from 1.0 means initial width is 1.0.
    
    term1 = np.exp(-((x - c1) ** 2) * (w1 ** 2))
    term2 = np.exp(-((x - c2) ** 2) * (w2 ** 2))
    term3 = np.exp(-((x - c3) ** 2) * (w3 ** 2))
    term4 = np.exp(-((x - c4) ** 2) * (w4 ** 2))
    
    result = scale * (term1 + term2 + term3 + term4) + bias
    
    # Check for non-finite values
    if not np.all(np.isfinite(result)):
        # Fallback to a simple linear fit if numerical instability occurs
        # This is a safeguard, though RBFs are generally stable
        return np.ones_like(x) * (p[9] if np.isfinite(p[9]) else 0.0)
        
    return result
