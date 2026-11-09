import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Anisotropic RBF Interpolation with Linear Trend.
    
    This approach uses 5 fixed RBF centers to create a local, flexible model.
    Unlike global polynomial bases (Chebyshev/Monomial) which struggle to 
    represent sharp features or non-polynomial shapes with low degrees, 
    RBFs can adapt to local structures.
    
    Parameters (10 total):
    p[0]: Linear bias (b)
    p[1]: Linear x coefficient (mx)
    p[2]: Linear y coefficient (my)
    p[3]: RBF width parameter (sigma). Note: BFGS optimizes this directly.
          We use a transformation to ensure stability if needed, but direct 
          optimization of sigma is often fine if initialized well. 
          However, since BFGS starts at 1, and we want positive sigma, 
          we might use exp(sigma) or just sigma. Let's use exp(p[3]) to 
          ensure sigma > 0 and allow wide range.
          Actually, let's stick to a simple transformation or just use p[3] 
          and hope BFGS stays positive? No, better to use exp(p[3]) for 
          robustness. Wait, if p[3] starts at 1, exp(1)=2.7. 
          Let's use p[3] directly but clip? No, clipping breaks gradients.
          Let's use exp(p[3]).
    p[4]: Weight for RBF center (0, 0)
    p[5]: Weight for RBF center (0.5, 0)
    p[6]: Weight for RBF center (-0.5, 0)
    p[7]: Weight for RBF center (0, 0.5)
    p[8]: Weight for RBF center (0, -0.5)
    p[9]: Additional interaction weight? Or maybe a second width? 
          Let's use p[9] as a weight for a global constant offset added 
          to RBFs? No, p[0] is bias.
          Let's add a simple interaction term x*y with weight p[9] to 
          capture any bilinear curvature that the linear + radial RBFs 
          might not handle well.
    
    Model:
    f(x, y) = p[0] + p[1]*x + p[2]*y + p[9]*x*y + 
              sum_{i=1 to 5} p[3+i] * exp(-|| (x, y) - C_i ||^2 / (2 * sigma^2))
    where sigma = exp(p[3])
    Centers C_i: (0,0), (0.5,0), (-0.5,0), (0,0.5), (0,-0.5)
    """
    start_time = time.time()
    
    # Ensure params length
    if len(params) < 10:
        p = np.zeros(10)
        p[:len(params)] = params
    else:
        p = params[:10]
        
    # Extract parameters
    b = p[0]
    mx = p[1]
    my = p[2]
    sigma_log = p[3]
    w0 = p[4]
    w1 = p[5]
    w2 = p[6]
    w3 = p[7]
    w4 = p[8]
    w_xy = p[9]
    
    # Compute sigma
    # Using exp to ensure positivity. If sigma_log is very negative, sigma ~ 0.
    # If sigma_log is very positive, sigma is large.
    # To prevent overflow in exp, clip sigma_log
    sigma_log_clipped = np.clip(sigma_log, -10, 10)
    sigma = np.exp(sigma_log_clipped)
    
    # Precompute centers
    # C0: (0, 0)
    # C1: (0.5, 0)
    # C2: (-0.5, 0)
    # C3: (0, 0.5)
    # C4: (0, -0.5)
    
    # Calculate squared distances to each center
    # d0^2 = x^2 + y^2
    # d1^2 = (x-0.5)^2 + y^2
    # d2^2 = (x+0.5)^2 + y^2
    # d3^2 = x^2 + (y-0.5)^2
    # d4^2 = x^2 + (y+0.5)^2
    
    x2 = x * x
    y2 = y * y
    
    d0_sq = x2 + y2
    d1_sq = (x - 0.5) * (x - 0.5) + y2
    d2_sq = (x + 0.5) * (x + 0.5) + y2
    d3_sq = x2 + (y - 0.5) * (y - 0.5)
    d4_sq = x2 + (y + 0.5) * (y + 0.5)
    
    # Compute RBF values: exp(-d_sq / (2 * sigma^2))
    # To avoid division by zero if sigma is 0 (sigma_log = -inf), 
    # but we clipped sigma_log to -10, so sigma > 0.
    inv_2_sigma_sq = 1.0 / (2.0 * sigma * sigma)
    
    rbf0 = np.exp(-d0_sq * inv_2_sigma_sq)
    rbf1 = np.exp(-d1_sq * inv_2_sigma_sq)
    rbf2 = np.exp(-d2_sq * inv_2_sigma_sq)
    rbf3 = np.exp(-d3_sq * inv_2_sigma_sq)
    rbf4 = np.exp(-d4_sq * inv_2_sigma_sq)
    
    # Linear + Interaction part
    linear_part = b + mx * x + my * y + w_xy * x * y
    
    # RBF part
    rbf_part = w0 * rbf0 + w1 * rbf1 + w2 * rbf2 + w3 * rbf3 + w4 * rbf4
    
    pred = linear_part + rbf_part
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(pred)):
        # Fallback to linear only if RBFs blow up
        pred = linear_part
        if not np.all(np.isfinite(pred)):
            return np.zeros_like(x)
            
    return pred
