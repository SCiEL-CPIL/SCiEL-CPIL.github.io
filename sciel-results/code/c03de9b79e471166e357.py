"""
Strategy: Radial Basis Function (RBF) approximation.
Instead of a fixed additive basis of physical terms, we use a small set of Gaussian
RBFs centered in the 3D input space (t, x, v). This allows the model to approximate
any smooth function of the inputs. The 10 parameters correspond to the weights,
centers, and widths of these RBFs. This creates a smooth, non-convex but well-behaved
landscape for BFGS, unlike the potentially ill-conditioned polynomial basis.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 135.0
_GLOBAL = {}

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF model with 2 basis functions to fit within 10 parameters.
    
    Parameters (10 total):
    p0: offset (bias)
    p1: weight of RBF 1
    p2: weight of RBF 2
    p3: center t of RBF 1
    p4: center x of RBF 1
    p5: center v of RBF 1
    p6: width of RBF 1 (sigma)
    p7: center t of RBF 2
    p8: center x of RBF 2
    p9: center v of RBF 2
    Note: We fix the width of RBF 2 to be the same as RBF 1 to save parameters, 
    or we can use a single width for both. Let's use a single width parameter 
    and 2 weights, 6 center coords? 
    Let's refine: 
    p0: offset
    p1: w1
    p2: w2
    p3: c_t1
    p4: c_x1
    p5: c_v1
    p6: c_t2
    p7: c_x2
    p8: c_v2
    p9: sigma (shared width)
    """
    # Extract parameters
    offset = params[0]
    w1 = params[1]
    w2 = params[2]
    ct1 = params[3]
    cx1 = params[4]
    cv1 = params[5]
    ct2 = params[6]
    cx2 = params[7]
    cv2 = params[8]
    sigma = params[9]
    
    # Ensure sigma is not zero to avoid division by zero
    sigma_safe = np.maximum(np.abs(sigma), 1e-6)
    
    # Calculate distances squared for each RBF
    # RBF 1
    dt1 = t - ct1
    dx1 = x - cx1
    dv1 = v - cv1
    dist_sq_1 = dt1**2 + dx1**2 + dv1**2
    
    # RBF 2
    dt2 = t - ct2
    dx2 = x - cx2
    dv2 = v - cv2
    dist_sq_2 = dt2**2 + dx2**2 + dv2**2
    
    # Gaussian kernels
    # exp(-dist^2 / (2 * sigma^2))
    # We use 2*sigma^2 in denominator for standard Gaussian
    denom = 2.0 * sigma_safe**2
    
    k1 = np.exp(-dist_sq_1 / denom)
    k2 = np.exp(-dist_sq_2 / denom)
    
    # Combine
    pred = offset + w1 * k1 + w2 * k2
    
    return pred
# EVOLVE-BLOCK-END
