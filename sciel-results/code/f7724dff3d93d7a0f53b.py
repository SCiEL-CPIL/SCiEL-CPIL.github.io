import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression for a nonlinear oscillator using a Gaussian Process kernel basis.
    
    We model the target y as a function of (x, v) using a sum of Gaussian kernels (RBF).
    The parameters control the length scales (anisotropy), signal variance, and noise variance.
    
    f(x, v) = p[8] * sum_{i=1}^{4} exp(-0.5 * ((x - c_xi)^2 / l_x^2 + (v - c_vi)^2 / l_v^2))
    
    However, we only have 10 params. 
    Let's use a simpler but powerful structure:
    f(x, v) = p[0] * exp(-((x - p[1])^2 / p[2]^2 + (v - p[3])^2 / p[4]^2)) 
             + p[5] * exp(-((x - p[6])^2 / p[7]^2 + (v - p[8])^2 / p[9]^2))
             
    Wait, this is a sum of 2 Gaussians with 4 params each (center x, center v, width x, width v) + amplitude.
    That's 5 params per Gaussian. 2 Gaussians = 10 params.
    
    This allows the function to form two "bumps" or "wells" in the (x,v) phase space, 
    which is a good inductive bias for oscillator dynamics that might have multiple attractors 
    or complex potential landscapes.
    
    To avoid numerical issues with small widths (division by zero or overflow in exp), 
    we clamp the widths.
    """
    p = params
    
    # Unpack parameters for two Gaussian components
    # Component 1: amplitude p[0], centers p[1], p[3], widths p[2], p[4]
    # Component 2: amplitude p[5], centers p[6], p[8], widths p[7], p[9]
    
    amp1 = p[0]
    cx1 = p[1]
    sx1 = p[2]
    cv1 = p[3]
    sv1 = p[4]
    
    amp2 = p[5]
    cx2 = p[6]
    sx2 = p[7]
    cv2 = p[8]
    sv2 = p[9]
    
    # Ensure widths are positive and not too small to avoid numerical instability
    # If width is close to 0, the Gaussian becomes a delta function, which is bad for BFGS gradients.
    # We use a soft clamp.
    min_width = 1e-3
    sx1 = np.where(np.abs(sx1) < min_width, np.sign(sx1) * min_width if sx1 != 0 else min_width, sx1)
    sv1 = np.where(np.abs(sv1) < min_width, np.sign(sv1) * min_width if sv1 != 0 else min_width, sv1)
    sx2 = np.where(np.abs(sx2) < min_width, np.sign(sx2) * min_width if sx2 != 0 else min_width, sx2)
    sv2 = np.where(np.abs(sv2) < min_width, np.sign(sv2) * min_width if sv2 != 0 else min_width, sv2)
    
    # Calculate Gaussian terms
    # Using log-sum-exp or just exp with clipping for stability
    # z = -0.5 * ( ((x-cx)/sx)^2 + ((v-cv)/sv)^2 )
    
    dx1 = x - cx1
    dv1 = v - cv1
    z1 = -0.5 * ( (dx1 / sx1)**2 + (dv1 / sv1)**2 )
    
    dx2 = x - cx2
    dv2 = v - cv2
    z2 = -0.5 * ( (dx2 / sx2)**2 + (dv2 / sv2)**2 )
    
    # Clip z to prevent underflow/overflow, though exp handles negative large numbers well (to 0)
    # The main risk is if sx is tiny, dx/sx is huge, z is -huge, exp is 0. That's fine.
    # If sx is negative, (dx/sx)^2 is positive, so z is negative. Fine.
    
    g1 = np.exp(z1)
    g2 = np.exp(z2)
    
    return amp1 * g1 + amp2 * g2
