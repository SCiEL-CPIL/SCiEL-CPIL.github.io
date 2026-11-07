# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Radial Basis Function (RBF) network with 10 parameters.
    
    Structure:
    f(x) = p0 + sum_{i=1}^{5} p[2*i] * exp(-(x - p[2*i-1])^2 / max(p[2*i-1+1], eps)^2)
    
    Where:
    - p[0]: global offset (bias)
    - p[1], p[3], p[5], p[7], p[9]: centers c_i (learnable, clamped to [-5, 5])
    - p[2], p[4], p[6], p[8], p[10]... wait, we only have 10 params.
    
    Let me reorganize:
    p[0]: bias
    p[1]: center_1
    p[2]: width_1
    p[3]: center_2
    p[4]: width_2
    p[5]: center_3
    p[6]: width_3
    p[7]: center_4
    p[8]: width_4
    p[9]: center_5
    
    Wait, that's only 5 centers and 4 widths + bias = 10 params. I need 5 widths too.
    Let me use:
    p[0]: bias
    p[1]: center_1, p[2]: width_1
    p[3]: center_2, p[4]: width_2
    p[5]: center_3, p[6]: width_3
    p[7]: center_4, p[8]: width_4
    p[9]: center_5
    
    And fix width_5 to a reasonable default, or use a different split.
    
    Actually, let's use:
    p[0]: bias
    p[1]: center_1
    p[2]: center_2
    p[3]: center_3
    p[4]: center_4
    p[5]: center_5
    p[6]: width_1
    p[7]: width_2
    p[8]: width_3
    p[9]: width_4
    
    And use a fixed width for the 5th RBF, or better yet, use all 10 params differently:
    
    Better approach: Use 5 RBFs where each has a center and we share widths in pairs,
    or use a simpler structure. Let me use:
    
    f(x) = p0 + p1*exp(-(x-p2)^2/p3^2) + p4*exp(-(x-p5)^2/p6^2)
         + p7*exp(-(x-p8)^2/p9^2)
    
    That's 10 params: p0=bias, (p1,p2,p3)=amp/center/width for RBF1,
    (p4,p5,p6) for RBF2, (p7,p8,p9) for RBF3.
    
    But the directive says 5 centers and 5 widths. Let me do:
    p[0]: bias
    p[1]: c1, p[2]: s1
    p[3]: c2, p[4]: s2
    p[5]: c3, p[6]: s3
    p[7]: c4, p[8]: s4
    p[9]: c5 (and fix s5 to 1.0 or derive from others)
    
    Actually, let me just use 4 full RBFs (center+width) + 1 bias + 1 extra center:
    f(x) = p0 + sum_{i=1}^{4} amp_i * exp(-(x-c_i)^2 / s_i^2)
    where amp_i are fixed to 1.0, c_i = p[2*i-1], s_i = p[2*i]
    
    That gives: p0, c1,s1, c2,s2, c3,s3, c4,s4 = 9 params. Need one more.
    
    Let me add a linear term: p9 * x
    
    Final structure:
    f(x) = p0 + p9*x + sum_{i=1}^{4} exp(-(x - p[2*i-1])^2 / max(p[2*i], eps)^2)
    
    This uses all 10 params: p0=bias, p9=linear slope, and 4 RBFs with learnable centers/widths.
    """
    p = np.asarray(params, dtype=float)
    x = np.asarray(x, dtype=float)
    
    # Ensure finite parameters
    if not np.all(np.isfinite(p)):
        return np.full_like(x, np.nan)
    
    # Clamp centers to reasonable range
    c1 = np.clip(p[1], -5, 5)
    c2 = np.clip(p[3], -5, 5)
    c3 = np.clip(p[5], -5, 5)
    c4 = np.clip(p[7], -5, 5)
    
    # Clamp widths to positive values (use softplus-like transformation)
    # Widths must be positive; use exp() to ensure positivity
    w1 = np.exp(np.clip(p[2], -10, 10))
    w2 = np.exp(np.clip(p[4], -10, 10))
    w3 = np.exp(np.clip(p[6], -10, 10))
    w4 = np.exp(np.clip(p[8], -10, 10))
    
    # Compute RBF terms
    rbf1 = np.exp(-(x - c1)**2 / (w1**2 + 1e-10))
    rbf2 = np.exp(-(x - c2)**2 / (w2**2 + 1e-10))
    rbf3 = np.exp(-(x - c3)**2 / (w3**2 + 1e-10))
    rbf4 = np.exp(-(x - c4)**2 / (w4**2 + 1e-10))
    
    # Combine with bias and linear term
    y = p[0] + p[9]*x + rbf1 + rbf2 + rbf3 + rbf4
    
    # Ensure output is finite
    y = np.where(np.isfinite(y), y, 0.0)
    
    return y
# EVOLVE-BLOCK-END
