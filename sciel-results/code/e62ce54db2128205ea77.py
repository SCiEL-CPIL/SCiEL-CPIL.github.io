"""Mixture of Logistic (Sigmoid) Units: structural change to capture sharp transitions."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: sum of 3 sigmoidal units + linear residual.
    Each unit i has parameters (a_i, b_i, c_i, d_i, s_i, w_i):
      - a_i, b_i, c_i: coefficients for linear combination of (x, y, z)
      - d_i: offset (center)
      - s_i: steepness
      - w_i: weight
    Plus a linear residual: e0 + e1*x + e2*y + e3*z
    
    Total params: 3*6 + 4 = 22... but we only have 10.
    
    Let me restructure: 2 sigmoid units + 6 linear terms.
    Each sigmoid: (a, b, c, d, s, w) = 6 params
    2 sigmoids = 12 params, too many.
    
    Alternative: 3 sigmoid units, each with (a, b, c, d, s, w) but share some.
    
    Let's use: 3 units, each with (center_x, center_y, center_z, steepness, weight)
    That's 5 params per unit = 15, still too many.
    
    Better: 2 sigmoid units, each with (a, b, c, d, s, w) = 6 params = 12, still too many.
    
    Let me use: 2 sigmoid units with shared structure.
    Unit i: sigmoid(s_i * (a_i*x + b_i*y + c_i*z + d_i)) * w_i
    That's 6 params per unit. 2 units = 12. Need 10 total.
    
    Alternative: 1 sigmoid unit (6 params) + 4 linear terms = 10 params.
    Or: 2 sigmoid units with 4 params each (a, b, c, d) where s and w are derived.
    
    Let's try: 2 sigmoid units, each with (a, b, c, d, s) = 5 params, weight = 1.
    2*5 = 10 params. No linear residual.
    
    Actually, let's be more flexible:
    params[0:3] = (a1, b1, c1) for unit 1 linear combo
    params[3:5] = (d1, s1) for unit 1 offset and steepness
    params[5:8] = (a2, b2, c2) for unit 2 linear combo
    params[8:10] = (d2, s2) for unit 2 offset and steepness
    
    Weights are fixed at 1.0. This limits expressiveness but keeps param count at 10.
    
    To add more flexibility, let's use:
    params[0:3] = (a1, b1, c1)
    params[3] = d1
    params[4] = s1
    params[5] = w1
    params[6:8] = (a2, b2)  # unit 2 only uses x and y
    params[8] = d2
    params[9] = s2
    
    Hmm, this is getting complicated. Let me just use a clean design:
    
    3 sigmoid units, each with 3 params: (linear_combo_coeff, center, steepness)
    The linear combo is just a weighted sum: coeff * (x + y + z) or similar.
    
    Actually, simplest robust approach:
    params[0:3] = weights for x, y, z in a single linear combination L = p0*x + p1*y + p2*z
    params[3:6] = centers for 3 sigmoids on L: c1, c2, c3
    params[6:9] = steepness for 3 sigmoids: s1, s2, s3
    params[9] = overall scale
    
    f = scale * (sigmoid(s1*(L-c1)) + sigmoid(s2*(L-c2)) + sigmoid(s3*(L-c3)))
    
    This is 10 params. The model is a sum of 3 sigmoids applied to a single linear projection.
    This can capture step-like behavior along one direction in input space.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Linear projection
    L = p0 * x + p1 * y + p2 * z
    
    # Three sigmoid units
    # Clip arguments to avoid overflow in exp
    arg1 = np.clip(p6 * (L - p3), -500, 500)
    arg2 = np.clip(p7 * (L - p4), -500, 500)
    arg3 = np.clip(p8 * (L - p5), -500, 500)
    
    sig1 = 1.0 / (1.0 + np.exp(-arg1))
    sig2 = 1.0 / (1.0 + np.exp(-arg2))
    sig3 = 1.0 / (1.0 + np.exp(-arg3))
    
    return p9 * (sig1 + sig2 + sig3)
# EVOLVE-BLOCK-END
