"""Symbolic regression: hybrid RBF + rational basis with 10 parameters."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model combining RBF and rational bases.
    
    f(x, y) = p0 + sum_{i=1}^{5} p_i * RBF_i(x, y) + rational_part(x, y; p6..p9)
    
    where RBF_i are Gaussian kernels at 5 well-chosen centers, and
    rational_part = (p6 + p7*x + p8*y + p9*x*y) / (1 + 0.5*tanh(p6)*x^2 + 0.5*tanh(p7)*y^2)
    
    The RBF part captures local features, while the rational part handles
    global smooth structure. The tanh reparameterization ensures the
    denominator stays bounded away from zero.
    
    Inputs x, y are in [-1, 1].
    Params start at 1.0.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    p = params

    # --- RBF part: 5 centers, using p0..p4 ---
    # Centers chosen to cover the domain well
    centers = [
        (0.0, 0.0),
        (-0.5, -0.5),
        (0.5, 0.5),
        (-0.5, 0.5),
        (0.5, -0.5)
    ]
    
    gamma = 3.0  # RBF width parameter
    
    rbf_part = np.full_like(x, p[0], dtype=np.float64)
    for i in range(5):
        cx, cy = centers[i]
        dist_sq = (x - cx)**2 + (y - cy)**2
        rbf_val = np.exp(-gamma * dist_sq)
        rbf_part += p[i+1] * rbf_val

    if time.time() > deadline:
        return rbf_part

    # --- Rational part: using p5..p9 ---
    # Numerator: p5 + p6*x + p7*y + p8*x*y + p9*(x^2 - y^2)
    # Denominator: 1 + 0.5*tanh(p5)*x^2 + 0.5*tanh(p6)*y^2 (bounded)
    # Wait, I need to be careful about param usage. Let me restructure.
    # 
    # Better: Use p0..p4 for RBF, p5..p9 for rational.
    # Rational: (p5 + p6*x + p7*y + p8*xy) / (1 + a*x^2 + b*y^2)
    # where a = 0.5*tanh(p9-5) ... no, let's use separate params.
    # 
    # Actually, let's use:
    # RBF: p0..p4 (5 params)
    # Rational numerator: p5 + p6*x + p7*y + p8*x*y
    # Rational denominator: 1 + 0.5*tanh(p9)*x^2 + 0.5*tanh(p9)*y^2
    # But that only uses p9 for denominator. Let me use:
    # Denominator: 1 + 0.3*tanh(p5)*x^2 + 0.3*tanh(p6)*y^2
    # But p5, p6 are also in numerator. That's fine, BFGS will optimize.
    #
    # Simpler approach: 
    # RBF part uses p0..p4
    # Rational part: (p5 + p6*x + p7*y + p8*x*y) / (1 + 0.5*exp(-p9)*x^2 + 0.5*exp(-p9)*y^2)
    # The exp(-p9) ensures the denominator coefficient is positive and bounded.
    # If p9 is large positive, exp(-p9) is small, denominator ~ 1.
    # If p9 is large negative, exp(-p9) is large, but we can cap it.
    # Actually, let's use tanh for safety.
    
    x2 = x * x
    y2 = y * y
    xy = x * y
    
    num = p[5] + p[6] * x + p[7] * y + p[8] * xy
    
    # Denominator with bounded, positive coefficients
    # a = 0.5 * tanh(p[9]) ensures a in [-0.5, 0.5]
    # den = 1 + a*(x^2 + y^2)
    # For x,y in [-1,1], x^2+y^2 in [0,2]
    # den in [1 - 0.5*2, 1 + 0.5*2] = [0, 2]
    # To avoid zero, use max(den, 0.1)
    a = 0.5 * np.tanh(p[9])
    den = 1.0 + a * (x2 + y2)
    den = np.maximum(den, 0.1)
    
    rational_part = num / den
    
    # Combine
    pred = rbf_part + rational_part
    
    # Ensure finite values
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
