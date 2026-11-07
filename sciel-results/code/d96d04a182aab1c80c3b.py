"""Symbolic regression: an equation for E. coli growth rate."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Multiplicative mechanistic growth model with smooth differentiable constraints.
    
    Strategy:
    1. Use a multiplicative structure (Monod * Temp * pH * Density) to enforce the biological
       reality that growth is zero if any factor is limiting. This creates a highly non-convex
       landscape, but the structure is strongly aligned with the data generation process.
    2. Use smooth functions (exp, log, rational) to ensure gradients are well-defined for BFGS.
    3. Use 10 parameters:
       p0: Max growth rate (mu_max)
       p1: Half-saturation constant (K_s)
       p2: Optimal temperature (T_opt)
       p3: Temperature sensitivity (sigma_T)
       p4: Optimal pH (pH_opt)
       p5: pH sensitivity (sigma_pH)
       p6: Density inhibition rate (k_b)
       p7: Substrate inhibition constant (K_i) - for Haldane effect if needed
       p8: Baseline offset (to handle noise or non-zero background)
       p9: Scaling factor for density effect
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure parameters are positive where necessary to avoid domain errors in log/exp
    # BFGS might step into negative territory. We use softplus-like transformations or clipping.
    # However, the task says params are fitted by the evaluator. We must ensure the function
    # is robust.
    
    # 1. Substrate: Haldane/Monod model
    # mu_s = s / (K_s + s + (s^2 / K_i))
    # To avoid division by zero or negative K_s, we use a safe version.
    # If p1 (K_s) is negative, the model breaks. 
    # Standard practice in these symbolic regression tasks: assume the optimizer stays in a reasonable basin
    # or use transformations. Given BFGS starts at all ones, and the data is positive growth,
    # p1, p7 should likely stay positive.
    
    # Let's use a robust form:
    # f_s = s / (1 + p1*s + p7*s*s)  -- This is a rational function.
    # If p1, p7 are positive, it's monotonic then decreasing (Haldane).
    # If p1 is large, it's like Monod with K_s = 1/p1.
    
    # Let's try a simpler Monod first: s / (1 + p1*s)
    # But Haldane is better for E. coli if substrate inhibition exists.
    # Let's use: s / (1 + p1*s + p7*s^2)
    # We need to ensure denominator > 0.
    # Since s >= 1, and p1, p7 start at 1, denom is positive.
    
    denom_s = 1.0 + p1 * s + p7 * s * s
    # Safe division
    with np.errstate(divide='ignore', invalid='ignore'):
        f_s = np.where(denom_s > 1e-6, s / denom_s, 0.0)
    
    # 2. Temperature: Gaussian response
    # f_T = exp(-((temp - T_opt)^2) / (2 * sigma_T^2))
    # p2 = T_opt, p3 = sigma_T
    # Ensure sigma_T is positive. If p3 < 0, exp(-x / neg) -> exp(pos) -> overflow.
    # Use abs(p3) or p3^2.
    sigma_T = np.abs(p3) + 1e-6
    diff_T = temp - p2
    f_T = np.exp(-(diff_T ** 2) / (2.0 * sigma_T ** 2))
    
    # 3. pH: Gaussian response
    # p4 = pH_opt, p5 = sigma_pH
    sigma_pH = np.abs(p5) + 1e-6
    diff_pH = pH - p4
    f_pH = np.exp(-(diff_pH ** 2) / (2.0 * sigma_pH ** 2))
    
    # 4. Density: Exponential inhibition
    # f_b = exp(-k_b * b)
    # p6 = k_b
    # If p6 is negative, growth increases with density (unlikely). 
    # Use abs(p6) to enforce inhibition? Or let it be free?
    # Biological prior: inhibition. Let's use exp(-abs(p6)*b) to be safe?
    # Or just exp(-p6*b) and let BFGS find the sign.
    # Given start at 1, it will likely stay positive.
    f_b = np.exp(-p6 * b)
    
    # Combine multiplicatively
    # Growth = p0 * f_s * f_T * f_pH * f_b + p8
    # p9 is unused in this specific formulation, or we can use it to scale f_b.
    # Let's use p9 to scale the density effect amplitude?
    # Or use p9 as a second order temp term?
    # Let's stick to 9 params and ignore p9? No, must use 10.
    # Let's make p9 a scaling factor for the whole multiplicative term.
    # Growth = p9 * p0 * f_s * f_T * f_pH * f_b + p8
    
    y = p9 * p0 * f_s * f_T * f_pH * f_b + p8
    
    return y
# EVOLVE-BLOCK-END
