import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Smooth softplus-based additive model for E. coli growth rate.
    
    Instead of rigid physical kernels (Monod, Gaussian) or a fixed polynomial,
    this model uses a sum of smooth nonlinear functions of linear combinations
    of the inputs. This allows the optimizer to discover complex response surfaces
    while maintaining smoothness and differentiability for BFGS.
    
    The model structure is:
    y = p0 + sum_i p_i * softplus(a_i * b + c_i * s + d_i * temp + e_i * pH + f_i)
    
    With 10 parameters, we can have:
    - 1 intercept (p0)
    - 1 linear term coefficient (p1) applied to a base linear combination
    - 2 softplus terms, each with 4 coefficients for the linear combo + 1 offset + 1 scale
    
    Actually, let's use a more efficient parameterization:
    p0: intercept
    p1: coefficient for linear term (b + s + temp + pH)
    p2-p5: coefficients for first softplus argument (b, s, temp, pH)
    p6: offset for first softplus
    p7: coefficient for second softplus argument (b - s + temp - pH)
    p8: offset for second softplus
    p9: scale for softplus nonlinearity (shared)
    
    This gives flexibility to capture non-linear saturation effects.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    b = np.asarray(b, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    pH = np.asarray(pH, dtype=np.float64)
    
    # Base linear combination
    linear_combo = b + s + temp + pH
    
    # First softplus term: softplus(p2*b + p3*s + p4*temp + p5*pH + p6)
    arg1 = p2 * b + p3 * s + p4 * temp + p5 * pH + p6
    soft1 = np.log1p(np.exp(p9 * arg1)) / max(p9, 1e-10)
    
    # Second softplus term: softplus(p7*(b - s + temp - pH) + p8)
    arg2 = p7 * (b - s + temp - pH) + p8
    soft2 = np.log1p(np.exp(p9 * arg2)) / max(p9, 1e-10)
    
    # Combine
    return p0 + p1 * linear_combo + soft1 + soft2
