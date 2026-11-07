import numpy as np
import time

def equation(strain, temp, params):
    """
    Symbolic regression for stress-strain of an aluminium rod.
    
    Strategy: Nonlinear Parametric Model (Ramberg-Osgood + Power Law).
    Unlike the linear basis approach, this model uses the parameters to
    define the physical structure:
    - p0, p1: Elastic Modulus E(T) = p0 + p1*T
    - p2, p3: Yield Stress Y(T) = p2 + p3*T
    - p4: Hardening Exponent n (controls shape of plastic curve)
    - p5: Hardening Scale K (controls magnitude of hardening)
    - p6, p7: Shift and Scale for strain axis (to handle thermal expansion effects)
    - p8: Residual Stress / Offset
    - p9: Temperature-dependent hardening coefficient
    
    This allows BFGS to adjust the *shape* of the curve (via exponent n)
    rather than just weighting fixed shapes.
    """
    s = strain
    t = temp
    p = params
    
    # Ensure inputs are 1D arrays
    s = np.asarray(s)
    t = np.asarray(t)
    
    # Safety: Clip parameters to prevent overflow/underflow in exp/pow
    # E should be positive, Y positive, n > 0, K > 0
    E = np.clip(p[0] + p[1] * t, 1.0, 1e6)
    Y = np.clip(p[2] + p[3] * t, 0.1, 1e3)
    n = np.clip(p[4], 0.5, 10.0)  # Hardening exponent
    K = np.clip(p[5], 0.1, 1e3)   # Hardening scale
    
    # Strain adjustments
    # p6: strain shift (e.g., thermal expansion offset)
    # p7: strain scale (e.g., gauge length correction)
    s_adj = (s - p[6]) / np.clip(p[7], 0.1, 10.0)
    
    # Ramberg-Osgood formulation:
    # epsilon = sigma/E + K*(sigma/Y)^n
    # We need sigma(epsilon). This is implicit.
    # However, for symbolic regression tasks where we provide sigma directly,
    # we can invert this or use a direct forward model that approximates it.
    # A direct forward model for sigma(epsilon) is difficult to invert analytically 
    # for arbitrary n. 
    # 
    # Alternative Strategy: Use a direct piecewise power law model which is 
    # easier to optimize directly as sigma(s).
    # 
    # Let's switch to a direct forward model that is smooth:
    # sigma = E * s + K * (s - s_y)^p * H(s - s_y)  is kinked.
    # 
    # Better: Use a smooth transition function like a logistic or tanh 
    # but with parameters controlling the transition width and position.
    # 
    # Re-evaluating: The prompt asks for a DIFFERENT strategy. 
    # The current one is Linear Basis. 
    # Inspiration 3 mentioned avoiding hard bifurcations.
    # Let's use a **Generalized Polynomial Basis with Adaptive Centers** 
    # or a **RBF-like basis** where centers are parameters.
    # 
    # Actually, a very robust nonlinear model for stress-strain is:
    # sigma = A * s + B * s^2 + C * (s - s0)^3 * H(s-s0) ...
    # But let's try a **Fractional Power Law** model which is smooth.
    #
    # Model: sigma = E*s + K * ( (s - s_y) / s_max )^n  for s > s_y
    # To make it smooth and differentiable everywhere for BFGS:
    # Use a smooth max function: smooth_max(x, 0) = 0.5 * (x + sqrt(x^2 + eps))
    
    s_y = np.clip(p[6], 0.0001, 0.1)  # Yield strain
    s_max = np.clip(p[7], 0.01, 1.0)   # Normalization strain
    eps = 1e-8
    
    # Elastic part
    E = np.clip(p[0] + p[1] * t, 1.0, 1e6)
    sigma_elastic = E * s
    
    # Plastic part
    # Excess strain
    excess = s - s_y
    # Smooth max to ensure differentiability
    excess_smooth = 0.5 * (excess + np.sqrt(excess**2 + eps))
    
    # Power law hardening
    # Base term
    base = excess_smooth / s_max
    # Clip base to avoid negative power issues if n is not integer
    base_clipped = np.clip(base, 0.0, 10.0)
    
    # Hardening exponent
    n_exp = np.clip(p[4], 0.5, 5.0)
    
    # Hardening magnitude
    # p5 is scale, p3 is temp dependence of hardening
    K_hard = np.clip(p[5] * (1.0 + p[3] * t), 0.0, 1e3)
    
    sigma_plastic = K_hard * (base_clipped ** n_exp)
    
    # Total stress
    sigma = sigma_elastic + sigma_plastic
    
    # Additional corrections
    # p8: Linear temp effect on stress (thermal stress)
    # p9: Quadratic strain correction
    sigma += p[8] * t + p[9] * s**2
    
    # Ensure finite
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    return sigma
