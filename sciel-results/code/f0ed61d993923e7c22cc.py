import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Voce-type exponential hardening model for stress-strain of an aluminium rod.
    
    sigma = sigma_y(T) + Q(T) * (1 - exp(-|strain| / H(T))) * sign(strain)
    
    Where:
    sigma_y(T) = p0 + p1*T          (Yield stress, linear in temperature)
    Q(T)       = p2 + p3*T          (Saturation stress, linear in temperature)
    H(T)       = p4 + p5*T          (Hardening modulus, linear in temperature)
    E(T)       = p6 + p7*T          (Elastic modulus, linear in temperature)
    p8, p9:    Additional linear-in-strain correction terms
    
    The model captures:
    - Linear elastic response via E(T)*strain for small strains
    - Exponential saturation of hardening (key feature of aluminium)
    - Temperature dependence of all material parameters
    - Asymmetric behavior if needed via the sign function
    
    For numerical stability, H is clipped to [1e-4, 1000] to avoid overflow in exp.
    The sign function is approximated smoothly using tanh for differentiability.
    """
    p = params
    
    # Ensure inputs are float arrays
    strain = np.asarray(strain, dtype=float)
    temp = np.asarray(temp, dtype=float)
    
    # If parameters are not finite, return zeros to avoid propagation of NaNs/Infs
    if not np.all(np.isfinite(p)):
        return np.zeros_like(strain)
    
    T = temp
    
    # Temperature-dependent material parameters
    sigma_y = p[0] + p[1] * T       # Yield stress
    Q = p[2] + p[3] * T             # Saturation stress
    H = p[4] + p[5] * T             # Hardening modulus
    E = p[6] + p[7] * T             # Elastic modulus
    
    # Clip H to avoid numerical issues in exp
    H_safe = np.clip(H, 1e-4, 1000.0)
    
    # Compute the exponential hardening term
    # Use |strain| for the exponential argument, and sign(strain) for direction
    abs_strain = np.abs(strain)
    
    # exp(-|strain|/H) - clip the exponent to avoid underflow issues
    exponent = -abs_strain / H_safe
    exponent = np.clip(exponent, -50.0, 0.0)  # exp(-50) ~ 1.9e-22, effectively 0
    exp_term = np.exp(exponent)
    
    # Voce hardening: Q * (1 - exp(-|strain|/H))
    voce_hardening = Q * (1.0 - exp_term)
    
    # Sign function: use tanh for smooth differentiability
    # tanh(strain / small_value) approximates sign(strain)
    # But we want it to be 0 at strain=0 and ±1 for large |strain|
    # Actually, for the hardening to be proportional to strain direction,
    # we use: sign(strain) * voce_hardening
    # But sign is not differentiable. Use a smooth approximation.
    # Alternative: just use strain/|strain| for strain != 0, 0 for strain = 0
    # But that's not differentiable either.
    
    # Better approach: model as sigma = E*strain + sigma_y*sign(strain) + Q*(1-exp(-|strain|/H))*sign(strain)
    # The sign terms can be problematic. Let's use a different formulation.
    
    # Actually, for metals, the stress-strain curve is typically:
    # - Linear elastic up to yield
    # - Then plastic hardening
    # The Voce model is typically written for the plastic part.
    
    # Let's use a simpler, more robust formulation:
    # sigma = E * strain + (sigma_y + Q*(1 - exp(-|strain|/H))) * tanh(strain / (H_safe * 0.01 + 1e-8))
    
    # The tanh term smoothly transitions from 0 to ±1 as strain increases
    # This makes the whole expression differentiable
    
    # Actually, let's think about this differently. The issue is that at strain=0,
    # the yield stress shouldn't kick in. Let's use:
    
    # sigma = E * strain + (sigma_y + Q*(1 - exp(-|strain|/H))) * (strain / (|strain| + eps))
    # where eps is small. But strain/(|strain|+eps) is still not perfectly smooth.
    
    # Let's use the formulation that's known to work well:
    # sigma = E * strain + K * (strain^2 + delta)^((n-1)/2) * strain  (Ramberg-Osgood style)
    # But we want Voce-type saturation.
    
    # Best approach: use the fact that for most of the strain range (away from 0),
    # sign(strain) is well-defined. At strain=0, the contribution is 0 anyway
    # if we multiply by strain.
    
    # sigma = E * strain + (sigma_y + Q*(1 - exp(-|strain|/H))) * strain / (|strain| + delta)
    
    # This is smooth for delta > 0 and captures the physics.
    
    delta = 1e-6
    sign_factor = strain / (abs_strain + delta)
    
    # The yield + hardening part, scaled by sign factor
    hardening_part = (sigma_y + voce_hardening) * sign_factor
    
    # Elastic part
    elastic_part = E * strain
    
    # Additional correction terms for flexibility
    correction = p[8] * strain + p[9] * strain * T
    
    sigma = elastic_part + hardening_part + correction
    
    # Ensure output is finite
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    return sigma
