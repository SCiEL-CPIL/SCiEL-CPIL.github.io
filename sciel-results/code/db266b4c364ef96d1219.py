import numpy as np
import time


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a piecewise-linear model with a smooth hinge function to separate
    elastic and plastic regimes. The hinge x*(1-exp(-k*x)) smoothly transitions
    from linear (elastic) to plateau (plastic saturation).
    
    Model:
        E(T) = p0 + p1*T        (temperature-dependent elastic modulus)
        eps_y(T) = p2 + p3*T    (temperature-dependent yield strain)
        H(T) = p4 + p5*T        (temperature-dependent hardening modulus)
        
        elastic_part = E(T) * strain
        plastic_part = (E(T)*eps_y(T) + H(T)*(strain - eps_y(T))) * hinge
        where hinge = (strain/eps_y) * (1 - exp(-k*strain/eps_y)) for strain > 0
        
        stress = elastic_part * (1 - hinge) + plastic_part * hinge + p6*strain*T + p7*strain^2 + p8*T + p9
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    eps = np.asarray(strain, dtype=np.float64)
    T = np.asarray(temp, dtype=np.float64)
    
    # Temperature-dependent elastic modulus
    E_T = p0 + p1 * T
    
    # Temperature-dependent yield strain (ensure positive)
    eps_y = p2 + p3 * T
    eps_y_safe = np.where(np.abs(eps_y) < 1e-12, 1e-12, eps_y)
    
    # Temperature-dependent hardening modulus
    H_T = p4 + p5 * T
    
    # Yield stress
    sigma_y = E_T * eps_y_safe
    
    # Smooth hinge: for x = eps/eps_y, hinge = x*(1 - exp(-k*x))
    # When x < 1 (elastic regime), hinge is small
    # When x >> 1 (plastic regime), hinge approaches x, giving linear hardening
    # We use a fixed shape parameter k embedded in the structure
    k = 3.0  # Fixed shape parameter for the smooth hinge
    
    x = eps / eps_y_safe
    # Smooth hinge function: h(x) = x * (1 - exp(-k*x)) for x >= 0
    # For x < 0 (compression), use a different branch
    x_pos = np.maximum(x, 0.0)
    hinge = x_pos * (1.0 - np.exp(-k * x_pos))
    
    # Normalize hinge so it goes from 0 (fully elastic) to 1 (fully plastic)
    # At x=1, hinge = 1*(1-exp(-k)) = 1-exp(-3) ≈ 0.95
    # We want a smooth transition, so scale to get proper blending
    hinge_normalized = hinge / (1.0 - np.exp(-k))
    # Clamp to [0, 1] for stable blending
    hinge_normalized = np.clip(hinge_normalized, 0.0, 1.0)
    
    # Elastic stress: E_T * eps
    elastic_stress = E_T * eps
    
    # Plastic stress: sigma_y + H_T * (eps - eps_y)
    plastic_stress = sigma_y + H_T * (eps - eps_y_safe)
    
    # Blend: in elastic regime (small strain), use elastic_stress
    # In plastic regime (large strain), use plastic_stress
    stress = elastic_stress * (1.0 - hinge_normalized) + plastic_stress * hinge_normalized
    
    # Additional correction terms for better fit
    stress += p6 * eps * T
    stress += p7 * eps * eps
    stress += p8 * T
    stress += p9
    
    return stress
