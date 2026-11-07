"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model combining Voce exponential saturation with power-law hardening.
    
    sigma = E(T)*eps + K_voce(T)*[1-exp(-eps/beta)]*tanh(eps/a) + K_pow(T)*|eps|^n*tanh(eps/a)
    
    This model uses:
    - Tanh-based smooth sign for better BFGS convergence
    - Voce-type exponential saturation for realistic hardening behavior
    - Power-law term for additional flexibility
    - Temperature-dependent parameters with flexible functional forms
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    s = strain
    t = temp
    p = params
    
    # Reference temperature
    Tref = 0.5
    dt = t - Tref
    
    # Temperature-dependent Young's modulus: E(T) = p0 * (1 + p1*dt + p2*dt^2)
    E = p[0] * (1.0 + p[1] * dt + p[2] * dt * dt)
    
    # Voce saturation parameter: beta = exp(p3) (ensures positive)
    beta = np.exp(np.clip(p[3], -5, 5))
    
    # Voce hardening coefficient: K_voce(T) = p4 * (1 + p5*dt)
    K_voce = p[4] * (1.0 + p[5] * dt)
    
    # Power-law exponent: n = exp(p6) (ensures positive)
    n = np.exp(np.clip(p[6], -3, 3))
    
    # Power-law coefficient: K_pow(T) = p7 * (1 + p8*dt)
    K_pow = p[7] * (1.0 + p[8] * dt)
    
    # Smooth sign function using tanh
    # tanh(eps/a) where a controls the sharpness
    a = np.exp(np.clip(p[9] - 10, -2, 2)) + 1e-6  # small positive value
    
    # For the tanh sign, use a small scaling to make it smooth but steep
    # We want tanh(eps / small_value) ≈ sign(eps) but smooth
    # Use a fixed small value for the sign-like term, and let p[9] control something else
    # Actually, let's use a simpler approach: tanh(eps * scale) where scale is large
    
    # Smooth sign: tanh(eps * 100) approximates sign(eps) smoothly
    # But for numerical stability, use a moderate scale
    sign_scale = 50.0
    smooth_sign = np.tanh(s * sign_scale)
    
    # Handle near-zero strain for power law
    abs_s = np.abs(s)
    eps_min = 1e-8
    abs_s_safe = np.maximum(abs_s, eps_min)
    
    # Voce term: K_voce * (1 - exp(-|eps|/beta)) * smooth_sign
    # Using |eps| to keep it monotonic in magnitude
    voce_term = K_voce * (1.0 - np.exp(-abs_s_safe / beta)) * smooth_sign
    
    # Power-law term: K_pow * |eps|^n * smooth_sign
    power_term = K_pow * np.power(abs_s_safe, n) * smooth_sign
    
    # Elastic term
    elastic_term = E * s
    
    # Total stress
    sigma = elastic_term + voce_term + power_term
    
    # Check deadline
    if time.time() > deadline:
        return sigma
    
    return sigma
# EVOLVE-BLOCK-END
