import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a power-law hardening (Ramberg-Osgood style) model:
        sigma = E(T) * eps + K(T) * |eps|^n * sign(eps)
    
    where:
        E(T) = p0 + p1 * T      (elastic modulus, linear in temperature)
        K(T) = p2 + p3 * T      (hardening coefficient, linear in temperature)
        n = softplus(p4)        (hardening exponent, kept positive via softplus)
    
    Additional flexibility terms:
        + p5 * eps^2            (quadratic correction)
        + p6 * T * eps          (thermal-stress coupling)
        + p7 * eps * T^2        (higher-order thermal coupling)
        + p8 * T^2              (thermal offset)
        + p9                    (global offset)
    
    This directly encodes standard metal plasticity physics:
    - Linear elastic rise at low strain (E*eps dominates)
    - Power-law hardening at higher strain (K*|eps|^n dominates)
    - Temperature dependence on both elastic and plastic regimes
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Softplus for positive exponent: n = log(1 + exp(p4))
    # This is smooth and differentiable everywhere
    # Clamp p4 to avoid overflow in exp
    p4_clamped = np.clip(p4, -50, 50)
    n = np.log1p(np.exp(p4_clamped))
    
    # Temperature-dependent coefficients
    E = p0 + p1 * temp          # Elastic modulus
    K = p2 + p3 * temp          # Hardening coefficient
    
    # Power-law hardening term: K * |eps|^n * sign(eps)
    # Use np.sign and absolute value for differentiability at 0
    abs_strain = np.abs(strain)
    sign_strain = np.sign(strain)
    
    # Compute |eps|^n safely (eps can be 0)
    # For eps=0, 0^n = 0 for n>0, which is correct
    power_term = np.power(abs_strain + 1e-15, n) * sign_strain
    
    # Main stress components
    elastic = E * strain
    hardening = K * power_term
    
    # Additional flexibility terms
    quad_corr = p5 * strain**2
    thermal_coupling = p6 * temp * strain
    thermal_coupling2 = p7 * strain * temp**2
    thermal_offset = p8 * temp**2
    global_offset = p9
    
    stress = elastic + hardening + quad_corr + thermal_coupling + thermal_coupling2 + thermal_offset + global_offset
    
    return stress
