"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Strategy: Physics-informed Ramberg-Osgood model.
    The Ramberg-Osgood equation is standard for metals (like aluminium) to model
    the transition from elastic to plastic deformation.
    
    Formula:
    sigma = E * eps + K * (E * eps / sigma_y)^n * sigma_y
    
    We allow parameters to scale the base physical constants to fit the specific
    dataset's units and temperature effects.
    
    Parameters (10):
    p[0]: Elastic Modulus scale (E)
    p[1]: Yield Stress scale (sigma_y)
    p[2]: Hardening Coefficient (K)
    p[3]: Hardening Exponent (n) - constrained to be > 0 via softplus or exp
    p[4]: Temperature coefficient for E (linear)
    p[5]: Temperature coefficient for sigma_y (linear)
    p[6]: Temperature coefficient for K (linear)
    p[7]: Temperature coefficient for n (linear)
    p[8]: Global scale factor
    p[9]: Global offset (bias)
    """
    eps = np.asarray(strain, dtype=np.float64)
    T = np.asarray(temp, dtype=np.float64)
    
    # Extract params
    p = params
    
    # Base physical constants (initial values are 1.0, optimizer will adjust)
    # We use exponential or softplus for parameters that must be positive to avoid
    # domain errors in power functions, but since BFGS starts at 1.0 and moves
    # locally, we can use a safer transformation for n.
    
    # To ensure n > 0 for np.power, we use a transformation.
    # However, if p[3] becomes negative, np.power(x, n) is fine for x>0, 
    # but if x is negative (tensile/compressive), fractional powers of negatives are NaN.
    # Stress-strain is usually modeled for positive strain in these datasets unless 
    # symmetric. Given the range [-0.0019, 0.96], we have negative strain.
    # np.power(negative_base, fractional_exponent) -> NaN.
    # To handle negative strain safely in a power law context:
    # sigma = E*eps + K * sign(eps) * (E * abs(eps) / sigma_y)^n * sigma_y
    # This preserves the sign and uses magnitude for the power law.
    
    E = p[0]
    sy = p[1]
    K = p[2]
    # Ensure n is positive. Using exp ensures n > 0.
    # If p[3] is large, exp(p[3]) might overflow, but BFGS usually keeps it reasonable.
    # Alternatively, we can just use p[3] directly if we assume the optimizer keeps it positive,
    # but exp is safer for the "hardening exponent" constraint.
    # Let's use a softplus-like or just exp. exp(1) = 2.718.
    n = np.exp(p[3])
    
    # Temperature adjustments
    # Linear temperature dependence
    E_T = E * (1.0 + p[4] * (T - 0.5)) # Center T at 0.5 for stability
    sy_T = sy * (1.0 + p[5] * (T - 0.5))
    K_T = K * (1.0 + p[6] * (T - 0.5))
    # n_T: Exponent should also be > 0. 
    # n = exp(p[3] + p[7]*(T-0.5))
    n_T = np.exp(p[3] + p[7] * (T - 0.5))
    
    # Handle division by zero if sy_T is close to 0
    sy_T_safe = np.where(np.abs(sy_T) < 1e-12, 1e-12, sy_T)
    
    # Elastic component
    sigma_elastic = E_T * eps
    
    # Plastic component (Ramberg-Osgood)
    # term = (E * abs(eps) / sy)^n
    # We use abs(eps) to avoid NaNs for negative strain with fractional n.
    eps_abs = np.abs(eps)
    ratio = (E_T * eps_abs) / sy_T_safe
    
    # Clip ratio to avoid overflow in power
    # If ratio is huge and n is large, ratio^n overflows.
    # Max float is ~1e308. 
    # We can clip the result of the power or the ratio.
    # Let's clip the ratio to a safe range, say 1e10, before powering.
    # But 1e10^10 is 1e100, which is safe.
    ratio_clipped = np.clip(ratio, 0.0, 1e10)
    
    plastic_term = K_T * (ratio_clipped ** n_T) * sy_T_safe
    
    # Apply sign of strain to the plastic term
    sigma_plastic = np.sign(eps) * plastic_term
    
    # Total stress
    sigma = sigma_elastic + sigma_plastic
    
    # Global scaling and offset
    sigma = p[8] * sigma + p[9]
    
    # Ensure finite output
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    return sigma
# EVOLVE-BLOCK-END
