"""Bootstrap portfolio: model-and-optimize for stress-strain of aluminium rod."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Record the start time to implement a deadline guard.
_START_TIME = time.time()
_TIMEOUT_LIMIT = 140.0

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature.
    
    Uses a physically-motivated model combining linear elasticity and 
    power-law strain hardening with temperature dependence.
    Parameters are internally scaled to improve BFGS convergence from initial guess of 1.0.
    """
    # Deadline guard
    if time.time() - _START_TIME > _TIMEOUT_LIMIT:
        return np.zeros_like(strain, dtype=np.float64)

    # Ensure inputs are 1D arrays
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # --- Internal Scaling for Better Conditioning ---
    # E_base ~ 70,000 MPa for Al. Start p0=1 -> 10,000 MPa.
    E_base = p0 * 10000.0
    # K_base ~ 100-500 MPa. Start p2=1 -> 100 MPa.
    K_base = p2 * 100.0
    # n ~ 0.2. Start p4=1 -> 0.3.
    n_base = 0.2 + p4 * 0.1
    # H ~ 10. Start p6=1 -> 10.
    H_base = p6 * 10.0
    
    # Temperature coefficients (keep small, assume linear effects are modest relative to base)
    # p1, p3, p5, p7, p8, p9 are scaled to be O(1) or small.
    
    # Temperature-dependent Elastic Modulus E(T)
    # Linear in temp
    E_T = E_base * (1.0 + p1 * temp)
    
    # Temperature-dependent Flow Stress Coefficient K(T)
    K_T = K_base * (1.0 + p3 * temp)
    
    # Temperature-dependent Hardening Exponent n(T)
    # Small variation
    n_T = n_base + (p5 * 0.1) * temp
    
    # Absolute strain for the power-law term
    abs_strain = np.abs(strain)
    
    # Plastic term: K * |eps|^n / (1 + H * |eps|)
    denom = 1.0 + H_base * abs_strain
    
    # Avoid division by zero in denominator
    den_safe = np.where(np.abs(denom) < 1e-8, 1e-8 * np.sign(denom + 1e-300), denom)
    
    # Safe strain to handle n < 0 (though unlikely with scaling)
    # If n is very small positive, 0**n is 0. 
    # If n is negative, 0**n is inf. 
    # We clip strain to 1e-8 to prevent inf if n < 0.
    safe_strain = np.maximum(abs_strain, 1e-8)
    
    # Calculate plastic term
    # Use np.power for flexibility
    try:
        power_term = safe_strain ** n_T
    except:
        # Fallback if n_T causes issues (e.g., negative base with non-integer power, but we use abs)
        power_term = np.ones_like(safe_strain)
        
    plastic_term = K_T * power_term / den_safe
    
    # Elastic term
    elastic_term = E_T * strain
    
    # Total stress
    stress = elastic_term + plastic_term
    
    # Additional corrections (Offset)
    # Scale offsets to be O(100) MPa if p is O(1)
    stress = stress + (p7 * 100.0) * strain + (p8 * 100.0) * temp + (p9 * 100.0)
    
    # Ensure output is finite
    stress = np.where(np.isfinite(stress), stress, 0.0)
    
    return stress
# EVOLVE-BLOCK-END
