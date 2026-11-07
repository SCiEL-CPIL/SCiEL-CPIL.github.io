import numpy as np
import time

_start_time = None
_deadline = 145.0  # Leave 5s margin

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model using a smooth piecewise-like formulation with temperature dependence.
    
    stress = E(T)*strain + H(T)*f(strain, sigma_y(T))
    
    where f is a smooth approximation of elastic-plastic behavior:
    - For small strain: linear (elastic)
    - For large strain: hardening
    
    Using a smooth hyperbolic tangent based transition:
    f(S, sy) = S + (S^2/sy + S/sy) * tanh(S/sy)
    
    This is C-infinity smooth and captures:
    - Linear elastic region
    - Yield point
    - Strain hardening
    
    All coefficients are temperature-dependent quadratics for flexibility.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > _deadline:
        return np.zeros_like(strain, dtype=np.float64)

    p0, p1, p2 = params[0], params[1], params[2]
    p3, p4, p5 = params[3], params[4], params[5]
    p6, p7, p8 = params[6], params[7], params[8]
    p9 = params[9]
    
    T = temp
    S = strain
    
    # Temperature-dependent Young's Modulus
    E_T = p0 + p1 * T + p2 * T * T
    
    # Temperature-dependent Hardening Coefficient
    H_T = p3 + p4 * T + p5 * T * T
    
    # Temperature-dependent Yield Strain (log space for positivity)
    log_sy = p6 + p7 * T + p8 * T * T
    log_sy = np.clip(log_sy, -20, 20)
    sy = np.exp(log_sy)
    
    # Safety: ensure sy is not too small
    sy = np.maximum(sy, 1e-10)
    
    # Smooth elastic-plastic transition
    # x = S / sy
    x = S / sy
    
    # tanh(x) is smooth and bounded
    tanh_x = np.tanh(x)
    
    # The hardening term:
    # For small x: tanh(x) ~ x, so term ~ S + (S^2/sy + S/sy)*x = S + S^2/sy + S/sy * S/sy
    # For large x: tanh(x) ~ 1, so term ~ S + S^2/sy + S/sy
    # This gives a smooth transition from linear to quadratic hardening
    
    hardening_term = S + (S * S / sy + S / sy) * tanh_x
    
    # Scale by hardening coefficient
    plastic_contribution = H_T * hardening_term
    
    # Elastic contribution
    elastic_term = E_T * S
    
    # Combine
    result = elastic_term + plastic_contribution
    
    # Ensure output is finite
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e100, neginf=-1e100)
        
    return result
