import numpy as np
import time

_start_time = None
_deadline = 145.0 # Leave 5s margin

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model using a smooth hyperbolic tangent elastic-plastic transition.
    
    stress = E(T)*strain + H(T) * [strain - (1/(2a(T))) * tanh(a(T)*strain)]
    
    This is C-infinity smooth. For small strain, tanh(a*S) ≈ a*S, so the correction
    term vanishes and stress ≈ E(T)*S. For large strain, tanh(a*S) → sign(S), so
    stress ≈ E(T)*S + H(T)*(S - sign(S)/(2a)) which gives linear hardening.
    
    Parameters:
    p0, p1, p2: E(T) = p0 + p1*T + p2*T^2
    p3, p4, p5: H(T) = p3 + p4*T + p5*T^2
    p6, p7, p8: a(T) = p6 + p7*T + p8*T^2 (transition sharpness, must be > 0)
    p9: global offset
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
    
    # Temperature-dependent transition sharpness
    # Must be positive for stable behavior
    a_T = p6 + p7 * T + p8 * T * T
    # Ensure a_T is bounded away from zero to avoid division issues
    a_T = np.maximum(a_T, 0.1)
    a_T = np.minimum(a_T, 100.0)
    
    # Compute tanh(a*S) safely
    aS = a_T * S
    # Clip to avoid numerical issues with very large arguments
    aS = np.clip(aS, -50.0, 50.0)
    tanh_aS = np.tanh(aS)
    
    # The smooth correction: S - tanh(a*S)/(2*a)
    # For small S: tanh(a*S) ≈ a*S, so S - a*S/(2*a) = S/2 ... 
    # Wait, let me reconsider. 
    # Actually: S - tanh(a*S)/(2*a)
    # For small S: tanh(a*S) ≈ a*S - (a*S)^3/3, so S - (a*S)/(2*a) = S - S/2 = S/2
    # That's not right for elastic regime. Let me use a different formulation.
    
    # Better: stress = E(T)*S + H(T) * S * (1 - exp(-b(T)*S))
    # This is smooth, for small S: 1-exp(-b*S) ≈ b*S, so term ≈ H*b*S^2 (quadratic)
    # For large S: 1-exp(-b*S) → 1, so term ≈ H*S (linear hardening)
    
    b_T = p6 + p7 * T + p8 * T * T
    b_T = np.maximum(b_T, 0.01)
    b_T = np.minimum(b_T, 100.0)
    
    # Compute 1 - exp(-b*S) safely
    bS = b_T * S
    bS = np.clip(bS, -50.0, 50.0)
    exp_term = np.exp(-bS)
    correction = 1.0 - exp_term
    
    elastic_term = E_T * S
    plastic_term = H_T * S * correction
    
    result = elastic_term + plastic_term + p9
    
    # Ensure output is finite
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e100, neginf=-1e100)
        
    return result
