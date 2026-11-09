import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for aluminium rod using a piecewise-linear ramp approximation
    with temperature modulation.
    
    Structure:
    1. Temperature-dependent elastic modulus E(T) = p0 + p1*T.
    2. Temperature-dependent yield strain eps_y(T) = p5 * (1 + p7*T).
    3. Plastic strain eps_p = max(0, eps - eps_y).
    4. Hardening via sum of clipped linear terms (ramps):
       Each ramp k has a start point s_k and slope m_k.
       Ramp contribution: m_k * max(0, eps_p - s_k).
       This creates a convex, piecewise-linear hardening curve.
    5. Temperature modulation of hardening slopes.
    6. Intercept and interaction terms.
    
    Parameters:
    p0: Elastic modulus at T=0
    p1: Temperature coefficient for elastic modulus
    p2: Yield stress scale (multiplies the hardening ramps)
    p3: Temperature coefficient for yield stress scale
    p4: Hardening slope for first ramp (linear hardening rate)
    p5: Yield strain location
    p6: Start point for second ramp (additional hardening onset)
    p7: Temperature coefficient for yield strain
    p8: Intercept
    p9: Slope for second ramp (additional hardening)
    """
    eps = np.asarray(strain, dtype=np.float64)
    T = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-dependent elastic modulus
    E = p0 + p1 * T
    
    # Elastic stress contribution
    elastic_stress = E * eps
    
    # Temperature-dependent yield strain
    eps_y = p5 * (1.0 + p7 * T)
    
    # Plastic strain (only for eps > yield strain)
    eps_p = np.maximum(0.0, eps - eps_y)
    
    # Temperature-dependent yield stress scale
    sigma_scale = p2 + p3 * T
    
    # Piecewise-linear hardening using ramp functions
    # Ramp 1: starts at 0, slope p4
    ramp1 = np.maximum(0.0, eps_p)
    
    # Ramp 2: starts at p6, slope p9
    ramp2 = np.maximum(0.0, eps_p - p6)
    
    # Total hardening stress
    hardening_stress = sigma_scale * (p4 * ramp1 + p9 * ramp2)
    
    # Interaction term
    interaction = p8 * eps * T
    
    # Total stress
    stress = elastic_stress + hardening_stress + interaction
    
    # Ensure finite output
    stress = np.where(np.isfinite(stress), stress, 0.0)
    
    return stress
