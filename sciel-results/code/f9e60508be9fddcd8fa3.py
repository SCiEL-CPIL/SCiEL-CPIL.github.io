import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear-in-parameters basis expansion for stress-strain of an aluminium rod.
    
    sigma = sum(p_i * phi_i(strain, temp))
    
    Basis functions (10 total):
    phi_0 = 1
    phi_1 = strain
    phi_2 = strain * temp
    phi_3 = strain * temp^2
    phi_4 = strain * (strain^2 + delta)^0.5   # ~ |strain|*|strain| for large strain, smooth at 0
    phi_5 = strain * (strain^2 + delta)^1.0   # ~ strain^3
    phi_6 = strain * (strain^2 + delta)^1.5   # ~ strain^5
    phi_7 = strain * temp * (strain^2 + delta)^0.5
    phi_8 = strain * temp^2 * (strain^2 + delta)^0.5
    phi_9 = strain * (strain^2 + delta)^2.0   # ~ strain^7
    """
    strain = np.asarray(strain, dtype=float)
    temp = np.asarray(temp, dtype=float)
    p = params
    
    if not np.all(np.isfinite(p)):
        return np.zeros_like(strain)
    
    # Small constant for numerical stability at strain=0
    delta = 1e-8
    s2 = strain * strain
    base = s2 + delta
    
    # Precompute powers of base
    base_half = np.power(base, 0.5)   # (strain^2 + delta)^0.5
    base_1 = base                     # (strain^2 + delta)^1.0
    base_1_5 = base * base_half       # (strain^2 + delta)^1.5
    base_2 = base * base              # (strain^2 + delta)^2.0
    
    # Basis functions
    phi_0 = 1.0
    phi_1 = strain
    phi_2 = strain * temp
    phi_3 = strain * temp * temp
    phi_4 = strain * base_half
    phi_5 = strain * base_1
    phi_6 = strain * base_1_5
    phi_7 = strain * temp * base_half
    phi_8 = strain * temp * temp * base_half
    phi_9 = strain * base_2
    
    # Linear combination
    sigma = (p[0] * phi_0 + p[1] * phi_1 + p[2] * phi_2 + p[3] * phi_3 +
             p[4] * phi_4 + p[5] * phi_5 + p[6] * phi_6 + p[7] * phi_7 +
             p[8] * phi_8 + p[9] * phi_9)
    
    return sigma
