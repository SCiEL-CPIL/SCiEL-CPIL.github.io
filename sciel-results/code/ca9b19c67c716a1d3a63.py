import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a smooth non-linear constitutive model:
    sigma = E(T)*eps + S(T) * (eps / (eps + C(T))) + Q(T)*eps^2 + Offset(T)
    
    Where:
    E(T) = p0 + p1*T  (Elastic Modulus)
    S(T) = p2 + p3*T  (Saturation Stress)
    C(T) = exp(p4 + p5*T) (Characteristic Strain, guaranteed positive)
    Q(T) = p6 + p7*T  (Quadratic Correction)
    Offset(T) = p8 + p9*T
    
    This model captures the elastic region, a smooth transition to plasticity,
    and strain hardening that saturates, which is typical for metals.
    The exponential ensures the characteristic strain parameter remains positive,
    avoiding division by zero and ensuring physical validity of the denominator.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    eps = strain
    T = temp
    
    # Coefficients
    E_T = p0 + p1 * T
    S_T = p2 + p3 * T
    C_T = np.exp(p4 + p5 * T)
    Q_T = p6 + p7 * T
    Offset_T = p8 + p9 * T
    
    # Non-linear hardening term: S * eps / (eps + C)
    # This term is 0 at eps=0, approaches S as eps -> infinity.
    # For small eps, it is approx (S/C) * eps.
    denom = eps + C_T
    # Safety check for division, though C_T > 0 and eps > -0.002 usually keeps denom > 0
    # If denom is very close to 0, we clip it to avoid inf/nan
    denom_safe = np.where(np.abs(denom) < 1e-8, 1e-8, denom)
    
    hardening_term = S_T * (eps / denom_safe)
    
    # Quadratic term
    quad_term = Q_T * eps**2
    
    # Total stress
    sigma = E_T * eps + hardening_term + quad_term + Offset_T
    
    return sigma
