"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a bivariate Chebyshev polynomial expansion.
    The model is linear in the 10 parameters, ensuring a convex optimization
    landscape for BFGS. This provides a robust fit to the global minimum
    of the training MSE, unlike non-linear physical models which may get
    stuck in local minima when started from arbitrary initial values.
    
    Basis functions:
    1. T0(S)*T0(T)
    2. T1(S)*T0(T)
    3. T0(S)*T1(T)
    4. T2(S)*T0(T)
    5. T1(S)*T1(T)
    6. T0(S)*T2(T)
    7. T3(S)*T0(T)
    8. T2(S)*T1(T)
    9. T1(S)*T2(T)
    10. T4(S)*T0(T)
    
    This allows for non-linear temperature dependence (up to order 2) and
    non-linear strain dependence (up to order 4), including cross-terms.
    """
    
    # Normalization constants from training data ranges
    e_min = -0.001911
    e_max = 0.9636
    t_min = 0.06667
    t_max = 1.0
    
    e_range = e_max - e_min
    t_range = t_max - t_min
    
    # Ensure ranges are non-zero
    if e_range < 1e-12:
        e_range = 1.0
    if t_range < 1e-12:
        t_range = 1.0
    
    # Normalize to [-1, 1] for Chebyshev evaluation
    u = 2.0 * (strain - e_min) / e_range - 1.0
    v = 2.0 * (temp - t_min) / t_range - 1.0
    
    # Clip to [-1, 1] for stability
    u = np.clip(u, -1.0, 1.0)
    v = np.clip(v, -1.0, 1.0)
    
    # Pre-compute Chebyshev polynomials for Strain (u)
    T0u = np.ones_like(u)
    T1u = u
    T2u = 2.0 * u * u - 1.0
    T3u = 4.0 * u * u * u - 3.0 * u
    T4u = 8.0 * u * u * u * u - 8.0 * u * u + 1.0
    
    # Pre-compute Chebyshev polynomials for Temp (v)
    T0v = np.ones_like(v)
    T1v = v
    T2v = 2.0 * v * v - 1.0
    
    # Parameters
    p = params
    
    # Construct the linear combination of basis functions
    # Order chosen to prioritize lower-order strain and temp terms,
    # but including higher-order strain for hardening and higher-order temp for property degradation.
    
    term_0 = p[0] * T0u * T0v  # Constant
    term_1 = p[1] * T1u * T0v  # Linear Strain
    term_2 = p[2] * T0u * T1v  # Linear Temp
    term_3 = p[3] * T2u * T0v  # Quadratic Strain
    term_4 = p[4] * T1u * T1v  # Interaction S*T
    term_5 = p[5] * T0u * T2v  # Quadratic Temp
    term_6 = p[6] * T3u * T0v  # Cubic Strain
    term_7 = p[7] * T2u * T1v  # Interaction S^2*T
    term_8 = p[8] * T1u * T2v  # Interaction S*T^2
    term_9 = p[9] * T4u * T0v  # Quartic Strain
    
    stress = term_0 + term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7 + term_8 + term_9
    
    # Ensure finite output
    stress = np.where(np.isfinite(stress), stress, 0.0)
    
    return stress
# EVOLVE-BLOCK-END
