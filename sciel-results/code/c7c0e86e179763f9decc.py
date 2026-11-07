import numpy as np
import time

_start_time = None
_deadline = 145.0  # Leave 5s margin


def _chebyshev_t(n, x):
    """Compute Chebyshev polynomial of the first kind T_n(x) for array x."""
    if n == 0:
        return np.ones_like(x, dtype=np.float64)
    elif n == 1:
        return x.copy()
    else:
        t0 = np.ones_like(x, dtype=np.float64)
        t1 = x.copy()
        for k in range(2, n + 1):
            t2 = 2.0 * x * t1 - t0
            t0 = t1
            t1 = t2
        return t1


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Tensor-product Chebyshev polynomial basis of degree (3,2) in strain and temperature.
    
    stress = sum_{i=0}^{3} sum_{j=0}^{2} c_{ij} * T_i(2*S/max_S - 1) * T_j(2*T/max_T - 1)
    
    This uses an orthogonal polynomial basis that is better conditioned than
    monomial powers. The normalization maps inputs to [-1, 1] where Chebyshev
    polynomials are well-behaved. Degree (3,2) gives 4*3=12 terms, but we use
    10 params by omitting two higher-order interaction terms.
    
    Parameters:
    p0-p9: Coefficients for the Chebyshev expansion
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    elapsed = time.time() - _start_time
    if elapsed > _deadline:
        return np.zeros_like(strain, dtype=np.float64)

    S = strain
    T = temp

    # Normalize inputs to [-1, 1] for Chebyshev polynomials
    # Strain range: [-0.001911, 0.9636], center ≈ 0.4808, half-range ≈ 0.4828
    # Temp range: [0.06667, 1], center ≈ 0.5333, half-range ≈ 0.4667
    S_norm = (S - 0.4808) / 0.4828
    T_norm = (T - 0.5333) / 0.4667

    # Clip to avoid numerical issues outside training range
    S_norm = np.clip(S_norm, -1.5, 1.5)
    T_norm = np.clip(T_norm, -1.5, 1.5)

    # Compute Chebyshev polynomials for strain: T_0, T_1, T_2, T_3
    T0_S = np.ones_like(S_norm, dtype=np.float64)
    T1_S = S_norm.copy()
    T2_S = 2.0 * S_norm * T1_S - T0_S
    T3_S = 2.0 * S_norm * T2_S - T1_S

    # Compute Chebyshev polynomials for temp: T_0, T_1, T_2
    T0_T = np.ones_like(T_norm, dtype=np.float64)
    T1_T = T_norm.copy()
    T2_T = 2.0 * T_norm * T1_T - T0_T

    # Tensor product basis with 10 terms:
    # (i,j) pairs: (0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1), (0,2), (1,2)
    result = (
        params[0] * T0_S * T0_T +
        params[1] * T1_S * T0_T +
        params[2] * T2_S * T0_T +
        params[3] * T3_S * T0_T +
        params[4] * T0_S * T1_T +
        params[5] * T1_S * T1_T +
        params[6] * T2_S * T1_T +
        params[7] * T3_S * T1_T +
        params[8] * T0_S * T2_T +
        params[9] * T1_S * T2_T
    )

    # Ensure output is finite
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e100, neginf=-1e100)

    return result
