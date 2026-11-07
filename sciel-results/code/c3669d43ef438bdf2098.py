"""Symbolic regression with 1 input(s)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a combination of global polynomial basis
    and localized Radial Basis Functions (RBFs).
    
    The model uses:
    - 4 global Chebyshev polynomials (T0 to T3) for the overall trend.
    - 6 localized Gaussian RBFs centered at distinct points in [-3, 3] with
      varying widths to capture local non-linear features.
    
    The 10 parameters correspond to the coefficients of these 10 basis functions.
    This hybrid approach allows the model to fit both global trends and local
    variations that a pure polynomial might underfit.
    """
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill
    
    # Ensure x is a 1D array for broadcasting
    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    
    # Normalize input to [-1, 1] for polynomial basis
    u = x / 3.0
    
    # Initialize result
    result = np.zeros(n, dtype=np.float64)
    
    # --- Global Chebyshev Polynomials (T0 to T3) ---
    # T_0(u) = 1
    T0 = np.ones(n)
    result += params[0] * T0
    
    # T_1(u) = u
    T1 = u.copy()
    result += params[1] * T1
    
    # T_2(u) = 2*u^2 - 1
    T2 = 2.0 * u * T1 - T0
    result += params[2] * T2
    
    # T_3(u) = 4*u^3 - 3*u
    T3 = 2.0 * u * T2 - T1
    result += params[3] * T3
    
    # --- Localized Gaussian RBFs ---
    # Centers spread across [-3, 3]
    centers = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    # Varying widths to capture different scales of variation
    # Smaller widths for finer detail, larger for broader features
    widths = np.array([0.5, 0.8, 1.0, 1.0, 0.8, 0.5])
    
    for i in range(6):
        if i + 4 >= len(params):
            break
        # Gaussian RBF: exp(-((x - center)^2) / (2 * width^2))
        diff = x - centers[i]
        gaussian = np.exp(-0.5 * (diff / widths[i]) ** 2)
        result += params[i + 4] * gaussian
    
    # Safety check: if computation took too long (unlikely), return what we have.
    if time.time() > deadline:
        return result

    return result
# EVOLVE-BLOCK-END
