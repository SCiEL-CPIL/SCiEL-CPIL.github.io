"""Model-and-optimize: Scaled Power Polynomial closure for Lorenz-96 subgrid tendency."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """
    Fit a degree-5 polynomial using a scaled power basis for numerical stability.
    
    Strategy:
    1. Scale x to z in [-1, 1] using min-max normalization.
    2. Fit a degree-5 polynomial P(z) = sum a_i * z^i using np.polynomial.polynomial.polyfit.
    3. Analytically transform coefficients from z-basis to x-basis using binomial expansion.
    4. This avoids ill-conditioning from large x values and grid-fitting errors.
    """
    t_start = time.time()
    DEADLINE = 135.0  # Safety margin before 150s kill
    
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    n = len(x)
    
    if n < 6:
        # Not enough points for degree 5, fallback to lower degree
        deg = min(5, n - 1)
        if deg < 0:
            return np.zeros(6)
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, deg)
            # Pad to length 6
            full_coeff = np.zeros(6)
            full_coeff[:deg+1] = coeff
            return full_coeff
        except Exception:
            return np.zeros(6)
    
    # 1. Scaling: Map x to z in [-1, 1]
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    
    if x_range < 1e-10:
        # Constant x, fit a constant
        mean_y = np.mean(y)
        coeff = np.zeros(6)
        coeff[0] = mean_y
        return coeff
    
    # z = (x - center) / scale
    # We want z = (x - c) / s such that min(x) -> -1 and max(x) -> 1
    # x_min -> (x_min - c)/s = -1
    # x_max -> (x_max - c)/s =  1
    # Subtracting: (x_max - x_min)/s = 2 => s = (x_max - x_min) / 2
    # Adding: (x_max + x_min - 2c)/s = 0 => 2c = x_max + x_min => c = (x_max + x_min) / 2
    
    x_center = (x_min + x_max) / 2.0
    x_scale = x_range / 2.0
    
    z = (x - x_center) / x_scale
    
    # 2. Fit polynomial in z
    degree = 5
    try:
        # polyfit returns coefficients in ascending powers: [a_0, a_1, ..., a_5]
        # such that P(z) = a_0 + a_1*z + ... + a_5*z^5
        a = np.polynomial.polynomial.polyfit(z, y, degree)
    except Exception:
        # Fallback to lower degree if rank deficient
        try:
            a = np.polynomial.polynomial.polyfit(z, y, degree - 1)
            a = np.append(a, 0.0)
        except Exception:
            a = np.zeros(6)
            if n >= 2:
                a[:2] = np.polynomial.polynomial.polyfit(z, y, 1)

    # 3. Transform coefficients from z to x
    # We have P(z) = sum_{i=0}^5 a_i * z^i
    # z = (x - c) / s = (1/s) * (x - c)
    # z^i = (1/s^i) * (x - c)^i
    # (x - c)^i = sum_{j=0}^i binom(i, j) * x^j * (-c)^(i-j)
    #
    # So P(x) = sum_{i=0}^5 a_i * (1/s^i) * sum_{j=0}^i binom(i, j) * x^j * (-c)^(i-j)
    # Let b_j be the coefficient of x^j in P(x).
    # b_j = sum_{i=j}^5 a_i * (1/s^i) * binom(i, j) * (-c)^(i-j)
    
    b = np.zeros(6)
    for i in range(6):
        if a[i] == 0:
            continue
        coeff_scale = a[i] / (x_scale ** i)
        for j in range(i + 1):
            # binom(i, j)
            binom_val = 1.0
            # Calculate binom(i, j) efficiently for small i
            if j > i - j:
                j_temp = i - j
            else:
                j_temp = j
            for k in range(1, j_temp + 1):
                binom_val *= (i - j_temp + k) / k
            
            term = coeff_scale * binom_val * ((-x_center) ** (i - j))
            b[j] += term
    
    # Ensure finite
    if not np.all(np.isfinite(b)):
        b = np.zeros(6)
        try:
            b[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        except Exception:
            pass
    
    return b
# EVOLVE-BLOCK-END
