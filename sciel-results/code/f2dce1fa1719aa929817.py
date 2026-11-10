"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a sum of Gaussian radial basis functions.
    
    This approach uses a localized, non-linear basis that can approximate sharp features,
    discontinuities, or multi-modal structures that global polynomial or Chebyshev bases
    smooth out excessively. Each Gaussian is parameterized by mean, width, and amplitude.
    
    We use 5 Gaussians with 2 parameters each (mean and width) plus a global amplitude
    per Gaussian, but since we have 10 params total, we use:
    - 5 means (p[0:5])
    - 5 widths (p[5:10])
    And use fixed amplitudes of 1.0 for each, letting the optimizer adjust means and widths.
    
    Actually, to be more flexible, let's use:
    - 5 means (p[0:5])
    - 3 widths (p[5:8]) - shared or varying
    - 2 amplitudes (p[8:10])
    
    Better approach: Use 5 Gaussians where each has a mean and width, but we only have
    10 params. So: 5 means + 5 widths. Amplitudes are fixed at 1.0, and the linear
    combination is implicit in the sum. But this limits expressiveness.
    
    Alternative: Use 5 Gaussians with means p[0:5] and widths p[5:10], and the output
    is the sum of exp(-(x - mean_i)^2 / (2 * width_i^2)). The optimizer can adjust
    means and widths to fit the target.
    
    To avoid division by zero or very small widths, we clip widths to a minimum value.
    """
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Extract parameters
    means = p[0:5]
    widths = p[5:10]
    
    # Clip widths to avoid numerical issues
    # Widths should be positive; use abs and add a small epsilon
    widths_safe = np.abs(widths) + 1e-6
    
    # Compute the sum of Gaussians
    result = np.zeros_like(x, dtype=np.float64)
    
    for i in range(5):
        # Gaussian: exp(-(x - mean)^2 / (2 * width^2))
        diff = x - means[i]
        exponent = -(diff * diff) / (2.0 * widths_safe[i] * widths_safe[i])
        # Clip exponent to avoid underflow issues (though exp handles small values fine)
        exponent = np.clip(exponent, -700, 700)
        result += np.exp(exponent)
    
    # Ensure finite output
    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return result

# EVOLVE-BLOCK-END
