"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 115.0  # Safety margin for 120s limit

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE:
        return np.zeros_like(x)
    
    # Ensure x is 1D for consistent indexing
    x = np.asarray(x).flatten()
    
    # 5 Gaussian RBFs with fixed centers to ensure domain coverage
    # Centers: -3, -1.5, 0, 1.5, 3
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    
    # Parameters:
    # 0-4: Log-widths (sigma)
    # 5-9: Amplitudes
    log_widths = params[:5]
    amplitudes = params[5:]
    
    # Calculate widths, ensuring positivity
    widths = np.exp(log_widths)
    
    # Vectorized calculation of RBF values
    # x: (N, 1), centers: (1, 5) -> diffs: (N, 5)
    x_col = x[:, np.newaxis]
    c_row = centers[np.newaxis, :]
    
    diffs = x_col - c_row
    w2 = 2.0 * (widths[np.newaxis, :] ** 2)
    
    # Avoid division by zero if width is extremely small (exp(-inf) -> 0)
    # But exp(log_widths) is never 0.
    # If log_widths is -100, width is ~0. w2 is ~0.
    # diffs^2 / w2 -> large number. exp(-large) -> 0.
    # This is stable.
    
    exponents = -(diffs ** 2) / w2
    rbf_values = np.exp(exponents)
    
    # Linear combination
    y = rbf_values @ amplitudes
    
    return y
# EVOLVE-BLOCK-END
