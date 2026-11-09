"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    
    # Deadline guard: track elapsed time
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    # 10 basis functions: 2 global linear trend + 8 localized Gaussian RBFs
    # Centers spread across [-3, 3] with varying widths to capture local features
    centers = np.array([-2.5, -1.5, -0.75, 0.0, 0.75, 1.5, 2.5, 3.0])
    widths = np.array([0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.5, 0.4])
    
    # Global linear trend (params[0], params[1])
    p0 = np.ones_like(x)
    p1 = x
    
    # Localized Gaussian RBFs (params[2] through params[9])
    p2 = np.exp(-((x - centers[0])**2) / (2.0 * widths[0]**2))
    p3 = np.exp(-((x - centers[1])**2) / (2.0 * widths[1]**2))
    p4 = np.exp(-((x - centers[2])**2) / (2.0 * widths[2]**2))
    p5 = np.exp(-((x - centers[3])**2) / (2.0 * widths[3]**2))
    p6 = np.exp(-((x - centers[4])**2) / (2.0 * widths[4]**2))
    p7 = np.exp(-((x - centers[5])**2) / (2.0 * widths[5]**2))
    p8 = np.exp(-((x - centers[6])**2) / (2.0 * widths[6]**2))
    p9 = np.exp(-((x - centers[7])**2) / (2.0 * widths[7]**2))
    
    result = (params[0] * p0 + params[1] * p1 + params[2] * p2 +
              params[3] * p3 + params[4] * p4 + params[5] * p5 +
              params[6] * p6 + params[7] * p7 + params[8] * p8 +
              params[9] * p9)
    
    # Check for non-finite values
    if not np.all(np.isfinite(result)):
        # Return zeros as fallback
        return np.zeros_like(x)
    
    return result
# EVOLVE-BLOCK-END
