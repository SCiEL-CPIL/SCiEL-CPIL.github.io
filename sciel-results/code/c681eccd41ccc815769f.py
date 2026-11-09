import numpy as np
from scipy.special import roots_genlaguerre
import time

def construct():
    """
    Returns a list of 12 positive double-root locations for the Laguerre combination.
    Uses zeros of generalized Laguerre polynomial L_12^(alpha)(x) with alpha=11
    (near-uniform spacing) scaled to large magnitude to minimize the C4 upper bound.
    """
    t0 = time.time()
    deadline = 590.0  # Safety margin

    m = 12
    # alpha = m-1 gives the most uniform spacing among generalized Laguerre zeros
    # This structurally differs from standard Laguerre (alpha=0) which clusters near 0
    alpha = 11.0
    
    try:
        zeros = roots_genlaguerre(m, alpha)[0]
        zeros = np.sort(zeros)
    except Exception:
        # Fallback: arithmetic progression at large scale
        return [4.0 * 1000.0 * (i + 1) for i in range(12)]

    # Scale to large magnitude. Knowledge shows scale 1000 on arithmetic 
    # (max=48000) gave score ~33.3. Use same max for comparison.
    scale_factor = 48000.0 / zeros[-1]
    scaled_zeros = zeros * scale_factor

    if time.time() - t0 > deadline:
        return list(scaled_zeros)

    return list(scaled_zeros)
