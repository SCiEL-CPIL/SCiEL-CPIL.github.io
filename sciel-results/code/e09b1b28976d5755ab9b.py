import numpy as np
from scipy.special import roots_laguerre

def construct():
    # The task specifies m=12, n=1.
    # The construction involves the square of the Laguerre polynomial L_12(x).
    # The roots of L_12(x) are the locations of the double roots.
    try:
        roots, _ = roots_laguerre(12)
        return list(roots)
    except Exception:
        # Fallback to seed if scipy fails for some reason
        return [4.0 * (i + 1) for i in range(12)]
