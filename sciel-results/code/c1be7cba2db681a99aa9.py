"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 590  # Safety margin before 600s limit
    # Use the 12 roots of the generalized Laguerre polynomial L_12^(0)(x)
    # These are the Gauss-Laguerre quadrature nodes, which minimize the
    # weighted L^2 norm and provide the physically consistent baseline
    # for the Cohn-Goncalves construction with m=12, n=1.
    try:
        roots = np.polynomial.laguerre.laggauss(12)[0]
        result = [float(r) for r in roots]
    except Exception:
        # Fallback to seed if numpy fails
        return [4.0 * (i + 1) for i in range(12)]
    if time.time() > deadline:
        return [4.0 * (i + 1) for i in range(12)]
    return result
# EVOLVE-BLOCK-END
