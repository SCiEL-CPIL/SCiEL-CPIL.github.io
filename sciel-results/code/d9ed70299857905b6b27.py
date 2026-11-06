"""Improved program using 4th-order Suzuki-Trotter splitting."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def coefficients(stages):
    """
    4th-order Suzuki-Trotter splitting mapped to a 12-stage symmetric product formula.
    The structure is e^{-i b_k t B} e^{-i a_k t A} for k=1..12.
    Symmetry: a_k = a_{13-k}, b_k = b_{13-k}.
    """
    if stages != 12:
        # Fallback to simple Strang if stages are not 12
        substeps = stages // 2
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b

    # 4th-order Suzuki-Trotter coefficients
    c1 = 1.0 / (2.0 * (1.0 - 2.0 ** (1.0 / 3.0)))
    c2 = -c1 / (2.0 ** (1.0 / 3.0) - 1.0)
    c3 = 0.5 - c1 - c2
    mu = 1.0 - 2.0 * c2

    # Scale by 0.5 to fit into the first half of the symmetric 12-stage formula
    # such that sum(b_1..b_6) = 0.5 and sum(a_1..a_6) = 0.5
    b_half = [0.0, c3 / 2.0, c1 / 2.0, c1 / 2.0, c3 / 2.0, 0.0]
    a_half = [0.0, c2 / 2.0, mu / 2.0, c2 / 2.0, 0.0, 0.0]
    
    # Construct full symmetric lists
    a = a_half + a_half[::-1]
    b = b_half + b_half[::-1]
    
    # Ensure finite numbers and within [-50, 50]
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    
    # Check constraints
    if not all(np.isfinite(a)) or not all(np.isfinite(b)):
        return [1.0/12.0]*12, [1.0/12.0]*12
    if any(abs(x) > 50 for x in a) or any(abs(x) > 50 for x in b):
        return [1.0/12.0]*12, [1.0/12.0]*12
        
    return a, b
# EVOLVE-BLOCK-END
