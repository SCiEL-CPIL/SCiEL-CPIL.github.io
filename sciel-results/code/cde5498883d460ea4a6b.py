"""Product-formula design for Heisenberg-chain time evolution using high-order Suzuki-Trotter."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def coefficients(stages):
    """
    4th-order Suzuki-Trotter product formula for 12 stages.
    
    Uses 3 substeps of the 4th-order Yoshida decomposition.
    The 4th-order coefficients are:
      c1 = 1/(2 - 2^(1/3))
      c2 = -2^(1/3)/(2 - 2^(1/3))
    
    For 12 stages (6 A-factors and 6 B-factors per substep, 3 substeps),
    we use a symmetric pattern that approximates the 4th-order scheme.
    
    The scheme: U(t) = prod_{k=1}^{12} exp(-i b_k t B) exp(-i a_k t A)
    
    We use a symmetric coefficient pattern:
      a = [c1/6, c2/6, c1/6, c2/6, c1/6, c2/6, c2/6, c1/6, c2/6, c1/6, c2/6, c1/6]
      b = same pattern
    scaled so that sum(a) = sum(b) = 1.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # 4th-order Suzuki-Trotter coefficients
    c1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    c2 = -2.0 ** (1.0 / 3.0) / (2.0 - 2.0 ** (1.0 / 3.0))
    
    # For 12 stages, use a symmetric pattern based on 4th-order coefficients
    # Pattern: c1, c2, c1, c2, c1, c2, c2, c1, c2, c1, c2, c1 (symmetric)
    n = stages
    if n == 12:
        pattern = [c1, c2, c1, c2, c1, c2, c2, c1, c2, c1, c2, c1]
    else:
        # Fallback: uniform split
        pattern = [1.0 / n] * n
    
    # Scale so that sum = 1
    s = sum(pattern)
    if abs(s) > 1e-15:
        a = [x / s for x in pattern]
        b = [x / s for x in pattern]
    else:
        a = [1.0 / n] * n
        b = [1.0 / n] * n
    
    # Clip to [-50, 50] and re-normalize
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]
    
    sa = sum(a)
    sb = sum(b)
    if abs(sa) > 1e-15:
        a = [x / sa for x in a]
    if abs(sb) > 1e-15:
        b = [x / sb for x in b]
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return a, b
    
    return a, b
# EVOLVE-BLOCK-END
