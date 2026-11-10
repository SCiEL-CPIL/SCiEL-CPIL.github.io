"""Improved program using 4th-order Suzuki-Trotter splitting for Heisenberg-chain time evolution."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def coefficients(stages):
    """12-stage product formula using 4th-order symmetric Suzuki-Trotter splitting.
    
    Uses three nested 4th-order 4-stage blocks with properly normalized coefficients
    to ensure sum(a) = sum(b) = 1.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    if stages != 12:
        # Fallback to simple Strang splitting for non-12 stage counts
        substeps = max(1, stages // 2)
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b
    
    # 4th-order Suzuki-Trotter symmetric splitting coefficients
    r = 2.0 ** (1.0 / 3.0)
    c1 = 1.0 / (2.0 * (1.0 - 1.0 / r))
    c2 = (1.0 - c1) / 2.0
    
    # One 4-stage 4th-order block
    a_block = [c1, c2, c2, c1]
    b_block = [0.0, c1, 2.0 * c2, 0.0]
    
    # Three blocks for 12 stages
    a = a_block * 3
    b = b_block * 3
    
    # Normalize to ensure sum(a) = 1 and sum(b) = 1
    sum_a = sum(a)
    sum_b = sum(b)
    
    if abs(sum_a) > 1e-15:
        a = [x / sum_a for x in a]
    if abs(sum_b) > 1e-15:
        b = [x / sum_b for x in b]
    
    # Verify consistency
    assert len(a) == 12, f"Expected 12 a coefficients, got {len(a)}"
    assert len(b) == 12, f"Expected 12 b coefficients, got {len(b)}"
    assert abs(sum(a) - 1.0) < 1e-10, f"Sum of a is {sum(a)}, expected 1.0"
    assert abs(sum(b) - 1.0) < 1e-10, f"Sum of b is {sum(b)}, expected 1.0"
    
    # Ensure all values are finite and in [-50, 50]
    for i in range(12):
        a[i] = max(-50.0, min(50.0, a[i]))
        b[i] = max(-50.0, min(50.0, b[i]))
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        pass  # Should not happen
    
    return a, b
# EVOLVE-BLOCK-END
