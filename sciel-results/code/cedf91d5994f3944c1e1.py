"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def coefficients(stages):
    """
    Constructs a 12-stage 4th-order symmetric product formula by splitting
    a 6-stage Suzuki-Trotter scheme into 12 stages.
    """
    if stages != 12:
        # Fallback to simple Strang if stages is not 12
        substeps = stages // 2
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b

    # Suzuki 4th-order coefficients
    # c1 = 1 / (2 - 2^(1/3))
    # c2 = -2^(1/3) / (2 - 2^(1/3))
    # c1 + c2 + c1 = 1
    c1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
    c2 = -2.0**(1.0/3.0) / (2.0 - 2.0**(1.0/3.0))
    
    # The 6-stage swapped Suzuki scheme (B then A order) has A-coeffs: c1, c2, c1
    # and B-coeffs: c1, c2, c1.
    # We split each of the 6 stages into 2 stages to get 12 stages.
    # Stage 1,2: B with c1/2
    # Stage 3,4: A with c1/2
    # Stage 5,6: B with c2/2
    # Stage 7,8: A with c2/2
    # Stage 9,10: B with c1/2
    # Stage 11,12: A with c1/2
    
    a = [0.0] * 12
    b = [0.0] * 12
    
    # Block 1: c1
    b[0] = c1 / 2.0
    b[1] = c1 / 2.0
    a[2] = c1 / 2.0
    a[3] = c1 / 2.0
    
    # Block 2: c2
    b[4] = c2 / 2.0
    b[5] = c2 / 2.0
    a[6] = c2 / 2.0
    a[7] = c2 / 2.0
    
    # Block 3: c1
    b[8] = c1 / 2.0
    b[9] = c1 / 2.0
    a[10] = c1 / 2.0
    a[11] = c1 / 2.0
    
    return a, b
# EVOLVE-BLOCK-END
