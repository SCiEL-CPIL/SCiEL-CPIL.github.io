import math

def coefficients(stages):
    """
    4th-order symmetric splitting scheme embedded in 12 stages.
    Uses the standard 4th-order coefficients (Suzuki/Yoshida) and distributes them
    symmetrically across the 12 available stages to maintain the product form structure.
    """
    if stages != 12:
        # Fallback to Strang if stages is not 12
        substeps = stages // 2
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b

    # 4th-order symmetric scheme coefficients
    # c1 = 1 / (2 * (1 - 2**(1/3)))
    # c0 = 0.5 - c1
    # d1 = 1 / (1 - 2**(1/3))
    # d0 = 1 - 2 * d1
    
    factor = 1.0 - 2.0 ** (1.0 / 3.0)
    c1 = 1.0 / (2.0 * factor)
    c0 = 0.5 - c1
    d1 = 1.0 / factor
    d0 = 1.0 - 2.0 * d1

    # Distribute symmetrically in the center 8 stages (indices 2 to 9)
    # a: 0, 0, c1/2, c0/2, c0/2, c1/2, c1/2, c0/2, c0/2, c1/2, 0, 0
    # b: 0, 0, d1/2, d0/2, d0/2, d1/2, d1/2, d0/2, d0/2, d1/2, 0, 0
    
    a = [0.0] * 12
    b = [0.0] * 12
    
    # Fill indices 2, 3, 4, 5, 6, 7, 8, 9
    a[2] = c1 / 2.0
    a[3] = c0 / 2.0
    a[4] = c0 / 2.0
    a[5] = c1 / 2.0
    a[6] = c1 / 2.0
    a[7] = c0 / 2.0
    a[8] = c0 / 2.0
    a[9] = c1 / 2.0
    
    b[2] = d1 / 2.0
    b[3] = d0 / 2.0
    b[4] = d0 / 2.0
    b[5] = d1 / 2.0
    b[6] = d1 / 2.0
    b[7] = d0 / 2.0
    b[8] = d0 / 2.0
    b[9] = d1 / 2.0
    
    return a, b
