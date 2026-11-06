"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def coefficients(stages):
    """
    4th-order symmetric Suzuki-Trotter splitting mapped to 12 stages.
    
    The 4th-order scheme S_4(t) = S_2(c1*t) S_2(c2*t) S_2(c3*t) 
    expands to 9 exponentials:
    exp(-i c1/2 t B) exp(-i c1 t A) exp(-i (c1+c2)/2 t B) exp(-i c2 t A) 
    exp(-i (c2+c3)/2 t B) exp(-i c3 t A) exp(-i c3/2 t B)
    
    We map these to the first 4 (B, A) pairs and pad the rest with zeros.
    """
    if stages != 12:
        # Fallback to seed behavior if stages != 12, though task implies 12
        substeps = max(1, stages // 2)
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b

    # 4th-order Suzuki-Trotter coefficients
    # c1 = c3 = 1 / (2 - 2^(1/3))
    # c2 = -2^(1/3) / (2 - 2^(1/3))
    c1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
    c2 = -2.0**(1.0/3.0) / (2.0 - 2.0**(1.0/3.0))
    c3 = c1

    # Calculate merged B coefficients
    b1 = c1 / 2.0
    b2 = (c1 + c2) / 2.0
    b3 = (c2 + c3) / 2.0
    b4 = c3 / 2.0
    
    # A coefficients
    a1 = c1
    a2 = c2
    a3 = c3
    a4 = 0.0

    # Initialize lists
    a = [0.0] * stages
    b = [0.0] * stages

    # Fill the first 4 stages
    a[0] = a1
    b[0] = b1
    
    a[1] = a2
    b[1] = b2
    
    a[2] = a3
    b[2] = b3
    
    a[3] = a4
    b[3] = b4

    # The rest remain 0.0
    
    # Verification (optional, for debugging)
    # assert abs(sum(a) - 1.0) < 1e-15
    # assert abs(sum(b) - 1.0) < 1e-15

    return a, b
# EVOLVE-BLOCK-END
