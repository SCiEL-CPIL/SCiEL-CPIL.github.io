"""Product-formula design for Heisenberg-chain time evolution."""
# EVOLVE-BLOCK-START
def coefficients(stages):
    """
    6th-order symmetric Suzuki-Trotter product formula for 12 stages.
    
    This uses the recursive construction S_3 = S_2(alpha) S_2(beta) S_2(alpha).
    S_2 is the 4th-order 4-stage scheme.
    The total number of stages is 3 * 4 = 12.
    """
    if stages != 12:
        # Fallback to simple Strang if stages is not 12, though task specifies 12
        substeps = stages // 2
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b

    # Coefficients for 4th-order 4-stage scheme S_2
    # alpha_2 = (2 - 2^(1/3))^-1
    alpha_2 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    beta_2 = 1.0 - 2.0 * alpha_2
    
    # a_2 and b_2 for one S_2 block (4 stages)
    # Order: B1 A1 B2 A2 B3 A3 B4 A4
    # Mapped to A B A B A B A by setting b1=0
    a_2 = [
        alpha_2 / 2.0,
        (alpha_2 + beta_2) / 2.0,
        (alpha_2 + beta_2) / 2.0,
        alpha_2 / 2.0
    ]
    b_2 = [
        0.0,
        alpha_2,
        beta_2,
        alpha_2
    ]
    
    # Coefficients for 6th-order 12-stage scheme S_3
    # alpha_3 = (2 - 2^(1/5))^-1
    alpha_3 = 1.0 / (2.0 - 2.0 ** (1.0 / 5.0))
    beta_3 = 1.0 - 2.0 * alpha_3
    
    a = []
    b = []
    
    # Block 1: S_2 scaled by alpha_3
    for i in range(4):
        a.append(a_2[i] * alpha_3)
        b.append(b_2[i] * alpha_3)
        
    # Block 2: S_2 scaled by beta_3
    for i in range(4):
        a.append(a_2[i] * beta_3)
        b.append(b_2[i] * beta_3)
        
    # Block 3: S_2 scaled by alpha_3
    for i in range(4):
        a.append(a_2[i] * alpha_3)
        b.append(b_2[i] * alpha_3)
        
    return a, b
# EVOLVE-BLOCK-END
