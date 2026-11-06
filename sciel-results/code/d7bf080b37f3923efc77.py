"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def coefficients(stages):
    """4th-order Suzuki-Trotter based 12-stage product formula."""
    # 4th-order Suzuki-Trotter coefficients
    c = 2**(1/3)
    denom = 2 - c
    c0 = 1/denom
    c1 = -c/denom
    c2 = 1/(2*denom)
    c3 = 1/(2*denom)
    
    # For 12 stages, use a symmetric pattern based on ST4
    # ST4: B(c3) A(c2) B(c1) A(c0) B(c1) A(c2) B(c3)
    # Map to 12 stages by repeating the 4-element pattern 3 times for each
    
    # A coefficients: repeat [c2, c0, c0, c2] three times, normalized
    a_raw = []
    for _ in range(3):
        a_raw += [c2, c0, c0, c2]
    a_sum_raw = sum(a_raw)
    a = [x / a_sum_raw for x in a_raw]
    
    # B coefficients: repeat [c3, c1, c1, c3] three times, normalized
    b_raw = []
    for _ in range(3):
        b_raw += [c3, c1, c1, c3]
    b_sum_raw = sum(b_raw)
    b = [x / b_sum_raw for x in b_raw]
    
    return a, b
# EVOLVE-BLOCK-END
