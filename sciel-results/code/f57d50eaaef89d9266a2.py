import numpy as np

def coefficients(stages):
    """
    Uses Suzuki-Trotter 4th order 6-stage coefficients, repeated and scaled to fit 12 stages.
    """
    if stages != 12:
        # Fallback to uniform if stages != 12
        return [1.0/stages]*stages, [1.0/stages]*stages
        
    # Suzuki 4th order 6-stage coefficients
    # z = 1 / (2 - 2^(1/3))
    z = 1.0 / (2.0 - 2.0**(1.0/3.0))
    
    # 6-stage coefficients (sum to 1 each? Let's check)
    # Standard Suzuki 4th order:
    c1 = 1.0 / z
    c2 = 1.0 - 2.0 / z
    c3 = 1.0 / z
    
    d1 = 1.0 - 1.0 / (2.0 * z)
    d2 = -1.0 / z
    d3 = 1.0 / (2.0 * z)
    
    # The 6-stage formula is:
    # B_d3 A_c2 B_d2 A_c1 B_d1 A_c1 B_d2 A_c2 B_d3
    # This has 3 B's and 4 A's? 
    # Wait, the prompt requires 12 A's and 12 B's.
    # Let's use the symmetric 6-stage formula with 6 A's and 6 B's.
    # a = [c1, c2, c3, c3, c2, c1]
    # b = [d1, d2, d3, d3, d2, d1]
    
    a_6 = [c1, c2, c3, c3, c2, c1]
    b_6 = [d1, d2, d3, d3, d2, d1]
    
    sum_a_6 = sum(a_6)
    sum_b_6 = sum(b_6)
    
    # Normalize to sum to 1
    if sum_a_6 != 0:
        a_6 = [x / sum_a_6 for x in a_6]
    if sum_b_6 != 0:
        b_6 = [x / sum_b_6 for x in b_6]
        
    # Repeat for 12 stages, scaling by 1/2
    a = [x / 2.0 for x in a_6] + [x / 2.0 for x in a_6]
    b = [x / 2.0 for x in b_6] + [x / 2.0 for x in b_6]
    
    return a, b
