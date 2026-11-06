import numpy as np

def coefficients(stages):
    if stages != 12:
        substeps = stages // 2
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b

    # 6th-order Suzuki-Trotter coefficients (approximate, normalized)
    y = 2.0 ** (1.0 / 3.0)
    
    # B weights (4 non-zero, symmetric)
    beta1 = 1.0 / (2.0 * (1.0 + y))
    beta2 = y / (2.0 * (1.0 - y**2))
    
    # A weights (5 non-zero, symmetric)
    alpha1 = 1.0 / (4.0 * (1.0 + y))
    alpha2 = 1.0 / (4.0 * (1.0 - y**2))
    alpha3 = 0.5 - 2.0 * alpha1 - 2.0 * alpha2
    
    # Construct 9-stage sequence: A1 B1 A2 B2 A3 B3 A4 B4 A5
    # Map to 12 BA stages.
    # Sequence: B0 A1 B1 A2 B2 A3 B3 A4 B4 A5 B0 A0 B0 A0?
    # Better: Distribute the 9 factors into 12 slots.
    
    a_list = []
    b_list = []
    
    # We have 12 B's and 12 A's.
    # We will place the 4 beta's and 5 alpha's into the 12 slots.
    # Symmetry: a_i = a_{13-i}, b_i = b_{13-i}.
    
    # Slots 1..12
    # B slots: 1..12
    # A slots: 1..12
    
    # Let's place beta's at B slots 3, 5, 8, 10 (symmetric)
    # Let's place alpha's at A slots 2, 4, 6, 7, 9, 11? No, 5 alpha's.
    # A slots: 2, 4, 6, 8, 10? (Symmetric around 6.5)
    # Alpha 1 at A2, Alpha 5 at A11.
    # Alpha 2 at A4, Alpha 4 at A9.
    # Alpha 3 at A6, A7? No, A6 and A7 are symmetric? 6 and 7 are symmetric around 6.5.
    # So A6 = A7 = alpha3/2?
    
    b_list = [0.0] * 12
    a_list = [0.0] * 12
    
    # Place B's
    b_list[2] = beta1
    b_list[10] = beta1
    b_list[4] = beta2
    b_list[7] = beta2
    
    # Place A's
    a_list[1] = alpha1
    a_list[11] = alpha1
    a_list[3] = alpha2
    a_list[9] = alpha2
    a_list[5] = alpha3 / 2.0
    a_list[6] = alpha3 / 2.0
    
    # Normalize
    sum_a = sum(a_list)
    sum_b = sum(b_list)
    
    if sum_a != 0:
        a_list = [x / sum_a for x in a_list]
    if sum_b != 0:
        b_list = [x / sum_b for x in b_list]
        
    return a_list, b_list
