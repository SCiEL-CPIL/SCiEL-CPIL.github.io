def construct():
    import time
    _t0 = time.time()
    _DEADLINE = 140.0

    n, m, p = 3, 4, 6
    terms = []

    def add_3x2x3_block(i_offset, j_offset, k_offset):
        """Add 3x2x3 decomposition for A[i_offset:i_offset+3, j_offset:j_offset+2] @ B[j_offset:j_offset+2, k_offset:k_offset+3] -> C[i_offset:i_offset+3, k_offset:k_offset+3]"""
        # A indices: (i, j) where i in [i_offset, i_offset+2], j in [j_offset, j_offset+1]
        # B indices: (j, k) where j in [j_offset, j_offset+1], k in [k_offset, k_offset+2]
        # C indices: (i, k) where i in [i_offset, i_offset+2], k in [k_offset, k_offset+2]
        
        A00 = (i_offset, j_offset)
        A01 = (i_offset, j_offset+1)
        A10 = (i_offset+1, j_offset)
        A11 = (i_offset+1, j_offset+1)
        A20 = (i_offset+2, j_offset)
        A21 = (i_offset+2, j_offset+1)
        
        B00 = (j_offset, k_offset)
        B01 = (j_offset, k_offset+1)
        B02 = (j_offset, k_offset+2)
        B10 = (j_offset+1, k_offset)
        B11 = (j_offset+1, k_offset+1)
        B12 = (j_offset+1, k_offset+2)
        
        C00 = (i_offset, k_offset)
        C01 = (i_offset, k_offset+1)
        C02 = (i_offset, k_offset+2)
        C10 = (i_offset+1, k_offset)
        C11 = (i_offset+1, k_offset+1)
        C12 = (i_offset+1, k_offset+2)
        C20 = (i_offset+2, k_offset)
        C21 = (i_offset+2, k_offset+1)
        C22 = (i_offset+2, k_offset+2)
        
        # Strassen on A[0:2,:] and B[:,0:2]
        # M1 = (A00+A11)(B00+B11)
        terms.append(({A00: 1, A11: 1}, {B00: 1, B11: 1}, {C00: 1, C11: 1}))
        # M2 = (A10+A11)*B00
        terms.append(({A10: 1, A11: 1}, {B00: 1}, {C10: 1, C11: -1}))
        # M3 = A00*(B11-B01)
        terms.append(({A00: 1}, {B11: 1, B01: -1}, {C01: 1, C11: 1}))
        # M4 = A11*(B10-B00)
        terms.append(({A11: 1}, {B10: 1, B00: -1}, {C00: 1, C10: 1}))
        # M5 = (A00+A01)*B11
        terms.append(({A00: 1, A01: 1}, {B11: 1}, {C00: -1, C01: 1}))
        # M6 = (A10-A00)*(B00+B01)
        terms.append(({A10: 1, A00: -1}, {B00: 1, B01: 1}, {C11: 1}))
        # M7 = (A01-A11)*(B10+B11)
        terms.append(({A01: 1, A11: -1}, {B10: 1, B11: 1}, {C00: 1}))
        
        # Schoolbook for A[0:2,:] and B[:,2]
        terms.append(({A00: 1}, {B02: 1}, {C02: 1}))
        terms.append(({A01: 1}, {B12: 1}, {C02: 1}))
        terms.append(({A10: 1}, {B02: 1}, {C12: 1}))
        terms.append(({A11: 1}, {B12: 1}, {C12: 1}))
        
        # Schoolbook for A[2,:] and B[:,0:3]
        terms.append(({A20: 1}, {B00: 1}, {C20: 1}))
        terms.append(({A21: 1}, {B10: 1}, {C20: 1}))
        terms.append(({A20: 1}, {B01: 1}, {C21: 1}))
        terms.append(({A21: 1}, {B11: 1}, {C21: 1}))
        terms.append(({A20: 1}, {B02: 1}, {C22: 1}))
        terms.append(({A21: 1}, {B12: 1}, {C22: 1}))

    # Block 1: A[:, 0:2] @ B[0:3, :] -> C[:, 0:3]
    add_3x2x3_block(0, 0, 0)
    # Block 2: A[:, 2:4] @ B[3:6, :] -> C[:, 3:6]
    add_3x2x3_block(0, 2, 3)

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for col, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][col] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][col] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i][col] = coeff

    if time.time() - _t0 > _DEADLINE:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
