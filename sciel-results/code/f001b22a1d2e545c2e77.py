"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    # Inner Strassen 2x2 patterns:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B10)
    # M7 = (A01 - A11) * (B01 + B11)
    
    # u_s: coefficients for A-side (4 entries: (0,0),(0,1),(1,0),(1,1))
    inner_u = [
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [-1, 0, 1, 0],  # M6: A10-A00
        [0, 1, 0, -1],  # M7: A01-A11
    ]
    
    inner_v = [
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, -1, 1, 0],  # M3: B10-B11
        [-1, 1, 0, 0],  # M4: B01-B00
        [0, 0, 0, 1],   # M5: B11
        [1, 0, 1, 0],   # M6: B00+B10
        [0, 1, 0, 1],   # M7: B01+B11
    ]
    
    inner_w = [
        [1, 0, 0, 0],   # M1 -> C00
        [0, 0, 1, 0],   # M2 -> C10
        [0, 1, 0, 0],   # M3 -> C01
        [1, 0, 0, 0],   # M4 -> C00
        [-1, 0, 0, 0],  # M5 -> -C00
        [0, 0, 0, 1],   # M6 -> C11
        [1, 0, 0, 0],   # M7 -> C00
    ]
    
    # Outer Strassen patterns for 4x4:
    outer_a = [
        {(0,0): 1, (1,1): 1},   # P1
        {(1,0): 1, (1,1): 1},   # P2
        {(0,0): 1},             # P3
        {(1,1): 1},             # P4
        {(0,0): 1, (0,1): 1},   # P5
        {(1,0): 1, (0,0): -1},  # P6
        {(0,1): 1, (1,1): -1},  # P7
    ]
    
    outer_b = [
        {(0,0): 1, (1,1): 1},   # P1
        {(0,0): 1},             # P2
        {(1,0): 1, (1,1): -1},  # P3
        {(0,1): 1, (0,0): -1},  # P4
        {(1,1): 1},             # P5
        {(0,0): 1, (1,0): 1},   # P6
        {(0,1): 1, (1,1): 1},   # P7
    ]
    
    outer_w = [
        {(0,0): 1, (1,1): 1},   # P1
        {(1,0): 1, (1,1): -1},  # P2
        {(0,1): 1, (1,1): 1},   # P3
        {(0,0): 1, (1,0): 1},   # P4
        {(0,0): -1, (0,1): 1},  # P5
        {(1,1): 1},             # P6
        {(0,0): 1},             # P7
    ]
    
    for ot in range(7):
        for st in range(7):
            u_idx = ot * 7 + st
            
            # Build U vector (16 elements)
            for (br, bc), a_coeff in outer_a[ot].items():
                for local_idx in range(4):
                    u_coeff = inner_u[st][local_idx]
                    if u_coeff != 0:
                        lr = local_idx // 2
                        lc = local_idx % 2
                        gr = br * 2 + lr
                        gc = bc * 2 + lc
                        U[gr * 4 + gc][u_idx] += a_coeff * u_coeff
            
            # Build V vector (16 elements)
            for (br, bc), b_coeff in outer_b[ot].items():
                for local_idx in range(4):
                    v_coeff = inner_v[st][local_idx]
                    if v_coeff != 0:
                        lr = local_idx // 2
                        lc = local_idx % 2
                        gr = br * 2 + lr
                        gc = bc * 2 + lc
                        V[gr * 4 + gc][u_idx] += b_coeff * v_coeff
            
            # Build W vector (16 elements)
            for (cr, cc), c_coeff in outer_w[ot].items():
                for local_idx in range(4):
                    w_coeff = inner_w[st][local_idx]
                    if w_coeff != 0:
                        lr = local_idx // 2
                        lc = local_idx % 2
                        gr = cr * 2 + lr
                        gc = cc * 2 + lc
                        W[gr * 4 + gc][u_idx] += c_coeff * w_coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
