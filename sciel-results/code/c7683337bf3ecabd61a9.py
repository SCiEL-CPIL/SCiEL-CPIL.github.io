"""Rewritten program using Strassen's algorithm for 2x2 blocks."""

def construct():
    n, m, p = 2, 2, 16
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (2x2)
    # s=0: M1=(A00+A11)(...) -> A00:1, A11:1
    # s=1: M2=(A10+A11)(...) -> A10:1, A11:1
    # s=2: M3=A00(...)      -> A00:1
    # s=3: M4=A11(...)      -> A11:1
    # s=4: M5=(A00+A01)(...) -> A00:1, A01:1
    # s=5: M6=(A10-A00)(...) -> A10:1, A00:-1
    # s=6: M7=(A01-A11)(...) -> A01:1, A11:-1
    
    u_coeffs = [
        {0: 1, 3: 1},  # s=0
        {2: 1, 3: 1},  # s=1
        {0: 1},        # s=2
        {3: 1},        # s=3
        {0: 1, 1: 1},  # s=4
        {2: 1, 0: -1}, # s=5
        {1: 1, 3: -1}  # s=6
    ]
    
    # Strassen coefficients for B (2x2)
    # s=0: M1=(...)(B00+B11) -> B00:1, B11:1
    # s=1: M2=(...)B00       -> B00:1
    # s=2: M3=(...)(B01-B11) -> B01:1, B11:-1
    # s=3: M4=(...)(B10-B00) -> B10:1, B00:-1
    # s=4: M5=(...)B11       -> B11:1
    # s=5: M6=(...)(B00+B01) -> B00:1, B01:1
    # s=6: M7=(...)(B10+B11) -> B10:1, B11:1
    
    v_coeffs = [
        {(0, 0): 1, (1, 1): 1}, # s=0
        {(0, 0): 1},            # s=1
        {(0, 1): 1, (1, 1): -1},# s=2
        {(1, 0): 1, (0, 0): -1},# s=3
        {(1, 1): 1},            # s=4
        {(0, 0): 1, (0, 1): 1}, # s=5
        {(1, 0): 1, (1, 1): 1}  # s=6
    ]
    
    # Strassen coefficients for C (2x2)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # w_coeffs[s] is a dict mapping (q, i) to coeff
    # q is row in 2x2 block (0,1), i is col in 2x2 block (0,1)
    w_coeffs = [
        {(0, 0): 1, (1, 1): 1},  # s=0 (M1)
        {(1, 0): 1, (1, 1): -1}, # s=1 (M2)
        {(0, 1): 1, (1, 1): 1},  # s=2 (M3)
        {(0, 0): 1, (1, 0): 1},  # s=3 (M4)
        {(0, 0): -1, (0, 1): 1}, # s=4 (M5)
        {(1, 1): 1},             # s=5 (M6)
        {(0, 0): 1}              # s=6 (M7)
    ]
    
    # Fill U
    for s in range(7):
        for i_j, coeff in u_coeffs[s].items():
            # i_j is index in A (0..3)
            for t in range(4):
                r = 7 * t + s
                U[i_j][r] += coeff
                
    # Fill V
    for s in range(7):
        for (j, q), coeff in v_coeffs[s].items():
            # j is row in B block (0,1), q is col in B block (0,1)
            # B global index: j * 16 + k, where k = 4*t + q
            for t in range(4):
                k = 4 * t + q
                idx_v = j * 16 + k
                r = 7 * t + s
                V[idx_v][r] += coeff
                
    # Fill W
    for s in range(7):
        for (q, i), coeff in w_coeffs[s].items():
            # q is row in C block (0,1), i is col in C block (0,1)
            # C global index: k * 2 + i, where k = 4*t + q
            for t in range(4):
                k = 4 * t + q
                idx_w = k * 2 + i
                r = 7 * t + s
                W[idx_w][r] += coeff
                
    return {"U": U, "V": V, "W": W}
