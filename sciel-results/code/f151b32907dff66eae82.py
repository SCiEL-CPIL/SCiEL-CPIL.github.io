import time

DIMENSIONS = [2, 2, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's algorithm for 2x2 matrices
    # We apply it to 3 blocks of columns (0-1, 2-3, 4-5)
    # Total rank = 3 blocks * 7 terms/block = 21
    
    # U coefficients for A (2x2)
    # Indices: 0:A00, 1:A01, 2:A10, 3:A11
    # Terms M1..M7 (indices 0..6 in the loop)
    U_terms = [
        {0: 1, 3: 1},       # M1: A00 + A11
        {2: 1, 3: 1},       # M2: A10 + A11
        {0: 1},             # M3: A00
        {3: 1},             # M4: A11
        {0: 1, 1: 1},       # M5: A00 + A01
        {2: 1, 0: -1},      # M6: A10 - A00
        {1: 1, 3: -1}       # M7: A01 - A11
    ]
    
    # V coefficients for B (2x2 block)
    # Local indices: j in {0,1}, k_local in {0,1}
    # Global V index: j * p + (2*block + k_local)
    # Terms correspond to M1..M7
    V_terms = [
        [(0, 0, 1), (1, 1, 1)],      # M1: B00 + B11
        [(0, 0, 1)],                 # M2: B00
        [(0, 1, 1), (1, 1, -1)],     # M3: B01 - B11
        [(1, 0, 1), (0, 0, -1)],     # M4: B10 - B00
        [(1, 1, 1)],                 # M5: B11
        [(0, 0, 1), (0, 1, 1)],      # M6: B00 + B01
        [(1, 0, 1), (1, 1, 1)]       # M7: B10 + B11
    ]
    
    # W coefficients for C (2x2 block)
    # Local indices: i in {0,1}, k_local in {0,1}
    # Global W index: (2*block + k_local) * n + i
    # Reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # W_map[t] contains list of (i, k_local, coeff) for term t (0..6)
    # t=0 (M1): C00(1), C11(1)
    # t=1 (M2): C10(1), C11(-1)
    # t=2 (M3): C01(1), C11(1)
    # t=3 (M4): C00(1), C10(1)
    # t=4 (M5): C00(-1), C01(1)
    # t=5 (M6): C11(1)
    # t=6 (M7): C00(1)
    W_map = [
        [(0, 0, 1), (1, 1, 1)],   # M1
        [(1, 0, 1), (1, 1, -1)],  # M2
        [(0, 1, 1), (1, 1, 1)],   # M3
        [(0, 0, 1), (1, 0, 1)],   # M4
        [(0, 0, -1), (0, 1, 1)],  # M5
        [(1, 1, 1)],              # M6
        [(0, 0, 1)]               # M7
    ]
    
    for block in range(3):
        for t in range(7):
            r = block * 7 + t
            
            # Fill U
            for idx, val in U_terms[t].items():
                U[idx][r] = val
                
            # Fill V
            for j_local, k_local, val in V_terms[t]:
                global_k = 2 * block + k_local
                v_idx = j_local * p + global_k
                V[v_idx][r] = val
                
            # Fill W
            for i, k_local, val in W_map[t]:
                global_k = 2 * block + k_local
                w_idx = global_k * n + i
                W[w_idx][r] = val
                
            # Time check (though this loop is extremely fast)
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
                
    return {"U": U, "V": V, "W": W}
