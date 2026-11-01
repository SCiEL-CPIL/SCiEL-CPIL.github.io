import numpy as np

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation
    # 6 blocks of 2 columns use Strassen (rank 7)
    # 1 block of 1 column uses standard (rank 4)
    num_full_blocks = p // 2
    num_remainder = p % 2
    rank = num_full_blocks * 7 + num_remainder * 4
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_vec, v_vec, w_vec):
        nonlocal col
        if col >= rank:
            return
        for i in range(len(u_vec)):
            if u_vec[i] != 0:
                U[i][col] = u_vec[i]
        for i in range(len(v_vec)):
            if v_vec[i] != 0:
                V[i][col] = v_vec[i]
        for i in range(len(w_vec)):
            if w_vec[i] != 0:
                W[i][col] = w_vec[i]
        col += 1

    # Strassen terms for 2x2x2 block
    # u_vec size 4 (A00, A01, A10, A11)
    # v_vec size 4 (B00, B01, B10, B11) relative to block
    # w_vec size 4 (C00, C01, C10, C11) relative to block
    
    strassen_terms = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1])
    ]
    
    # Process full blocks
    for b in range(num_full_blocks):
        k1 = 2 * b
        k2 = 2 * b + 1
        
        # Map local indices to global
        # A is same for all
        # B indices: j*p + k
        # C indices: k*n + i
        
        for u_local, v_local, w_local in strassen_terms:
            u_global = [0] * (n*m)
            v_global = [0] * (m*p)
            w_global = [0] * (p*n)
            
            # Fill U
            for idx, val in enumerate(u_local):
                if val != 0:
                    u_global[idx] = val
            
            # Fill V
            # v_local[0] -> B00 -> B[0*p + k1]
            # v_local[1] -> B01 -> B[0*p + k2]
            # v_local[2] -> B10 -> B[1*p + k1]
            # v_local[3] -> B11 -> B[1*p + k2]
            v_global[0 * p + k1] = v_local[0]
            v_global[0 * p + k2] = v_local[1]
            v_global[1 * p + k1] = v_local[2]
            v_global[1 * p + k2] = v_local[3]
            
            # Fill W
            # w_local[0] -> C00 -> C[k1*n + 0]
            # w_local[1] -> C01 -> C[k1*n + 1]
            # w_local[2] -> C10 -> C[k2*n + 0]
            # w_local[3] -> C11 -> C[k2*n + 1]
            w_global[k1 * n + 0] = w_local[0]
            w_global[k1 * n + 1] = w_local[1]
            w_global[k2 * n + 0] = w_local[2]
            w_global[k2 * n + 1] = w_local[3]
            
            add_term(u_global, v_global, w_global)

    # Process remainder (1 column)
    if num_remainder > 0:
        k1 = 2 * num_full_blocks
        
        # Standard 2x2x1 decomposition
        # C00 = A00 B0 + A01 B1
        # C01 = A10 B0 + A11 B1
        # Terms:
        # 1. A00 B0 -> C00
        # 2. A01 B1 -> C00
        # 3. A10 B0 -> C01
        # 4. A11 B1 -> C01
        
        std_terms = [
            ([1, 0, 0, 0], [1, 0], [1, 0]), # A00 B0 -> C00
            ([0, 1, 0, 0], [0, 1], [1, 0]), # A01 B1 -> C00
            ([0, 0, 1, 0], [1, 0], [0, 1]), # A10 B0 -> C01
            ([0, 0, 0, 1], [0, 1], [0, 1])  # A11 B1 -> C01
        ]
        
        for u_local, v_local, w_local in std_terms:
            u_global = [0] * (n*m)
            v_global = [0] * (m*p)
            w_global = [0] * (p*n)
            
            for idx, val in enumerate(u_local):
                if val != 0:
                    u_global[idx] = val
            
            # v_local[0] -> B0 -> B[0*p + k1]
            # v_local[1] -> B1 -> B[1*p + k1]
            v_global[0 * p + k1] = v_local[0]
            v_global[1 * p + k1] = v_local[1]
            
            # w_local[0] -> C0 -> C[k1*n + 0]
            # w_local[1] -> C1 -> C[k1*n + 1]
            w_global[k1 * n + 0] = w_local[0]
            w_global[k1 * n + 1] = w_local[1]
            
            add_term(u_global, v_global, w_global)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
