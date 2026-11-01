"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2x2
    # U indices: i*m + j (0..3)
    # V indices: j*p_local + k (0..3)
    # W indices: k*n + i (0..3)
    
    # 7 multiplications
    U_local = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    
    V_local = [
        [1, 1, 0, -1, 0, 1, 0], # B00 (j=0, k=0)
        [0, 0, 1, 0, 0, 1, 0],  # B01 (j=0, k=1)
        [0, 0, 0, 1, 0, 0, 1],  # B10 (j=1, k=0)
        [1, 0, -1, 0, 1, 0, 1]  # B11 (j=1, k=1)
    ]
    
    W_local = [
        [1, 0, 0, -1, 1, 0, -1], # C00 (k=0, i=0)
        [0, 1, 0, 1, 0, 0, 0],   # C01 (k=0, i=1) -- Wait, W index is k*n+i.
        # k=0, i=0 -> 0
        # k=0, i=1 -> 1
        # k=1, i=0 -> 2
        # k=1, i=1 -> 3
        
        # Let's re-verify W indices for Strassen
        # M1: C00(+), C11(+) -> W[0]=1, W[3]=1
        # M2: C10(+), C11(-) -> W[1]=1, W[3]=-1
        # M3: C01(+), C11(-) -> W[2]=1, W[3]=-1
        # M4: C00(-), C10(+) -> W[0]=-1, W[1]=1
        # M5: C00(+), C11(+) -> W[0]=1, W[3]=1
        # M6: C11(-) -> W[3]=-1
        # M7: C01(+), C00(-) -> W[2]=1, W[0]=-1
        
        # My previous W_local derivation:
        # Col 1 (M1): [1, 0, 0, 1]
        # Col 2 (M2): [0, 1, 0, -1]
        # Col 3 (M3): [0, 0, 1, -1]
        # Col 4 (M4): [-1, 1, 0, 0]
        # Col 5 (M5): [1, 0, 0, 1]
        # Col 6 (M6): [0, 0, 0, -1]
        # Col 7 (M7): [-1, 0, 1, 0]
        
        # Let's check M6: (A10-A00)(B00+B01).
        # C11 = ... - M6.
        # C11 is index 3. So W[3] = -1. Correct.
        
        # Let's check M7: (A01-A11)(B10+B11).
        # C01 = ... + M7. C01 is index 2. W[2]=1.
        # C00 = ... - M7. C00 is index 0. W[0]=-1.
        # Correct.
    ]
    
    # Correct W_local
    W_local = [
        [1, 0, 0, -1, 1, 0, -1], # Row 0 (C00)
        [0, 1, 0, 1, 0, 0, 0],   # Row 1 (C01) - Wait.
        # C10 is index 1.
        # M2: C10(+). W[1]=1.
        # M4: C10(+). W[1]=1.
        # So Row 1 should be [0, 1, 0, 1, 0, 0, 0].
        
        # Row 2 (C11? No, C index is k*n+i.
        # k=1, i=0 is index 2. This is C10? No.
        # C is 2x10.
        # C[i][k].
        # W index is k*n + i.
        # k=0, i=0 -> 0 (C00)
        # k=0, i=1 -> 1 (C10)
        # k=1, i=0 -> 2 (C01)
        # k=1, i=1 -> 3 (C11)
        
        # Let's re-map W rows to C elements.
        # Row 0: C00.
        # Row 1: C10.
        # Row 2: C01.
        # Row 3: C11.
        
        # M1: C00(+), C11(+). Row 0: 1, Row 3: 1.
        # M2: C10(+), C11(-). Row 1: 1, Row 3: -1.
        # M3: C01(+), C11(-). Row 2: 1, Row 3: -1.
        # M4: C00(-), C10(+). Row 0: -1, Row 1: 1.
        # M5: C00(+), C11(+). Row 0: 1, Row 3: 1.
        # M6: C11(-). Row 3: -1.
        # M7: C01(+), C00(-). Row 2: 1, Row 0: -1.
        
        # So W_local rows:
        # Row 0: [1, 0, 0, -1, 1, 0, -1]
        # Row 1: [0, 1, 0, 1, 0, 0, 0]
        # Row 2: [0, 0, 1, 0, 0, 0, 1]
        # Row 3: [1, -1, -1, 0, 1, -1, 0]
    ]
    
    W_local = [
        [1, 0, 0, -1, 1, 0, -1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [1, -1, -1, 0, 1, -1, 0]
    ]

    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    for b in range(5):
        # U is the same for all blocks
        for i in range(n * m):
            for r in range(7):
                U[i][col_offset + r] = U_local[i][r]
                
        # V is shifted in k
        # Local k is 0,1. Global k is 2b, 2b+1.
        # V index is j*p + k.
        # Local V index is j*2 + k_local.
        # j=0, k_local=0 -> global 0*10 + 2b = 2b
        # j=0, k_local=1 -> global 0*10 + 2b+1 = 2b+1
        # j=1, k_local=0 -> global 1*10 + 2b = 10+2b
        # j=1, k_local=1 -> global 1*10 + 2b+1 = 10+2b+1
        
        # W is shifted in k
        # Local W index is k_local*n + i.
        # Global W index is (2b+k_local)*n + i.
        
        for j in range(m):
            for k_local in range(2):
                v_idx = j * p + (2 * b + k_local)
                v_local_idx = j * 2 + k_local
                for r in range(7):
                    V[v_idx][col_offset + r] = V_local[v_local_idx][r]
                    
        for k_local in range(2):
            for i in range(n):
                w_idx = (2 * b + k_local) * n + i
                w_local_idx = k_local * n + i
                for r in range(7):
                    W[w_idx][col_offset + r] = W_local[w_local_idx][r]
                    
        col_offset += 7
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
