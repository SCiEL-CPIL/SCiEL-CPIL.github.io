import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    deadline = time.time() + 100
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(U_entry, V_entry, W_entry):
        U_col = [0] * (n * m)
        for idx, val in U_entry.items():
            U_col[idx] = val
        V_col = [0] * (m * p)
        for idx, val in V_entry.items():
            V_col[idx] = val
        W_col = [0] * (p * n)
        for idx, val in W_entry.items():
            W_col[idx] = val
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)
        
    # Strassen terms for 2x2 x 2x2
    # A indices: 0:00, 1:01, 2:10, 3:11
    # B indices: 0:00, 1:01, 2:10, 3:11
    # C indices: 0:00, 1:01, 2:10, 3:11
    
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),       # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),             # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),             # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),             # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 2: 1}),             # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),             # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})              # M7
    ]
    
    # Part 1: A[0:2, 0:2] * B[0:2, 0:8]
    # B[0:2, 0:8] is 2x8. Split into four 2x2 blocks.
    for block_idx in range(4):
        if time.time() > deadline:
            break
        # B block starts at col block_idx * 2
        b_offset = block_idx * 2
        # A is fixed 2x2 (cols 0,1)
        # U indices are 0..3
        # V indices are (0 * 9 + b_offset) .. (0 * 9 + b_offset + 1) and (1 * 9 + b_offset) .. (1 * 9 + b_offset + 1)
        # Wait, B is 3x9. B[0:2, 0:8] uses rows 0,1 of B.
        # V index for B[j, k] is j * p + k.
        # Here j in {0, 1}, k in {b_offset, b_offset+1}.
        
        for u_ent, v_ent, w_ent in strassen_terms:
            # Map V indices
            v_mapped = {}
            for idx, val in v_ent.items():
                # idx 0 -> B[0, b_offset] -> 0 * 9 + b_offset
                # idx 1 -> B[0, b_offset+1] -> 0 * 9 + b_offset + 1
                # idx 2 -> B[1, b_offset] -> 1 * 9 + b_offset
                # idx 3 -> B[1, b_offset+1] -> 1 * 9 + b_offset + 1
                if idx == 0:
                    v_mapped[0 * p + b_offset] = val
                elif idx == 1:
                    v_mapped[0 * p + b_offset + 1] = val
                elif idx == 2:
                    v_mapped[1 * p + b_offset] = val
                elif idx == 3:
                    v_mapped[1 * p + b_offset + 1] = val
            
            # Map W indices
            # W index for C[i, k] is k * n + i.
            # C is 2x8 for this part. k in {b_offset, b_offset+1}. i in {0, 1}.
            # w_ent keys 0..3 correspond to C[0,0], C[0,1], C[1,0], C[1,1] relative to block.
            w_mapped = {}
            for idx, val in w_ent.items():
                if idx == 0: # C[0, b_offset]
                    w_mapped[b_offset * n + 0] = val
                elif idx == 1: # C[0, b_offset+1]
                    w_mapped[(b_offset + 1) * n + 0] = val
                elif idx == 2: # C[1, b_offset]
                    w_mapped[b_offset * n + 1] = val
                elif idx == 3: # C[1, b_offset+1]
                    w_mapped[(b_offset + 1) * n + 1] = val
            
            add_term(u_ent, v_mapped, w_mapped)
            
    # Part 2: A[0:2, 2] * B[2, 0:9]
    # A[0:2, 2] is 2x1. B[2, 0:9] is 1x9.
    # C[0:2, 0:9] = A[0:2, 2] * B[2, 0:9]
    # C[i, k] = A[i, 2] * B[2, k]
    # U index for A[i, 2] is i * m + 2 = i * 3 + 2.
    # V index for B[2, k] is 2 * p + k = 18 + k.
    # W index for C[i, k] is k * n + i = 2 * i + k? No, k * n + i = k * 2 + i.
    
    for i in range(n):
        for k in range(p):
            if time.time() > deadline:
                break
            add_term(
                {i * m + 2: 1},
                {2 * p + k: 1},
                {k * n + i: 1}
            )
            
    rank = len(U_cols)
    U = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V = [[V_cols[r][j] for r in range(rank)] for j in range(m*p)]
    W = [[W_cols[r][k] for r in range(rank)] for k in range(p*n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
