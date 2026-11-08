"""Matrix multiplication 2,4,5 — rank-minimal tensor decomposition."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 36
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for M_{2,2,2}
    # u, v, w are local vectors
    # u: 4 components (a00, a01, a10, a11)
    # v: 4 components (b00, b01, b10, b11)
    # w: 4 components (c00, c01, c10, c11)
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    # Naive terms for M_{2,2,1}
    # u: 4 components (a00, a01, a10, a11)
    # v: 2 components (b00, b10)
    # w: 2 components (c00, c10)
    naive_terms = [
        ([1, 0, 0, 0], [1, 0], [1, 0]),
        ([0, 1, 0, 0], [0, 1], [1, 0]),
        ([0, 0, 1, 0], [1, 0], [0, 1]),
        ([0, 0, 0, 1], [0, 1], [0, 1])
    ]
    
    col = 0
    # Two blocks: Block 1 (A cols 0-1, B rows 0-1), Block 2 (A cols 2-3, B rows 2-3)
    for block in range(2):
        a_offset = block * 2
        b_offset = block * 2
        
        # 7 Strassen terms for B sub-block 1 (local k 0,1)
        for u, v, w in strassen_terms:
            # Map u to U
            for li in range(4):
                i, lj = divmod(li, 2)
                gj = lj + a_offset
                U[i * m + gj][col] += u[li]
            
            # Map v to V
            # v is for B (2x2), local j 0,1, local k 0,1
            for lj in range(2):
                for lk in range(2):
                    li = lj * 2 + lk
                    gj = lj + b_offset
                    gk = lk # local k 0,1 maps to global k 0,1
                    V[gj * p + gk][col] += v[li]
            
            # Map w to W
            # w is for C (2x2), local i 0,1, local k 0,1
            for li in range(4):
                i, lk = divmod(li, 2)
                gk = lk # local k 0,1 maps to global k 0,1
                W[gk * n + i][col] += w[li]
            col += 1
        
        # 7 Strassen terms for B sub-block 2 (local k 2,3)
        for u, v, w in strassen_terms:
            # Map u to U (same as above)
            for li in range(4):
                i, lj = divmod(li, 2)
                gj = lj + a_offset
                U[i * m + gj][col] += u[li]
            
            # Map v to V
            # v is for B (2x2), local j 0,1, local k 0,1
            # But this corresponds to global k 2,3
            for lj in range(2):
                for lk in range(2):
                    li = lj * 2 + lk
                    gj = lj + b_offset
                    gk = 2 + lk # local k 0,1 maps to global k 2,3
                    V[gj * p + gk][col] += v[li]
            
            # Map w to W
            # w is for C (2x2), local i 0,1, local k 0,1
            # But this corresponds to global k 2,3
            for li in range(4):
                i, lk = divmod(li, 2)
                gk = 2 + lk # local k 0,1 maps to global k 2,3
                W[gk * n + i][col] += w[li]
            col += 1
        
        # 4 Naive terms for B sub-block 3 (local k 4)
        for u, v, w in naive_terms:
            # Map u to U
            for li in range(4):
                i, lj = divmod(li, 2)
                gj = lj + a_offset
                U[i * m + gj][col] += u[li]
            
            # Map v to V
            # v is for B (2x1), local j 0,1
            for lj in range(2):
                li = lj
                gj = lj + b_offset
                gk = 4 # local k 4 maps to global k 4
                V[gj * p + gk][col] += v[li]
            
            # Map w to W
            # w is for C (2x1), local i 0,1
            for li in range(2):
                i = li
                gk = 4 # local k 4 maps to global k 4
                W[gk * n + i][col] += w[li]
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
