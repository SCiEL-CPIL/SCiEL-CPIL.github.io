import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank-56 decomposition: 8 independent 2x2x2 blocks, each using Strassen (rank 7)
    # The 2x2x16 multiplication is a direct sum of 8 copies of 2x2x2 multiplication.
    # Each 2x2x2 block has tensor rank 7 (Strassen), total rank = 8*7 = 56.
    
    rank = 56
    # U shape: (n*m, rank) = (4, 56)
    # V shape: (m*p, rank) = (32, 56)
    # W shape: (p*n, rank) = (32, 56)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen vectors for 2x2 matrix multiplication (rank 7)
    # Local indices for A (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local indices for B (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local indices for C (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # u_vecs[s][li] is the coefficient for A[li] in the s-th product
    u_vecs = [
        [1, 0, 0, 1],   # M1: A00 + A11
        [0, 0, 1, 1],   # M2: A10 + A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00 + A01
        [-1, 0, 1, 0],  # M6: A10 - A00
        [0, 1, 0, -1]   # M7: A01 - A11
    ]
    
    # v_vecs[s][li] is the coefficient for B[li] in the s-th product
    v_vecs = [
        [1, 0, 0, 1],   # M1: B00 + B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01 - B11
        [-1, 0, 1, 0],  # M4: B10 - B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00 + B01
        [0, 0, 1, 1]    # M7: B10 + B11
    ]
    
    # w_vecs[s][li] is the coefficient for C[li] in the s-th product
    w_vecs = [
        [1, 0, 0, 1],   # M1: C00 + C11
        [0, 0, 1, -1],  # M2: C10 - C11
        [0, 1, 0, 1],   # M3: C01 + C11
        [1, 0, 1, 0],   # M4: C00 + C10
        [-1, 1, 0, 0],  # M5: -C00 + C01
        [0, 0, 0, 1],   # M6: C11
        [1, 0, 0, 0]    # M7: C00
    ]
    
    for r in range(8):
        base_col = 7 * r
        for s in range(7):
            col = base_col + s
            
            # U: A indices are global 0..3
            # li=0 -> A[0,0] -> idx 0
            # li=1 -> A[0,1] -> idx 1
            # li=2 -> A[1,0] -> idx 2
            # li=3 -> A[1,1] -> idx 3
            for li in range(4):
                val = u_vecs[s][li]
                if val != 0:
                    U[li][col] = val
                    
            # V: B indices for block r
            # B is 2x16. Flattened row-major: idx = j*16 + k
            # li=0 -> B[0, 2r]   -> idx 2r
            # li=1 -> B[0, 2r+1] -> idx 2r+1
            # li=2 -> B[1, 2r]   -> idx 16+2r
            # li=3 -> B[1, 2r+1] -> idx 16+2r+1
            v_idx = [2*r, 2*r+1, 16+2*r, 16+2*r+1]
            for li in range(4):
                val = v_vecs[s][li]
                if val != 0:
                    V[v_idx[li]][col] = val
                    
            # W: C indices for block r
            # C is 2x16. Flattened per convention W[k*n+i]: idx = k*2 + i
            # li=0 -> C[0, 2r]   -> i=0, k=2r   -> idx 4r
            # li=1 -> C[0, 2r+1] -> i=0, k=2r+1 -> idx 4r+2
            # li=2 -> C[1, 2r]   -> i=1, k=2r   -> idx 4r+1
            # li=3 -> C[1, 2r+1] -> i=1, k=2r+1 -> idx 4r+3
            w_idx = [4*r, 4*r+2, 4*r+1, 4*r+3]
            for li in range(4):
                val = w_vecs[s][li]
                if val != 0:
                    W[w_idx[li]][col] = val
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
