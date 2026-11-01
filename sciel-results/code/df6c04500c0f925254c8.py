import numpy as np
import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=16
    # U shape: (n*m, rank) = (4, rank)
    # V shape: (m*p, rank) = (32, rank)
    # W shape: (p*n, rank) = (32, rank)
    
    # We use 8 blocks of 2x2 matrix multiplication.
    # Rank of 2x2 matmul is 7.
    # Total rank = 8 * 7 = 56.
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen vectors for 2x2 multiplication
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # C = A * B
    #
    # M1 = (a + d)(e + h)
    # M2 = (c + d)e
    # M3 = a(f - h)
    # M4 = d(g - e)
    # M5 = (a + b)h
    # M6 = (c - a)(e + f)
    # M7 = (b - d)(g + h)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Indices in U (flattened A):
    # 0: A[0,0]=a, 1: A[0,1]=b, 2: A[1,0]=c, 3: A[1,1]=d
    
    # Indices in V (flattened B block):
    # For block r, columns 2r, 2r+1.
    # V index for B[j, k] is j*p + k.
    # Block r has k in {2r, 2r+1}.
    # j=0: indices 2r, 2r+1 (e, f)
    # j=1: indices 16+2r, 16+2r+1 (g, h)
    
    # Indices in W (flattened C block):
    # W index for C[i, k] is k*n + i.
    # Block r has k in {2r, 2r+1}.
    # C[0, 2r] -> index 4r + 0
    # C[0, 2r+1] -> index 4r + 2
    # C[1, 2r] -> index 4r + 1
    # C[1, 2r+1] -> index 4r + 3
    
    # Strassen coefficients
    # u_vecs[s][li] where s is 0..6 (term index), li is 0..3 (A index)
    u_vecs = [
        [1, 0, 0, 1],  # M1: a+d
        [0, 0, 1, 1],  # M2: c+d
        [1, 0, 0, 0],  # M3: a
        [0, 0, 0, 1],  # M4: d
        [1, 1, 0, 0],  # M5: a+b
        [-1, 0, 1, 0], # M6: c-a
        [0, 1, 0, -1]  # M7: b-d
    ]
    
    # v_vecs[s][li] where li corresponds to B indices in block:
    # li=0: e, li=1: f, li=2: g, li=3: h
    v_vecs = [
        [1, 0, 0, 1],  # M1: e+h
        [1, 0, 0, 0],  # M2: e
        [0, 1, 0, -1], # M3: f-h
        [-1, 0, 1, 0], # M4: g-e
        [0, 0, 0, 1],  # M5: h
        [1, 1, 0, 0],  # M6: e+f
        [0, 0, 1, 1]   # M7: g+h
    ]
    
    # w_vecs[s][li] where li corresponds to C indices in block:
    # li=0: C11, li=1: C12, li=2: C21, li=3: C22
    w_vecs = [
        [1, 0, 0, 1],  # M1: C11, C22
        [0, 0, 1, -1], # M2: C21, -C22
        [0, 1, 0, 1],  # M3: C12, C22
        [1, 0, 1, 0],  # M4: C11, C21
        [-1, 1, 0, 0], # M5: -C11, C12
        [0, 0, 0, 1],  # M6: C22
        [1, 0, 0, 0]   # M7: C11
    ]
    
    for r in range(8):
        base_col = 7 * r
        # V indices for block r
        # e: 2r, f: 2r+1, g: 16+2r, h: 16+2r+1
        v_idx = [2*r, 2*r+1, 16+2*r, 16+2*r+1]
        
        # W indices for block r
        # C11: 4r, C12: 4r+2, C21: 4r+1, C22: 4r+3
        w_idx = [4*r, 4*r+2, 4*r+1, 4*r+3]
        
        for s in range(7):
            col = base_col + s
            
            # Fill U
            for li in range(4):
                val = u_vecs[s][li]
                if val != 0:
                    U[li, col] = val
                    
            # Fill V
            for li in range(4):
                val = v_vecs[s][li]
                if val != 0:
                    V[v_idx[li], col] = val
                    
            # Fill W
            for li in range(4):
                val = w_vecs[s][li]
                if val != 0:
                    W[w_idx[li], col] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
