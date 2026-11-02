"""Optimized construction for 2x3x8 matrix multiplication tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 8
    
    # Approach: Modular Block Decomposition
    # We decompose M_{2,3,8} into M_{2,1,8} + M_{2,2,8}.
    # M_{2,1,8} has rank 16 (outer products).
    # M_{2,2,8} is decomposed into four M_{2,2,2} blocks, each using Strassen (rank 7).
    # Total rank = 16 + 4 * 7 = 44.
    
    rank = 44
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    r = 0
    
    # Helper to add a term
    def add_term(u_map, v_map, w_map):
        nonlocal r
        for (i, j), coeff in u_map.items():
            U[u_idx(i, j), r] += coeff
        for (j, k), coeff in v_map.items():
            V[v_idx(j, k), r] += coeff
        for (k, i), coeff in w_map.items():
            W[w_idx(k, i), r] += coeff
        r += 1

    # Part 1: M_{2,1,8} (j=0)
    # C[i,k] = A[i,0] * B[0,k]
    for i in range(n):
        for k in range(p):
            add_term(
                {(i, 0): 1},
                {(0, k): 1},
                {(k, i): 1}
            )
            
    # Part 2: M_{2,2,8} (j=1,2)
    # Split into 4 blocks of k (size 2)
    # Strassen for 2x2x2
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # Local A: 0:a(i=0,j=1), 1:b(i=0,j=2), 2:c(i=1,j=1), 3:d(i=1,j=2)
        # Local B: 0:e(j=1,k0), 1:f(j=1,k1), 2:g(j=2,k0), 3:h(j=2,k1)
        # Local C: 0:C00(i=0,k0), 1:C01(i=0,k1), 2:C10(i=1,k0), 3:C11(i=1,k1)
        
        ({(0,1): 1, (1,2): 1}, {(1,0): 1, (2,1): 1}, {(0,0): 1, (1,1): 1}), # M1
        ({(1,1): 1, (1,2): 1}, {(1,0): 1}, {(1,0): 1, (1,1): -1}), # M2 (adjusted indices below)
        # Note: The local indices in the tuple above are local to the 2x2 block.
        # I need to map them to global (i,j) and (j,k) and (k,i).
        
    ]
    
    # Re-implement Strassen with explicit global mapping for clarity
    def strassen_block(k0, k1, start_r):
        # A: (0,1), (0,2), (1,1), (1,2)
        # B: (1,k0), (1,k1), (2,k0), (2,k1)
        # C: (k0,0), (k1,0), (k0,1), (k1,1) -- Note W index is (k,i)
        
        # M1 = (A01+A12)(B1k0+B2k1) -> C00 + C11
        # U: (0,1):1, (1,2):1
        # V: (1,k0):1, (2,k1):1
        # W: (k0,0):1, (k1,1):1
        add_term(
            {(0,1): 1, (1,2): 1},
            {(1,k0): 1, (2,k1): 1},
            {(k0,0): 1, (k1,1): 1}
        )
        
        # M2 = (A11+A12)B1k0 -> C10 - C11
        # U: (1,1):1, (1,2):1
        # V: (1,k0):1
        # W: (k0,1):1, (k1,1):-1
        add_term(
            {(1,1): 1, (1,2): 1},
            {(1,k0): 1},
            {(k0,1): 1, (k1,1): -1}
        )
        
        # M3 = A01(B1k1-B2k1) -> C01 + C11
        # U: (0,1):1
        # V: (1,k1):1, (2,k1):-1
        # W: (k1,0):1, (k1,1):1
        add_term(
            {(0,1): 1},
            {(1,k1): 1, (2,k1): -1},
            {(k1,0): 1, (k1,1): 1}
        )
        
        # M4 = A12(B2k0-B1k0) -> C00 + C10
        # U: (1,2):1
        # V: (2,k0):1, (1,k0):-1
        # W: (k0,0):1, (k0,1):1
        add_term(
            {(1,2): 1},
            {(2,k0): 1, (1,k0): -1},
            {(k0,0): 1, (k0,1): 1}
        )
        
        # M5 = (A01+A02)B2k1 -> -C00 + C01
        # U: (0,1):1, (0,2):1
        # V: (2,k1):1
        # W: (k0,0):-1, (k1,0):1
        add_term(
            {(0,1): 1, (0,2): 1},
            {(2,k1): 1},
            {(k0,0): -1, (k1,0): 1}
        )
        
        # M6 = (A11-A01)(B1k0+B1k1) -> C11
        # U: (1,1):1, (0,1):-1
        # V: (1,k0):1, (1,k1):1
        # W: (k1,1):1
        add_term(
            {(1,1): 1, (0,1): -1},
            {(1,k0): 1, (1,k1): 1},
            {(k1,1): 1}
        )
        
        # M7 = (A02-A12)(B2k0+B2k1) -> C00
        # U: (0,2):1, (1,2):-1
        # V: (2,k0):1, (2,k1):1
        # W: (k0,0):1
        add_term(
            {(0,2): 1, (1,2): -1},
            {(2,k0): 1, (2,k1): 1},
            {(k0,0): 1}
        )
        
        return start_r + 7

    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        r = strassen_block(k0, k1, r)
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
