"""Optimized construction for 2x3x8 matrix multiplication tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 8
    
    # Total rank will be 1 (for 2x1x8 part) + 28 (for 2x2x8 part) = 29
    rank = 29
    U = np.zeros((n * m, rank), dtype=int)  # 6 x 29
    V = np.zeros((m * p, rank), dtype=int)  # 24 x 29
    W = np.zeros((p * n, rank), dtype=int)  # 16 x 29
    
    # Helper to index U, V, W
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    r = 0
    
    # --- Part 1: m1=1 (j=0) ---
    # A is 2x1 (indices i=0,1; j=0)
    # B is 1x8 (indices j=0; k=0..7)
    # C is 2x8 (indices i=0,1; k=0..7)
    # C[i,k] = A[i,0] * B[0,k]
    # This can be done with rank 1.
    # U[i, r] = 1 for i=0,1
    # V[k, r] = 1 for k=0..7
    # W[2k+i, r] = 1
    
    for i in range(n):
        U[u_idx(i, 0), r] = 1
    for k in range(p):
        V[v_idx(0, k), r] = 1
    for k in range(p):
        for i in range(n):
            W[w_idx(k, i), r] = 1
            
    r += 1
    
    # --- Part 2: m2=2 (j=1,2) ---
    # A is 2x2 (indices i=0,1; j=1,2)
    # B is 2x8 (indices j=1,2; k=0..7)
    # We split B into 4 blocks of columns: k in {0,1}, {2,3}, {4,5}, {6,7}
    # For each block b, B_block is 2x2.
    # We use Strassen for each 2x2x2 multiplication.
    
    # Strassen coefficients for 2x2x2
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # Map local indices to global indices for A, B, C
    # A local 0 (a) -> global i=0, j=1
    # A local 1 (b) -> global i=0, j=2
    # A local 2 (c) -> global i=1, j=1
    # A local 3 (d) -> global i=1, j=2
    a_local_to_global = {
        0: (0, 1),
        1: (0, 2),
        2: (1, 1),
        3: (1, 2)
    }
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        b_local_to_global = {
            0: (1, k0),  # e
            1: (1, k1),  # f
            2: (2, k0),  # g
            3: (2, k1)   # h
        }
        
        c_local_to_global = {
            0: (0, k0),  # C00
            1: (0, k1),  # C01
            2: (1, k0),  # C10
            3: (1, k1)   # C11
        }
        
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            for loc_a, coeff in u_coeffs.items():
                gi, gj = a_local_to_global[loc_a]
                U[u_idx(gi, gj), r] += coeff
            
            for loc_b, coeff in v_coeffs.items():
                gj, gk = b_local_to_global[loc_b]
                V[v_idx(gj, gk), r] += coeff
            
            for loc_c, coeff in w_coeffs.items():
                gi, gk = c_local_to_global[loc_c]
                W[w_idx(gk, gi), r] += coeff
            
            r += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
