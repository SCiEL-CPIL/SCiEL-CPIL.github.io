import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    
    # We will construct a decomposition with rank 55.
    # Part 1: j=0 (2x1x10), rank 20
    # Part 2: j=1,2 (2x2x10), decomposed into 5 blocks of 2x2x2 using Strassen, rank 5*7=35
    
    total_rank = 20 + 35
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    # Part 1: j = 0
    # A[i, 0] * B[0, k] -> C[i, k]
    # U[i*3 + 0, r] = 1
    # V[0*10 + k, r] = 1
    # W[k*2 + i, r] = 1
    for k in range(p):
        r = k
        for i in range(n):
            U[i * m + 0][r] = 1.0
            W[k * n + i][r] = 1.0
        V[0 * p + k][r] = 1.0
        
    # Part 2: j = 1, 2
    # We split k into 5 blocks of size 2.
    # For each block, we use Strassen.
    # Local indices for A: i in {0,1}, j_loc in {0,1} corresponding to global j in {1,2}
    # Local indices for B: j_loc in {0,1} corresponding to global j in {1,2}, k_loc in {0,1} corresponding to global k in {2b, 2b+1}
    # Local indices for C: i in {0,1}, k_loc in {0,1} corresponding to global k in {2b, 2b+1}
    
    strassen_blocks = 5
    for b in range(strassen_blocks):
        k_start = 2 * b
        k_end = k_start + 2
        
        # 7 Strassen terms
        for s in range(7):
            r = 20 + 7 * b + s
            
            # Define coefficients for U (A part), V (B part), W (C part)
            # We need to map the Strassen formulas to the global indices.
            # Global j for A is 1 or 2.
            # Global k for B is k_start or k_start+1.
            
            u_coeffs = {}
            v_coeffs = {}
            w_coeffs = {}
            
            # Helper to add to coeffs
            def add_u(i, j_global, val):
                key = (i, j_global)
                u_coeffs[key] = u_coeffs.get(key, 0.0) + val
            
            def add_v(j_global, k_global, val):
                key = (j_global, k_global)
                v_coeffs[key] = v_coeffs.get(key, 0.0) + val
            
            def add_w(k_global, i, val):
                key = (k_global, i)
                w_coeffs[key] = w_coeffs.get(key, 0.0) + val

            # Strassen M1..M7 definitions
            # A_loc[0][0] = A[0][1], A_loc[1][1] = A[1][2]
            # A_loc[1][0] = A[1][1], A_loc[0][1] = A[0][2]
            # B_loc[0][0] = B[1][k0], B_loc[1][1] = B[2][k1]
            # B_loc[0][1] = B[1][k1], B_loc[1][0] = B[2][k0]
            
            k0 = k_start
            k1 = k_start + 1
            
            if s == 0: # M1
                # (A00 + A11)(B00 + B11)
                add_u(0, 1, 1.0); add_u(1, 2, 1.0)
                add_v(1, k0, 1.0); add_v(2, k1, 1.0)
                # C00 = M1 + ...
                add_w(k0, 0, 1.0)
                # C11 = M1 - ...
                add_w(k1, 1, 1.0)
            elif s == 1: # M2
                # (A10 + A11)B00
                add_u(1, 1, 1.0); add_u(1, 2, 1.0)
                add_v(1, k0, 1.0)
                # C10 = M2 + ...
                add_w(k0, 1, 1.0)
                # C11 = M1 - M2 + ...
                add_w(k1, 1, -1.0)
            elif s == 2: # M3
                # A00(B01 - B11)
                add_u(0, 1, 1.0)
                add_v(1, k1, 1.0); add_v(2, k1, -1.0)
                # C01 = M3 + M5
                add_w(k1, 0, 1.0)
                # C11 = M1 - M2 + M3 + M6
                add_w(k1, 1, 1.0)
            elif s == 3: # M4
                # A11(B10 - B00)
                add_u(1, 2, 1.0)
                add_v(2, k0, 1.0); add_v(1, k0, -1.0)
                # C00 = M1 + M4 - M5 + M7
                add_w(k0, 0, 1.0)
                # C10 = M2 + M4
                add_w(k0, 1, 1.0)
            elif s == 4: # M5
                # (A00 + A01)B11
                add_u(0, 1, 1.0); add_u(0, 2, 1.0)
                add_v(2, k1, 1.0)
                # C00 = M1 + M4 - M5 + M7
                add_w(k0, 0, -1.0)
                # C01 = M3 + M5
                add_w(k1, 0, 1.0)
            elif s == 5: # M6
                # (A10 - A00)(B00 + B01)
                add_u(1, 1, 1.0); add_u(0, 1, -1.0)
                add_v(1, k0, 1.0); add_v(1, k1, 1.0)
                # C11 = M1 - M2 + M3 + M6
                add_w(k1, 1, 1.0)
            elif s == 6: # M7
                # (A01 - A11)(B10 + B11)
                add_u(0, 2, 1.0); add_u(1, 2, -1.0)
                add_v(2, k0, 1.0); add_v(2, k1, 1.0)
                # C00 = M1 + M4 - M5 + M7
                add_w(k0, 0, 1.0)

            # Write to U, V, W
            for (i, j_global), val in u_coeffs.items():
                if val != 0:
                    U[i * m + j_global][r] += val
            
            for (j_global, k_global), val in v_coeffs.items():
                if val != 0:
                    V[j_global * p + k_global][r] += val
            
            for (k_global, i), val in w_coeffs.items():
                if val != 0:
                    W[k_global * n + i][r] += val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
