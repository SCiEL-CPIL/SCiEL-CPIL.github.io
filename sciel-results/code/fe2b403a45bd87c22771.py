"""Matrix multiplication 2x3 x 3x8 rank-minimal decomposition."""
import numpy as np
import time
import itertools
import random

DIMENSIONS = [2, 3, 8]
TIME_LIMIT = 150.0
START_TIME = time.time()

def _time_left():
    return TIME_LIMIT - (time.time() - START_TIME) - 5.0  # 5s safety margin

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Dimensions of factor matrices
    dim_U = n * m  # 6
    dim_V = m * p  # 24
    dim_W = p * n  # 16
    
    # We will construct 44 terms.
    # 4 blocks, each with 7 Strassen terms + 4 Trivial terms.
    rank = 44
    
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to create a zero vector and set specific indices
    def make_vec(dim, indices_vals):
        v = [0.0] * dim
        for idx, val in indices_vals:
            if 0 <= idx < dim:
                v[idx] += val
        return v
    
    # Indices for A (U): i*m + j
    # i in 0..1, j in 0..2
    # A_00: 0, A_01: 1, A_02: 2
    # A_10: 3, A_11: 4, A_12: 5
    
    # Indices for B (V): j*p + k
    # j in 0..2, k in 0..7
    # B_0k: k
    # B_1k: 8+k
    # B_2k: 16+k
    
    # Indices for C (W): k*n + i
    # k in 0..7, i in 0..1
    # C_k0: 2k
    # C_k1: 2k+1
    
    for q in range(4):
        # Block q uses B columns 2q, 2q+1 and C columns 2q, 2q+1
        
        # --- Strassen Terms ---
        
        # M1 = (A00 + A11)(B00 + B11)
        # A: A00(0), A11(4)
        # B: B00(2q), B11(8 + 2q + 1)
        # C: C00(4q), C11(4q + 3)
        U_list.append(make_vec(dim_U, [(0, 1.0), (4, 1.0)]))
        V_list.append(make_vec(dim_V, [(2*q, 1.0), (8 + 2*q + 1, 1.0)]))
        W_list.append(make_vec(dim_W, [(4*q, 1.0), (4*q + 3, 1.0)]))
        
        # M2 = (A10 + A11)B00
        # A: A10(3), A11(4)
        # B: B00(2q)
        # C: C10(4q+2) - C11(4q+3)  -> Coeffs in W: 1 at C10, -1 at C11
        U_list.append(make_vec(dim_U, [(3, 1.0), (4, 1.0)]))
        V_list.append(make_vec(dim_V, [(2*q, 1.0)]))
        W_list.append(make_vec(dim_W, [(4*q + 2, 1.0), (4*q + 3, -1.0)]))
        
        # M3 = A00(B01 - B11)
        # A: A00(0)
        # B: B01(2q+1), B11(8 + 2q + 1)
        # C: C01(4q+1) + C11(4q+3)
        U_list.append(make_vec(dim_U, [(0, 1.0)]))
        V_list.append(make_vec(dim_V, [(2*q + 1, 1.0), (8 + 2*q + 1, -1.0)]))
        W_list.append(make_vec(dim_W, [(4*q + 1, 1.0), (4*q + 3, 1.0)]))
        
        # M4 = A11(B10 - B00)
        # A: A11(4)
        # B: B10(8 + 2q), B00(2q)
        # C: C00(4q) + C10(4q+2)
        U_list.append(make_vec(dim_U, [(4, 1.0)]))
        V_list.append(make_vec(dim_V, [(8 + 2*q, 1.0), (2*q, -1.0)]))
        W_list.append(make_vec(dim_W, [(4*q, 1.0), (4*q + 2, 1.0)]))
        
        # M5 = (A00 + A10)B11
        # A: A00(0), A10(3)
        # B: B11(8 + 2q + 1)
        # C: -C00(4q) + C01(4q+1)
        U_list.append(make_vec(dim_U, [(0, 1.0), (3, 1.0)]))
        V_list.append(make_vec(dim_V, [(8 + 2*q + 1, 1.0)]))
        W_list.append(make_vec(dim_W, [(4*q, -1.0), (4*q + 1, 1.0)]))
        
        # M6 = (A11 - A00)(B00 + B01)
        # A: A11(4), A00(0)
        # B: B00(2q), B01(2q+1)
        # C: C11(4q+3)
        U_list.append(make_vec(dim_U, [(4, 1.0), (0, -1.0)]))
        V_list.append(make_vec(dim_V, [(2*q, 1.0), (2*q + 1, 1.0)]))
        W_list.append(make_vec(dim_W, [(4*q + 3, 1.0)]))
        
        # M7 = (A01 - A11)(B10 + B11)
        # A: A01(1), A11(4)
        # B: B10(8 + 2q), B11(8 + 2q + 1)
        # C: C00(4q)
        U_list.append(make_vec(dim_U, [(1, 1.0), (4, -1.0)]))
        V_list.append(make_vec(dim_V, [(8 + 2*q, 1.0), (8 + 2*q + 1, 1.0)]))
        W_list.append(make_vec(dim_W, [(4*q, 1.0)]))
        
        # --- Trivial Terms ---
        # A12 (idx 2), A22 (idx 5)  [Note: A is 2x3, so indices 2 and 5 are col 2]
        # B20 (idx 16+2q), B21 (idx 16+2q+1)
        # C00 (idx 4q), C01 (idx 4q+1), C10 (idx 4q+2), C11 (idx 4q+3)
        
        # T1: A12 * B20 -> C00
        U_list.append(make_vec(dim_U, [(2, 1.0)]))
        V_list.append(make_vec(dim_V, [(16 + 2*q, 1.0)]))
        W_list.append(make_vec(dim_W, [(4*q, 1.0)]))
        
        # T2: A12 * B21 -> C01
        U_list.append(make_vec(dim_U, [(2, 1.0)]))
        V_list.append(make_vec(dim_V, [(16 + 2*q + 1, 1.0)]))
        W_list.append(make_vec(dim_W, [(4*q + 1, 1.0)]))
        
        # T3: A22 * B20 -> C10
        U_list.append(make_vec(dim_U, [(5, 1.0)]))
        V_list.append(make_vec(dim_V, [(16 + 2*q, 1.0)]))
        W_list.append(make_vec(dim_W, [(4*q + 2, 1.0)]))
        
        # T4: A22 * B21 -> C11
        U_list.append(make_vec(dim_U, [(5, 1.0)]))
        V_list.append(make_vec(dim_V, [(16 + 2*q + 1, 1.0)]))
        W_list.append(make_vec(dim_W, [(4*q + 3, 1.0)]))

    U = np.array(U_list, dtype=float).T
    V = np.array(V_list, dtype=float).T
    W = np.array(W_list, dtype=float).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
