"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill

    # Build the tensor T as a 3D array
    # T[a, b, c] where:
    #   a = i*m + j (U index), i in [0,n), j in [0,m)
    #   b = j*p + l (V index), j in [0,m), l in [0,p)
    #   c = l*n + i (W index), l in [0,p), i in [0,n)
    # T[i*m+j, j*p+l, l*n+i] = 1 for all valid i,j,l

    # Schoolbook decomposition (rank = n*m*p = 128)
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for l in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + l][col] = 1
                    W[l * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Strassen-based decomposition for 4x4x4
    # This gives rank 49 for 4x4x4, so rank 98 for 4x4x8
    
    # Strassen rank-7 terms for 2x2x2 multiplication
    # Each term is (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict { (i, j): coeff } for A
    # V_coeffs: dict { (j, k): coeff } for B
    # W_coeffs: dict { (i, k): coeff } for C
    strassen_terms_2x2 = [
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]

    # Top-level Strassen terms for 4x4x4
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    top_level_terms = [
        # M1
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        # M2
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)], [((1,0), 1), ((1,1), -1)]),
        # M3
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)], [((0,1), 1), ((1,1), 1)]),
        # M4
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)], [((0,0), 1), ((1,0), 1)]),
        # M5
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)], [((0,0), -1), ((0,1), 1)]),
        # M6
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)], [((1,1), 1)]),
        # M7
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)], [((0,0), 1)])
    ]

    # Generate rank-49 terms for 4x4x4 multiplication
    all_terms = []
    
    for t_idx in range(7):
        a_blocks, b_blocks, c_blocks = top_level_terms[t_idx]
        
        # For each of the 7 Strassen terms of the 2x2x2 multiplication:
        for s_idx in range(7):
            s_u, s_v, s_w = strassen_terms_2x2[s_idx]
            
            # Map A_virtual entries to global A entries
            u_global = {}
            for (r, c), coeff_s in s_u.items():
                for (u, v), coeff_a in a_blocks:
                    i = 2*u + r
                    j = 2*v + c
                    u_global[(i, j)] = u_global.get((i, j), 0) + coeff_s * coeff_a
            
            # Map B_virtual entries to global B entries
            v_global = {}
            for (r, c), coeff_s in s_v.items():
                for (v, w), coeff_b in b_blocks:
                    j = 2*v + r
                    k = 2*w + c
                    v_global[(j, k)] = v_global.get((j, k), 0) + coeff_s * coeff_b
            
            # Map C_virtual entries to global C entries
            w_global = {}
            for (r, c), coeff_s in s_w.items():
                for (u, w), coeff_c in c_blocks:
                    i = 2*u + r
                    k = 2*w + c
                    w_global[(i, k)] = w_global.get((i, k), 0) + coeff_s * coeff_c
            
            # Clean up zeros
            u_global = {k: v for k, v in u_global.items() if v != 0}
            v_global = {k: v for k, v in v_global.items() if v != 0}
            w_global = {k: v for k, v in w_global.items() if v != 0}
            
            if u_global and v_global and w_global:
                all_terms.append((u_global, v_global, w_global))

    # We have 49 terms for 4x4x4 (left part).
    # Now handle the right part (4x4x4 for columns 4-7).
    terms_right = []
    for u_dict, v_dict, w_dict in all_terms:
        u_new = u_dict  # A is the same
        v_new = {}
        for (j, k), coeff in v_dict.items():
            v_new[(j, k + 4)] = coeff
        w_new = {}
        for (i, k), coeff in w_dict.items():
            w_new[(i, k + 4)] = coeff
        terms_right.append((u_new, v_new, w_new))

    # Total terms: 49 + 49 = 98
    total_terms = all_terms + terms_right
    rank = len(total_terms)

    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for col_idx, (u_dict, v_dict, w_dict) in enumerate(total_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][col_idx] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][col_idx] = coeff
        for (k, i), coeff in w_dict.items():
            W[k * n + i][col_idx] = coeff

    # Verify the decomposition
    def verify(U, V, W):
        rank = len(U[0])
        # Build the tensor T
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for l in range(p):
                    a = i * m + j
                    b = j * p + l
                    c = l * n + i
                    T[a, b, c] = 1
        
        # Compute the decomposition
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        # sum_s U[a,s] * V[b,s] * W[c,s]
        # This is a tensor contraction
        result = np.zeros((n*m, m*p, p*n))
        for s in range(rank):
            u_col = U_arr[:, s]
            v_col = V_arr[:, s]
            w_col = W_arr[:, s]
            # Outer product of three vectors
            result += np.einsum('a,b,c->abc', u_col, v_col, w_col)
        
        # Check if result matches T
        max_diff = np.max(np.abs(result - T))
        return max_diff < 1e-10

    # Verify our Strassen decomposition
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Fall back to schoolbook if verification fails
        return schoolbook()
# EVOLVE-BLOCK-END
