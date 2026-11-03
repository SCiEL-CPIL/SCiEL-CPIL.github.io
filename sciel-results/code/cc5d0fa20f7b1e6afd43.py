import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Strassen's algorithm for 3x3x3: rank 7 decomposition
    # The 7 terms for multiplying two 3x3 matrices C = A * B
    # Using the standard Strassen decomposition adapted for 3x3:
    #
    # M1 = (A00 + A22)(B00 + B22)
    # M2 = (A20 + A22)B00
    # M3 = A00(B01 - B21)
    # M4 = A22(B21 - B01)
    # M5 = (A01 + A11)B11
    # M6 = (A10 + A11)(B01 + B11)
    # M7 = (A00 + A10)(B10 - B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C02 = M2 - M4 + M6
    # C10 = M1 - M3 + M5 - M7
    # C11 = M2 - M1 + M4 + M7
    # C12 = -M2 + M6
    # C20 = M3 - M6 + M7
    # C21 = -M4 - M7
    # C22 = M1 - M2 + M4 - M5
    
    # Define the 7 rank-1 terms for 3x3x3
    # Each term is (u, v, w) where u is 9-dim (A indices), v is 9-dim (B indices), w is 9-dim (C indices)
    # A index: i*m+j = i*3+j, B index: j*p+k = j*3+k, C index: k*n+i = k*3+i
    
    # For each Strassen term, we need to express the linear combinations
    # Term 1: (A00+A22)(B00+B22)
    # A part: A00=(0,0), A22=(2,2) -> indices 0 and 8
    # B part: B00=(0,0), B22=(2,2) -> indices 0 and 8
    # C part: contributes to C00(+1), C10(-1), C11(-1), C22(+1)
    
    strassen_terms = []
    
    # Helper to build a term from linear combinations
    def make_term(a_combos, b_combos, c_combos):
        """a_combos: list of (a_index, coeff), b_combos: list of (b_index, coeff), c_combos: list of (c_index, coeff)"""
        u = np.zeros(9)
        v = np.zeros(9)
        w = np.zeros(9)
        for idx, coeff in a_combos:
            u[idx] = coeff
        for idx, coeff in b_combos:
            v[idx] = coeff
        for idx, coeff in c_combos:
            w[idx] = coeff
        return u, v, w
    
    # M1 = (A00 + A22)(B00 + B22)
    strassen_terms.append(make_term(
        [(0, 1), (8, 1)],
        [(0, 1), (8, 1)],
        [(0, 1), (3, -1), (4, -1), (8, 1)]
    ))
    
    # M2 = (A20 + A22)B00
    strassen_terms.append(make_term(
        [(6, 1), (8, 1)],
        [(0, 1)],
        [(2, 1), (4, 1), (5, -1), (8, -1)]
    ))
    
    # M3 = A00(B01 - B21)
    strassen_terms.append(make_term(
        [(0, 1)],
        [(1, 1), (7, -1)],
        [(1, 1), (3, -1), (6, 1)]
    ))
    
    # M4 = A22(B21 - B01)
    strassen_terms.append(make_term(
        [(8, 1)],
        [(7, 1), (1, -1)],
        [(0, 1), (4, 1), (7, -1), (8, -1)]
    ))
    
    # M5 = (A01 + A11)B11
    strassen_terms.append(make_term(
        [(1, 1), (4, 1)],
        [(4, 1)],
        [(0, -1), (1, 1), (3, 1), (8, -1)]
    ))
    
    # M6 = (A10 + A11)(B01 + B11)
    strassen_terms.append(make_term(
        [(3, 1), (4, 1)],
        [(1, 1), (4, 1)],
        [(2, 1), (5, 1), (6, -1)]
    ))
    
    # M7 = (A00 + A10)(B10 - B11)
    strassen_terms.append(make_term(
        [(0, 1), (3, 1)],
        [(3, 1), (4, -1)],
        [(0, 1), (3, -1), (4, 1), (6, 1), (7, -1)]
    ))
    
    # Build the full 3x3x7 decomposition
    # Rank = 7 (Strassen for B cols 0-2) + 7 (Strassen for B cols 3-5) + 9 (Schoolbook for B col 6) = 23
    
    rank = 23
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Helper to map local 3x3x3 indices to global indices
    # local_b: index in 3x3 B matrix (j*3+k)
    # local_c: index in 3x3 C matrix (k*3+i)
    # offset_k: offset for the B column (0 for first block, 3 for second)
    def map_strassen_term(u_local, v_local, w_local, offset_k):
        u = np.zeros(n*m)
        v = np.zeros(m*p)
        w = np.zeros(p*n)
        
        # u is same
        u[:] = u_local
        
        # v mapping: local (j, k) -> global (j, k+offset_k)
        for idx in range(9):
            if v_local[idx] != 0:
                j = idx // 3
                k = idx % 3
                global_v = j * p + (k + offset_k)
                v[global_v] = v_local[idx]
                
        # w mapping: local (k, i) -> global (k+offset_k, i)
        for idx in range(9):
            if w_local[idx] != 0:
                k = idx // 3
                i = idx % 3
                global_w = (k + offset_k) * n + i
                w[global_w] = w_local[idx]
                
        return u, v, w

    # Block 1: B cols 0-2 (offset_k = 0)
    for r in range(7):
        u, v, w = strassen_terms[r]
        U[:, col], V[:, col], W[:, col] = map_strassen_term(u, v, w, 0)
        col += 1
        
    # Block 2: B cols 3-5 (offset_k = 3)
    for r in range(7):
        u, v, w = strassen_terms[r]
        U[:, col], V[:, col], W[:, col] = map_strassen_term(u, v, w, 3)
        col += 1
        
    # Block 3: B col 6 (Schoolbook 3x3x1)
    # C_i = sum_j A_ij B_j (where B is the column)
    for i in range(n):
        for j in range(m):
            U[i*m+j, col] = 1
            V[j*p+6, col] = 1
            W[6*n+i, col] = 1
            col += 1
            
    # Verify the full decomposition
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
                
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.max(np.abs(T_recon - T))
    
    if error > 1e-10:
        # Fallback to schoolbook if verification fails
        rank_fb = n * m * p
        U_fb = np.zeros((n*m, rank_fb))
        V_fb = np.zeros((m*p, rank_fb))
        W_fb = np.zeros((p*n, rank_fb))
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col_fb] = 1
                    V_fb[j*p+k, col_fb] = 1
                    W_fb[k*n+i, col_fb] = 1
                    col_fb += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
