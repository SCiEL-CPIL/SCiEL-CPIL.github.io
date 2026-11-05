import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
                
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_school

    # Partitions for Block Strassen
    # I: 3 -> [2, 1]
    I_groups = [
        [0, 1], # I_2
        [2]     # I_1
    ]
    
    # J: 5 -> [2, 2, 1]
    J_groups = [
        [0, 1], # J_2a
        [2, 3], # J_2b
        [4]     # J_1
    ]
    
    # K: 6 -> [2, 2, 2]
    K_groups = [
        [0, 1], # K_2a
        [2, 3], # K_2b
        [4, 5]  # K_2c
    ]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a schoolbook term
    def add_schoolbook_term(i, j, k):
        u = np.zeros(nm)
        v = np.zeros(mp)
        w = np.zeros(pn)
        u[i*m + j] = 1.0
        v[j*p + k] = 1.0
        w[k*n + i] = 1.0
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
        
    # Helper to add a Strassen term
    # We need to map the local 2x2 indices to global indices
    def add_strassen_term(I_set, J_set, K_set):
        # I_set, J_set, K_set are lists of global indices of length 2
        i0, i1 = I_set
        j0, j1 = J_set
        k0, k1 = K_set
        
        # Local indices:
        # A: (i0, j0)=00, (i0, j1)=01, (i1, j0)=10, (i1, j1)=11
        # B: (j0, k0)=00, (j0, k1)=01, (j1, k0)=10, (j1, k1)=11
        # C: (i0, k0)=00, (i0, k1)=01, (i1, k0)=10, (i1, k1)=11
        
        # Define the 7 Strassen terms
        # Each term is a linear combination of A entries, B entries, and contributes to C entries.
        # We represent each rank-1 term by (u_vec, v_vec, w_vec)
        
        # M1 = (A00 + A11) * (B00 + B11)
        # u: A00=1, A11=1
        # v: B00=1, B11=1
        # w: C00 += 1, C11 += 1
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i0*m + j0] = 1.0; u[i1*m + j1] = 1.0
        v[j0*p + k0] = 1.0; v[j1*p + k1] = 1.0
        w[k0*n + i0] = 1.0; w[k1*n + i1] = 1.0
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # M2 = (A10 + A11) * B00
        # u: A10=1, A11=1
        # v: B00=1
        # w: C10 += 1
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i1*m + j0] = 1.0; u[i1*m + j1] = 1.0
        v[j0*p + k0] = 1.0
        w[k0*n + i1] = 1.0
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # M3 = A00 * (B01 - B11)
        # u: A00=1
        # v: B01=1, B11=-1
        # w: C01 += 1
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i0*m + j0] = 1.0
        v[j0*p + k1] = 1.0; v[j1*p + k1] = -1.0
        w[k1*n + i0] = 1.0
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # M4 = A11 * (B10 - B00)
        # u: A11=1
        # v: B10=1, B00=-1
        # w: C00 += 1, C10 += 1
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i1*m + j1] = 1.0
        v[j1*p + k0] = 1.0; v[j0*p + k0] = -1.0
        w[k0*n + i0] = 1.0; w[k0*n + i1] = 1.0
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # M5 = (A00 + A01) * B11
        # u: A00=1, A01=1
        # v: B11=1
        # w: C01 += 1, C00 -= 1
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i0*m + j0] = 1.0; u[i0*m + j1] = 1.0
        v[j1*p + k1] = 1.0
        w[k1*n + i0] = 1.0; w[k0*n + i0] = -1.0
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # M6 = (A10 - A00) * (B00 + B01)
        # u: A10=1, A00=-1
        # v: B00=1, B01=1
        # w: C11 += 1
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i1*m + j0] = 1.0; u[i0*m + j0] = -1.0
        v[j0*p + k0] = 1.0; v[j0*p + k1] = 1.0
        w[k1*n + i1] = 1.0
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # M7 = (A01 - A11) * (B10 + B11)
        # u: A01=1, A11=-1
        # v: B10=1, B11=1
        # w: C00 += 1, C11 -= 1
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i0*m + j1] = 1.0; u[i1*m + j1] = -1.0
        v[j1*p + k0] = 1.0; v[j1*p + k1] = 1.0
        w[k0*n + i0] = 1.0; w[k1*n + i1] = -1.0
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
    # Iterate over all block combinations
    for I_set in I_groups:
        for J_set in J_groups:
            for K_set in K_groups:
                if len(I_set) == 2 and len(J_set) == 2 and len(K_set) == 2:
                    add_strassen_term(I_set, J_set, K_set)
                else:
                    # Schoolbook for this block
                    for i in I_set:
                        for j in J_set:
                            for k in K_set:
                                add_schoolbook_term(i, j, k)
                                
    # Construct final matrices
    rank_found = len(U_cols)
    U_final = np.column_stack(U_cols) if U_cols else np.zeros((nm, 0))
    V_final = np.column_stack(V_cols) if V_cols else np.zeros((mp, 0))
    W_final = np.column_stack(W_cols) if W_cols else np.zeros((pn, 0))
    
    # Verify correctness
    # T = U @ V.T @ W.T ? No, T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # Let's check a few random entries or all entries if small enough
    # nm=15, mp=30, pn=18. 15*30*18 = 8100 entries.
    # rank=84.
    # Sum of 84 products of 15x30x18 tensors.
    # Can verify by reconstructing the tensor and comparing to schoolbook.
    
    T_found = np.einsum('ar,br,cr->abc', U_final, V_final, W_final)
    
    # Construct Schoolbook Tensor for comparison
    T_school = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_school[i*m+j, j*p+k, k*n+i] = 1.0
                
    if not np.allclose(T_found, T_school, atol=1e-9):
        # If verification fails, return schoolbook
        return best_result
    else:
        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
