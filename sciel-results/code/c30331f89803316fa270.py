"""Matrix multiplication 3,3,5 — rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 9, 15, 15
    
    # Build the target tensor T (flattened)
    # T[a, b, c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for consistent i,j,k
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition (rank 45)
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Strategy: Replace the 8 schoolbook terms for i,j,k in {0,1} with 7 Strassen terms
    # This reduces rank from 45 to 44.
    
    # Indices for the 2x2x2 sub-tensor (i,j,k in {0,1})
    # a in {0, 1, 3, 4}
    # b in {0, 1, 5, 6}
    # c in {0, 1, 3, 4}
    
    # Identify the 8 columns in U_school, V_school, W_school corresponding to these terms
    # Term (i,j,k) is at index i*m*p + j*p + k in the loop? No, col increments.
    # We can just rebuild the factors.
    
    U_new = []
    V_new = []
    W_new = []
    
    # Add the 37 non-overlapping schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i in [0, 1] and j in [0, 1] and k in [0, 1]:
                    continue
                a = i * m + j
                b = j * p + k
                c = k * n + i
                u = np.zeros(nm)
                v = np.zeros(mp)
                w = np.zeros(pn)
                u[a] = 1
                v[b] = 1
                w[c] = 1
                U_new.append(u)
                V_new.append(v)
                W_new.append(w)
    
    # Add the 7 Strassen terms for the 2x2x2 sub-tensor
    # Corrected Strassen terms for C_ki = sum_j A_ij B_jk
    # Basis order for A: (A00, A01, A10, A11)
    # Basis order for B: (B00, B01, B10, B11)
    # Basis order for C: (C00, C01, C10, C11)
    
    strassen_terms = [
        # M1: (A00+A11)(B00+B11) -> C00+C11
        (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),
        # M2: (A10+A11)B00 -> C10
        (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 0, 1, 0])),
        # M3: A00(B10-B11) -> C01
        (np.array([1, 0, 0, 0]), np.array([0, 0, 1, -1]), np.array([0, 1, 0, 0])),
        # M4: A11(B01-B00) -> C10
        (np.array([0, 0, 0, 1]), np.array([-1, 1, 0, 0]), np.array([0, 0, 1, 0])),
        # M5: (A00+A01)B11 -> C00
        (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([1, 0, 0, 0])),
        # M6: (A10-A00)(B00+B01) -> C11
        (np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1])),
        # M7: (A01-A11)(B10+B11) -> C00+C10
        (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 0, 1, 0]))
    ]
    
    # Mappings
    map_a = {0: 0, 1: 1, 2: 3, 3: 4}
    map_b = {0: 0, 1: 1, 2: 5, 3: 6}
    # c = 5*k + i
    # (k=0, i=0)->0, (k=0, i=1)->1, (k=1, i=0)->5, (k=1, i=1)->6
    map_c = {0: 0, 1: 1, 2: 5, 3: 6}
    
    for u_sub, v_sub, w_sub in strassen_terms:
        u_global = np.zeros(nm)
        v_global = np.zeros(mp)
        w_global = np.zeros(pn)
        
        for s, g in map_a.items():
            u_global[g] = u_sub[s]
        for s, g in map_b.items():
            v_global[g] = v_sub[s]
        for s, g in map_c.items():
            w_global[g] = w_sub[s]
            
        U_new.append(u_global)
        V_new.append(v_global)
        W_new.append(w_global)
        
    U_final = np.column_stack(U_new)
    V_final = np.column_stack(V_new)
    W_final = np.column_stack(W_new)
    
    # Verify
    T_recon = np.einsum('ar,br,cr->abc', U_final, V_final, W_final)
    if not np.allclose(T_recon, T, atol=1e-10):
        # Fallback to schoolbook if verification fails
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
# EVOLVE-BLOCK-END
