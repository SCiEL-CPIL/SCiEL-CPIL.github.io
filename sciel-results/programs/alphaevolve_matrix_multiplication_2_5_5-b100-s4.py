import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # We will build the matrices row by row or column by column.
    # Since we don't know the final rank, we can use lists of lists and append.
    # Or we can pre-allocate with a large enough size and trim.
    # Max rank is 50. Min rank is 46.
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    # Helper to add a rank-1 term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        # u_coeffs: dict mapping (i, j) to coeff
        # v_coeffs: dict mapping (j, k) to coeff
        # w_coeffs: dict mapping (k, i) to coeff
        # We need to append a new column to U, V, W
        # U has shape (n*m, rank). Row index is i*m + j.
        # V has shape (m*p, rank). Row index is j*p + k.
        # W has shape (p*n, rank). Row index is k*n + i.
        
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        
        for (i, j), val in u_coeffs.items():
            u_vec[i * m + j] = val
        for (j, k), val in v_coeffs.items():
            v_vec[j * p + k] = val
        for (k, i), val in w_coeffs.items():
            w_vec[k * n + i] = val
            
        U_rows.append(u_vec)
        V_rows.append(v_vec)
        W_rows.append(w_vec)

    # Helper to add a Strassen block for A rows I, A cols J, B cols K
    # I is always {0, 1} for n=2
    def add_strassen_block(J, K):
        j1, j2 = J
        k1, k2 = K
        i1, i2 = 0, 1
        
        # Local indices:
        # A: (i1, j1), (i1, j2), (i2, j1), (i2, j2)
        # B: (j1, k1), (j1, k2), (j2, k1), (j2, k2)
        # C: (i1, k1), (i1, k2), (i2, k1), (i2, k2)
        
        # M1 = (A00 + A11)(B00 + B11)
        # Note: Local A00 is A[i1, j1], A11 is A[i2, j2]
        # Local B00 is B[j1, k1], B11 is B[j2, k2]
        
        u_m1 = {(i1, j1): 1, (i2, j2): 1}
        v_m1 = {(j1, k1): 1, (j2, k2): 1}
        w_m1 = {(k1, i1): 1, (k2, i2): 1} # C00 gets +M1, C11 gets +M1
        # Wait, C00 is (i1, k1), C11 is (i2, k2).
        # In Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So for M1:
        # w_m1 should have 1 at (k1, i1) and 1 at (k2, i2).
        # My w_m1 above is correct for C00 and C11 contributions.
        
        add_term(u_m1, v_m1, w_m1)
        
        # M2 = (A10 + A11)B00
        # Local A10 is A[i2, j1], A11 is A[i2, j2]
        # Local B00 is B[j1, k1]
        u_m2 = {(i2, j1): 1, (i2, j2): 1}
        v_m2 = {(j1, k1): 1}
        # C10 gets +M2, C11 gets -M2
        w_m2 = {(k1, i2): 1, (k2, i2): -1}
        add_term(u_m2, v_m2, w_m2)
        
        # M3 = A00(B01 - B11)
        # Local A00 is A[i1, j1]
        # Local B01 is B[j1, k2], B11 is B[j2, k2]
        u_m3 = {(i1, j1): 1}
        v_m3 = {(j1, k2): 1, (j2, k2): -1}
        # C01 gets +M3, C11 gets +M3
        w_m3 = {(k2, i1): 1, (k2, i2): 1}
        add_term(u_m3, v_m3, w_m3)
        
        # M4 = A11(B10 - B00)
        # Local A11 is A[i2, j2]
        # Local B10 is B[j2, k1], B00 is B[j1, k1]
        u_m4 = {(i2, j2): 1}
        v_m4 = {(j2, k1): 1, (j1, k1): -1}
        # C00 gets +M4, C10 gets +M4
        w_m4 = {(k1, i1): 1, (k1, i2): 1}
        add_term(u_m4, v_m4, w_m4)
        
        # M5 = (A00 + A01)B11
        # Local A00 is A[i1, j1], A01 is A[i1, j2]
        # Local B11 is B[j2, k2]
        u_m5 = {(i1, j1): 1, (i1, j2): 1}
        v_m5 = {(j2, k2): 1}
        # C00 gets -M5, C01 gets +M5
        w_m5 = {(k1, i1): -1, (k2, i1): 1}
        add_term(u_m5, v_m5, w_m5)
        
        # M6 = (A10 - A00)(B00 + B01)
        # Local A10 is A[i2, j1], A00 is A[i1, j1]
        # Local B00 is B[j1, k1], B01 is B[j1, k2]
        u_m6 = {(i2, j1): 1, (i1, j1): -1}
        v_m6 = {(j1, k1): 1, (j1, k2): 1}
        # C11 gets +M6
        w_m6 = {(k2, i2): 1}
        add_term(u_m6, v_m6, w_m6)
        
        # M7 = (A01 - A11)(B10 + B11)
        # Local A01 is A[i1, j2], A11 is A[i2, j2]
        # Local B10 is B[j2, k1], B11 is B[j2, k2]
        u_m7 = {(i1, j2): 1, (i2, j2): -1}
        v_m7 = {(j2, k1): 1, (j2, k2): 1}
        # C00 gets +M7
        w_m7 = {(k1, i1): 1}
        add_term(u_m7, v_m7, w_m7)

    # Helper to add a schoolbook block for A rows I, A cols J, B cols K
    # This is for the remaining parts that are not 2x2x2
    def add_schoolbook_block(J, K):
        # J is a list of j indices, K is a list of k indices
        # I is {0, 1}
        for j in J:
            for k in K:
                for i in range(n):
                    u = {(i, j): 1}
                    v = {(j, k): 1}
                    w = {(k, i): 1}
                    add_term(u, v, w)

    # Partition m=5 into J sets and p=5 into K sets
    # J sets: {0,1}, {2,3}, {4}
    # K sets: {0,1}, {2,3}, {4}
    
    J_sets = [(0, 1), (2, 3), (4,)]
    K_sets = [(0, 1), (2, 3), (4,)]
    
    for J in J_sets:
        for K in K_sets:
            if len(J) == 2 and len(K) == 2:
                add_strassen_block(J, K)
            else:
                add_schoolbook_block(J, K)
                
    # Convert lists of lists to the required format
    # U is list of lists (n*m) x rank
    # The current U_rows is a list of vectors, each of length n*m.
    # We need to transpose it.
    
    rank = len(U_rows)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for i in range(n * m):
            U[i][r] = U_rows[r][i]
        for i in range(m * p):
            V[i][r] = V_rows[r][i]
        for i in range(p * n):
            W[i][r] = W_rows[r][i]
            
    return {"U": U, "V": V, "W": W}
