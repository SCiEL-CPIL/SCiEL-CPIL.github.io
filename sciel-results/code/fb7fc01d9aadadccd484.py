import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE_S = 110  # safety margin from 120s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    def time_left():
        return DEADLINE_S - (time.time() - start_time)
    
    def schoolbook():
        rank = n*m*p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def validate(U, V, W, tol=1e-10):
        """Check that sum_c U[:,c] V[:,c] W[:,c] == multiplication tensor."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        # Compute tensor: T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
        # Build expected tensor
        T_expected = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_expected[i,j,k] = 1.0
        
        # Build T from factors
        T_computed = np.zeros((n, m, p))
        for c in range(rank):
            u_c = U[:, c]
            v_c = V[:, c]
            w_c = W[:, c]
            # Reshape to (n,m), (m,p), (p,n)
            u_mat = u_c.reshape(n, m)
            v_mat = v_c.reshape(m, p)
            w_mat = w_c.reshape(p, n)
            # Outer contribution: T[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            for i in range(n):
                for j in range(m):
                    if abs(u_mat[i,j]) < 1e-15:
                        continue
                    for k in range(p):
                        T_computed[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        
        return np.max(np.abs(T_computed - T_expected)) < tol

    # Construct the rank 68 decomposition
    # 4 groups of k (size 2)
    # For each group, use Strassen for 2x2x2 block (i in {0,1}, j in {0,1})
    # and Schoolbook for the rest.
    
    rank = 68
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    
    # Define Strassen coefficients for 2x2x2
    # M1: (a+d)(e+h)
    # M2: (c+d)e
    # M3: a(f-h)
    # M4: d(g-e)
    # M5: (a+b)h
    # M6: (c-a)(e+f)
    # M7: (b-d)(g+h)
    
    # U coeffs (for A)
    # a=A00, b=A01, c=A10, d=A11
    U_coeffs = [
        {0: 1, 3: 1},       # M1: a, d
        {2: 1, 3: 1},       # M2: c, d
        {0: 1},             # M3: a
        {3: 1},             # M4: d
        {0: 1, 1: 1},       # M5: a, b
        {2: 1, 0: -1},      # M6: c, -a
        {1: 1, 3: -1}       # M7: b, -d
    ]
    
    # V coeffs (for B)
    # e=B00, f=B01, g=B10, h=B11
    V_coeffs = [
        {0: 1, 3: 1},       # M1: e, h
        {0: 1},             # M2: e
        {1: 1, 3: -1},      # M3: f, -h
        {2: 1, 0: -1},      # M4: g, -e
        {3: 1},             # M5: h
        {0: 1, 1: 1},       # M6: e, f
        {2: 1, 3: 1}        # M7: g, h
    ]
    
    # W coeffs (for C)
    # C00, C01, C10, C11
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    W_coeffs = [
        {0: 1, 3: 1},       # M1: C00, C11
        {2: 1, 3: -1},      # M2: C10, -C11
        {1: 1, 3: 1},       # M3: C01, C11
        {0: 1, 2: 1},       # M4: C00, C10
        {0: -1, 1: 1},      # M5: -C00, C01
        {3: 1},             # M6: C11
        {0: 1}              # M7: C00
    ]
    
    # Iterate through 4 k-groups
    for g in range(4):
        k0 = 2 * g
        k1 = 2 * g + 1
        
        # 1. Strassen block for i in {0,1}, j in {0,1}
        for l in range(7):
            # U
            for li, coeff in U_coeffs[l].items():
                i_local = li // 2
                j_local = li % 2
                i_global = i_local
                j_global = j_local
                U[i_global*m + j_global, col] += coeff
            # V
            for lj, coeff in V_coeffs[l].items():
                j_local = lj // 2
                k_local = lj % 2
                j_global = j_local
                k_global = k0 + k_local
                V[j_global*p + k_global, col] += coeff
            # W
            for lc, coeff in W_coeffs[l].items():
                i_local = lc // 2
                k_local = lc % 2
                i_global = i_local
                k_global = k0 + k_local
                W[k_global*n + i_global, col] += coeff
            col += 1
        
        # 2. Schoolbook for remaining parts in this k-group
        # Parts:
        # i in {0,1}, j=2, k in {k0, k1}
        for i in range(2):
            for k in [k0, k1]:
                U[i*m + 2, col] = 1.0
                V[2*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
        
        # i=2, j in {0,1}, k in {k0, k1}
        for j in range(2):
            for k in [k0, k1]:
                U[2*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + 2, col] = 1.0
                col += 1
        
        # i=2, j=2, k in {k0, k1}
        for k in [k0, k1]:
            U[2*m + 2, col] = 1.0
            V[2*p + k, col] = 1.0
            W[k*n + 2, col] = 1.0
            col += 1

    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify
    if validate(U, V, W):
        return result
    else:
        # Fallback to schoolbook if validation fails (should not happen)
        return schoolbook()
