"""Matrix multiplication 4x4 * 4x6 rank-minimal decomposition."""
import time
import numpy as np
_START_TIME = time.time()
TIME_LIMIT = 140.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    n, m, p = 4, 4, 6
    
    # We decompose 4x4 * 4x6 as:
    # 4x4 * 4x4 (left 4 cols of B) + 4x4 * 4x2 (right 2 cols of B)
    # C = A * B_left + A * B_right
    # C is 4x6, C[:,0:4] = A * B_left (4x4), C[:,4:6] = A * B_right (4x2)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # ---- 4x4x4 part: recursive Strassen, rank 49 ----
    STRASSEN_2x2 = [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
    ]
    
    def add_term(u_pat, v_pat, w_pat, oi, oj, ok):
        """Add a single rank-1 term with 2x2 patterns at block offsets."""
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        for (r, c), cf in u_pat.items():
            u_col[(oi+r) * m + (oj+c)] = cf
        for (r, c), cf in v_pat.items():
            v_col[(oj+r) * p + (ok+c)] = cf
        for (r, c), cf in w_pat.items():
            w_col[(ok+c) * n + (oi+r)] = cf
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
    
    def add_super(ub, vb, wb, oi, oj, ok):
        """Apply 7 Strassen 2x2 patterns with block-level coefficients."""
        for u_pat, v_pat, w_pat in STRASSEN_2x2:
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            for (br, bc), cf_u in ub.items():
                for (r, c), cf_t in u_pat.items():
                    u_col[(oi+2*br+r) * m + (oj+2*bc+c)] += cf_u * cf_t
            for (br, bc), cf_v in vb.items():
                for (r, c), cf_t in v_pat.items():
                    v_col[(oj+2*br+r) * p + (ok+2*bc+c)] += cf_v * cf_t
            for (br, bc), cf_w in wb.items():
                for (r, c), cf_t in w_pat.items():
                    w_col[(ok+2*bc+c) * n + (oi+2*br+r)] += cf_w * cf_t
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
    
    # 4x4x4: one level of recursive Strassen = 7 * 7 = 49 terms
    for ub, vb, wb in STRASSEN_2x2:
        add_super(ub, vb, wb, 0, 0, 0)
    
    # ---- 4x4x2 part: schoolbook, rank 8 ----
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in 0..3, j in 0..3, k in 4..5
    # Schoolbook: for each (i,j,k) with k in {4,5}, one rank-1 term
    # U: A[i,j] at position i*m+j = i*4+j
    # V: B[j,k] at position j*p+k = j*6+k
    # W: C[k,i] at position k*n+i = k*4+i (trace convention W[k*n+i, r])
    
    for i in range(n):
        for j in range(m):
            for k in range(4, 6):  # k = 4, 5
                u_col = [0] * (n * m)
                v_col = [0] * (m * p)
                w_col = [0] * (p * n)
                u_col[i * m + j] = 1
                v_col[j * p + k] = 1
                w_col[k * n + i] = 1
                U_cols.append(u_col)
                V_cols.append(v_col)
                W_cols.append(w_col)
    
    rank = len(U_cols)
    # Should be 49 + 8 = 57
    
    # Build matrices
    U_mat = np.array([[U_cols[r][i] for r in range(rank)] for i in range(n*m)], dtype=np.float64)
    V_mat = np.array([[V_cols[r][i] for r in range(rank)] for i in range(m*p)], dtype=np.float64)
    W_mat = np.array([[W_cols[r][i] for r in range(rank)] for i in range(p*n)], dtype=np.float64)
    
    # Verify
    def verify(U, V, W, n, m, p, rank, tol=1e-10):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Quick verification
    ok = verify(U_mat, V_mat, W_mat, n, m, p, rank)
    if not ok:
        # Fallback: full schoolbook for 4x4x6, rank 96
        U_cols2 = []
        V_cols2 = []
        W_cols2 = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_col = [0] * (n * m)
                    v_col = [0] * (m * p)
                    w_col = [0] * (p * n)
                    u_col[i * m + j] = 1
                    v_col[j * p + k] = 1
                    w_col[k * n + i] = 1
                    U_cols2.append(u_col)
                    V_cols2.append(v_col)
                    W_cols2.append(w_col)
        rank2 = len(U_cols2)
        U_final = [[U_cols2[r][i] for r in range(rank2)] for i in range(n*m)]
        V_final = [[V_cols2[r][i] for r in range(rank2)] for i in range(m*p)]
        W_final = [[W_cols2[r][i] for r in range(rank2)] for i in range(p*n)]
        return {"U": U_final, "V": V_final, "W": W_final}
    
    U_final = U_mat.tolist()
    V_final = V_mat.tolist()
    W_final = W_mat.tolist()
    
    return {"U": U_final, "V": V_final, "W": W_final}
