import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=int)
    V_school = np.zeros((m * p, rank_school), dtype=int)
    W_school = np.zeros((p * n, rank_school), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    deadline = time.time() + 110  # safety margin
    
    def verify(U_mat, V_mat, W_mat):
        """Verify exact decomposition. Returns True if valid."""
        for j in range(m):
            for k in range(p):
                for i in range(n):
                    for jp in range(m):
                        for kp in range(p):
                            for ip in range(n):
                                s = 0
                                for c in range(W_mat.shape[1]):
                                    s += U_mat[i*m+j, c] * V_mat[jp*p+k, c] * W_mat[kp*n+ip, c]
                                expected = 1 if (j == jp and k == kp and i == ip) else 0
                                if s != expected:
                                    return False
        return True
    
    def try_rank(r, max_trials=500):
        nonlocal best_rank, best_U, best_V, best_W
        if r >= best_rank:
            return False
        
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        
        for trial in range(max_trials):
            if time.time() > deadline:
                return False
            
            # Generate random U and V with small integers
            U = np.random.randint(-2, 3, size=(n*m, r))
            V = np.random.randint(-2, 3, size=(m*p, r))
            
            # Build linear system for W
            # Unknowns: W[k'*n + i', c] for k' in 0..p-1, i' in 0..n-1, c in 0..r-1
            num_unknowns = p * n * r
            
            # Build sparse matrix A and vector b
            rows = []
            cols = []
            vals = []
            b_vec = []
            
            row_idx = 0
            for j in range(m):
                for k in range(p):
                    for i in range(n):
                        for jp in range(m):
                            for kp in range(p):
                                for ip in range(n):
                                    rhs = 1.0 if (j == jp and k == kp and i == ip) else 0.0
                                    has_coeff = False
                                    for c in range(r):
                                        coeff = U[i*m + j, c] * V[jp*p + k, c]
                                        if coeff != 0:
                                            idx = (kp * n + ip) * r + c
                                            rows.append(row_idx)
                                            cols.append(idx)
                                            vals.append(coeff)
                                            has_coeff = True
                                    if has_coeff or rhs != 0:
                                        b_vec.append(rhs)
                                        row_idx += 1
            
            if row_idx == 0:
                continue
                
            A_mat = sp.csr_matrix((vals, (rows, cols)), shape=(row_idx, num_unknowns))
            b_arr = np.array(b_vec)
            
            # Solve least squares
            try:
                x = spla.lsqr(A_mat, b_arr, atol=1e-10, btol=1e-10, itermax=10000)[0]
                residual = A_mat @ x - b_arr
                max_res = np.max(np.abs(residual))
                
                if max_res < 1e-6:
                    W_flat = x
                    W_mat = W_flat.reshape(p * n, r)
                    
                    # Check if W is close to integer
                    W_rounded = np.round(W_mat)
                    if np.max(np.abs(W_mat - W_rounded)) < 1e-4:
                        W_int = W_rounded.astype(int)
                        U_int = U.astype(int)
                        V_int = V.astype(int)
                        
                        # Verify exactly
                        if verify(U_int, V_int, W_int):
                            best_rank = r
                            best_U = U_int
                            best_V = V_int
                            best_W = W_int
                            return True
            except Exception:
                pass
        
        return False
    
    # Try ranks from 7 upward
    for r in range(7, best_rank + 1):
        if time.time() > deadline:
            break
        if try_rank(r, max_trials=300):
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
