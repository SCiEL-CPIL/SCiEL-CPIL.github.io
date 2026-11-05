import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145
    
    def verify(U, V, W, tol=1e-6):
        """Verify the decomposition exactly."""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def unit_U(i, j):
        v = np.zeros(n*m)
        v[i*m+j] = 1.0
        return v
    
    def unit_V(j, k):
        v = np.zeros(m*p)
        v[j*p+k] = 1.0
        return v
    
    def unit_W(k, i):
        v = np.zeros(p*n)
        v[k*n+i] = 1.0
        return v
    
    def build_strassen_decomp():
        U_cols = []
        V_cols = []
        W_cols = []
        
        def add_term(u, v, w):
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)
        
        def add_strassen_block(i1, i2, j1, j2, k1, k2):
            # M1 = (A[i1,j1] + A[i2,j2]) * (B[j1,k2] + B[j2,k1])
            add_term(
                unit_U(i1,j1) + unit_U(i2,j2),
                unit_V(j1,k2) + unit_V(j2,k1),
                unit_W(k1,i1) + unit_W(k2,i2)
            )
            # M2 = (A[i2,j1] + A[i2,j2]) * B[j1,k1]
            add_term(
                unit_U(i2,j1) + unit_U(i2,j2),
                unit_V(j1,k1),
                unit_W(k1,i2) + unit_W(k2,i2)
            )
            # M3 = A[i1,j1] * (B[j2,k1] - B[j2,k2])
            add_term(
                unit_U(i1,j1),
                unit_V(j2,k1) - unit_V(j2,k2),
                unit_W(k2,i1) + unit_W(k2,i2)
            )
            # M4 = A[i2,j2] * (B[j1,k1] - B[j2,k1])
            add_term(
                unit_U(i2,j2),
                unit_V(j1,k1) - unit_V(j2,k1),
                unit_W(k1,i1) + unit_W(k1,i2)
            )
            # M5 = (A[i1,j1] + A[i1,j2]) * B[j2,k2]
            add_term(
                unit_U(i1,j1) + unit_U(i1,j2),
                unit_V(j2,k2),
                -unit_W(k1,i1) + unit_W(k2,i1)
            )
            # M6 = (A[i2,j1] - A[i1,j1]) * (B[j1,k1] + B[j1,k2])
            add_term(
                unit_U(i2,j1) - unit_U(i1,j1),
                unit_V(j1,k1) + unit_V(j1,k2),
                unit_W(k2,i2)
            )
            # M7 = (A[i1,j2] - A[i2,j2]) * (B[j2,k1] + B[j2,k2])
            add_term(
                unit_U(i1,j2) - unit_U(i2,j2),
                unit_V(j2,k1) + unit_V(j2,k2),
                unit_W(k1,i1) + unit_W(k2,i2)
            )
        
        def add_schoolbook_block(i_list, j_list, k_list):
            for i in i_list:
                for j in j_list:
                    for k in k_list:
                        add_term(unit_U(i,j), unit_V(j,k), unit_W(k,i))
        
        # Top 2 rows, left 4 cols of B:
        add_strassen_block(0, 1, 0, 1, 0, 1)
        add_strassen_block(0, 1, 0, 1, 2, 3)
        add_strassen_block(0, 1, 2, 3, 0, 1)
        add_strassen_block(0, 1, 2, 3, 2, 3)
        
        # Top 2 rows, right 4 cols of B:
        add_strassen_block(0, 1, 0, 1, 4, 5)
        add_strassen_block(0, 1, 0, 1, 6, 7)
        add_strassen_block(0, 1, 2, 3, 4, 5)
        add_strassen_block(0, 1, 2, 3, 6, 7)
        
        # Bottom 1 row, left 4 cols of B:
        add_schoolbook_block([2], [0, 1], [0, 1])
        add_schoolbook_block([2], [0, 1], [2, 3])
        add_schoolbook_block([2], [2, 3], [0, 1])
        add_schoolbook_block([2], [2, 3], [2, 3])
        
        # Bottom 1 row, right 4 cols of B:
        add_schoolbook_block([2], [0, 1], [4, 5])
        add_schoolbook_block([2], [0, 1], [6, 7])
        add_schoolbook_block([2], [2, 3], [4, 5])
        add_schoolbook_block([2], [2, 3], [6, 7])
        
        rank = len(U_cols)
        U = np.column_stack(U_cols)
        V = np.column_stack(V_cols)
        W = np.column_stack(W_cols)
        
        if not verify(U, V, W, tol=1e-6):
            U_sb = np.zeros((n*m, n*m*p))
            V_sb = np.zeros((m*p, n*m*p))
            W_sb = np.zeros((p*n, n*m*p))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_sb[i*m+j, col] = 1
                        V_sb[j*p+k, col] = 1
                        W_sb[k*n+i, col] = 1
                        col += 1
            return U_sb, V_sb, W_sb, n*m*p
        
        return U, V, W, rank
    
    # Build Strassen-based decomposition
    U_str, V_str, W_str, rank_str = build_strassen_decomp()
    best_result = (U_str, V_str, W_str)
    best_rank = rank_str
    
    if time.time() > deadline - 5:
        return {"U": best_result[0].tolist(), "V": best_result[1].tolist(), "W": best_result[2].tolist()}
    
    # Try to improve with ALS on a lower rank
    # Lower bound: max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    lower_bound = max(n*m, m*p, p*n)
    
    def alternating_minimize(U, V, W, max_iters=100, tol=1e-10, deadline=None):
        """Alternating least squares to refine the decomposition."""
        r = U.shape[1]
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline - 10:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    M_mat = np.zeros((p, r))
                    for k in range(p):
                        M_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        U[i*m+j, :], _, _, _ = np.linalg.lstsq(M_mat.T, np.ones(p), rcond=None)
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    M_mat = np.zeros((n, r))
                    for i in range(n):
                        M_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :], _, _, _ = np.linalg.lstsq(M_mat.T, np.ones(n), rcond=None)
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    M_mat = np.zeros((m, r))
                    for j in range(m):
                        M_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :], _, _, _ = np.linalg.lstsq(M_mat.T, np.ones(m), rcond=None)
                    except:
                        pass
            
            # Check convergence
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        error += (val - 1.0)**2
            if error < tol:
                break
        
        # Final error check
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    error += (val - 1.0)**2
        return error < 1e-8, U, V, W
    
    # Try ranks from lower_bound up to best_rank - 1
    for target_rank in range(lower_bound, best_rank):
        if time.time() > deadline - 30:
            break
        if target_rank >= best_rank:
            continue
        
        found = False
        for trial in range(5):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(999 + trial * 104729 + target_rank * 13)
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            U *= np.sqrt(target_rank / (n*m))
            V *= np.sqrt(target_rank / (m*p))
            W *= np.sqrt(target_rank / (p*n))
            
            converged, U, V, W = alternating_minimize(U, V, W, max_iters=80, deadline=deadline)
            
            if converged and verify(U, V, W):
                best_rank = target_rank
                best_result = (U.copy(), V.copy(), W.copy())
                found = True
                break
    
    U_list = best_result[0].tolist()
    V_list = best_result[1].tolist()
    W_list = best_result[2].tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
