import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Build the target tensor T
    # U shape (n*m, R), V shape (m*p, R), W shape (p*n, R)
    # T shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    # Schoolbook solution (fallback)
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # ALS Solver
    def solve_als(T, R, max_iters=100, n_restarts=5):
        n_m = T.shape[0]
        m_p = T.shape[1]
        p_n = T.shape[2]
        
        for r in range(n_restarts):
            if time.time() - start_time > 100:
                return None
                
            # Initialize
            U = np.random.randn(n_m, R) * 0.1
            V = np.random.randn(m_p, R) * 0.1
            W = np.random.randn(p_n, R) * 0.1
            
            for it in range(max_iters):
                if time.time() - start_time > 100:
                    return None
                    
                # Update U
                # For each i, U[i, :] = argmin || T[i,:,:] - sum_r U[i,r] * (V[:,r] * W[:,r]) ||^2
                # Flatten T[i,:,:] to vector of size m_p * p_n
                # Matrix A of size (m_p * p_n, R) where A[(j,k), r] = V[j,r] * W[k,r]
                A = np.outer(V.flatten(), W.flatten()).reshape(m_p, p_n, R).transpose(2, 0, 1).reshape(R, m_p * p_n).T
                # A is (m_p * p_n, R)
                # Solve U[:, r] for all r? No, U[i, :] is a row vector.
                # We can solve for all i at once if we stack T.
                # T_flat is (n_m, m_p * p_n)
                T_flat = T.reshape(n_m, m_p * p_n)
                # U = (A^T A)^-1 A^T T_flat^T ?
                # T_flat[i, :] is the vector for row i.
                # We want U[i, :] such that U[i, :] @ A approx T_flat[i, :]
                # So U = T_flat @ A^T @ (A @ A^T)^-1 ?
                # Let's use lstsq for each i or vectorized.
                # U = np.linalg.lstsq(A.T, T_flat.T, rcond=None)[0].T
                try:
                    U = np.linalg.lstsq(A.T, T_flat.T, rcond=None)[0].T
                except:
                    pass
                
                # Update V
                # For each j, V[j, :] = argmin || T[:,j,:] - sum_r V[j,r] * (U[:,r] * W[:,r]) ||^2
                # Flatten T[:,j,:] to vector of size n_m * p_n
                # Matrix B of size (n_m * p_n, R) where B[(i,k), r] = U[i,r] * W[k,r]
                B = np.outer(U.flatten(), W.flatten()).reshape(n_m, p_n, R).transpose(2, 0, 1).reshape(R, n_m * p_n).T
                T_flat_v = T.transpose(1, 0, 2).reshape(m_p, n_m * p_n)
                try:
                    V = np.linalg.lstsq(B.T, T_flat_v.T, rcond=None)[0].T
                except:
                    pass
                
                # Update W
                # For each k, W[k, :] = argmin || T[:,:,k] - sum_r W[k,r] * (U[:,r] * V[:,r]) ||^2
                # Flatten T[:,:,k] to vector of size n_m * m_p
                # Matrix C of size (n_m * m_p, R) where C[(i,j), r] = U[i,r] * V[j,r]
                C = np.outer(U.flatten(), V.flatten()).reshape(n_m, m_p, R).transpose(2, 0, 1).reshape(R, n_m * m_p).T
                T_flat_w = T.transpose(2, 0, 1).reshape(p_n, n_m * m_p)
                try:
                    W = np.linalg.lstsq(C.T, T_flat_w.T, rcond=None)[0].T
                except:
                    pass
                
                # Check for exact integer/half-integer solution
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Reconstruct T
                T_rec = np.zeros_like(T)
                for r_idx in range(R):
                    T_rec += np.einsum('i,j,k->ijk', U_r[:, r_idx], V_r[:, r_idx], W_r[:, r_idx])
                
                if np.allclose(T_rec, T, atol=1e-9):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

        return None

    # Try ranks from lower bound up
    # Lower bound is max(n*m, m*p, p*n) = max(8, 28, 14) = 28
    for R in range(28, 56):
        if time.time() - start_time > 100:
            break
        res = solve_als(T, R, max_iters=50, n_restarts=3)
        if res is not None:
            return res

    return get_schoolbook()
