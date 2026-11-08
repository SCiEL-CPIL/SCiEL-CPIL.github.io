import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    # Target ranks to try
    # We know 36 works. Try 35, 34, ...
    # Limit search time
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Build the target tensor T as a flat vector or 3D array
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if i=i', j=j', k=k' else 0
    # Dimensions: (n*m, m*p, p*n)
    # Size: 6 * 18 * 12 = 1296
    
    def get_tensor_flat():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1
        return T.flatten()
    
    T_flat = get_tensor_flat()
    
    def check_exact(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Compute sum u_r v_r w_r
        # Result shape (nm, mp, pn)
        # Check if it equals T
        R = U.shape[1]
        # Efficient computation:
        # T_approx[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
        # We can use einsum
        T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
        T_target = T_flat.reshape(n*m, m*p, p*n)
        # Check exact match
        if np.allclose(T_approx, T_target, atol=1e-9):
            return True
        return False

    def try_rank(R, max_iters=50):
        # Random search
        for _ in range(max_iters):
            if time.time() > deadline:
                return None
            # Random initialization
            # Scale appropriately
            scale = 1.0
            U = np.random.randn(n*m, R) * scale
            V = np.random.randn(m*p, R) * scale
            W = np.random.randn(p*n, R) * scale
            
            # Simple optimization: Gradient Descent
            # Loss = || T - sum u v w ||^2
            # Gradient w.r.t U: 2 * (T - approx) * (V W^T) ? No.
            # Let E = T - approx.
            # dL/dU[i, r] = -2 * sum_{j,k} E[i,j,k] V[j,r] W[k,r]
            
            lr = 0.01
            for it in range(500):
                if time.time() > deadline:
                    return None
                T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
                E = T_flat.reshape(n*m, m*p, p*n) - T_approx
                
                # Check convergence
                err = np.linalg.norm(E)
                if err < 1e-6:
                    # Try to round
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    if check_exact(U_r, V_r, W_r):
                        return U_r, V_r, W_r
                    # Also try integer rounding
                    U_r2 = np.round(U)
                    V_r2 = np.round(V)
                    W_r2 = np.round(W)
                    if check_exact(U_r2, V_r2, W_r2):
                        return U_r2, V_r2, W_r2
                    # If close but not exact, maybe adjust?
                    # For now, just break and try another init
                    break
                    
                # Update U
                # grad_U[i, r] = -2 * sum_{j,k} E[i,j,k] V[j,r] W[k,r]
                # grad_U = -2 * einsum('ijk,jr,kr->ir', E, V, W)
                grad_U = -2 * np.einsum('ijk,jr,kr->ir', E, V, W)
                grad_V = -2 * np.einsum('ijk,ir,kr->jr', E, U, W)
                grad_W = -2 * np.einsum('ijk,ir,jr->kr', E, U, V)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Normalize to prevent blow up
                # Not strictly necessary if lr is small
                
        return None

    # Try ranks
    for R in range(rank_schoolbook - 1, 10, -1):
        if time.time() > deadline:
            break
        res = try_rank(R)
        if res is not None:
            U, V, W = res
            best_result = {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            best_rank = R
            break # Found a better one, stop? Or continue to find even better?
            # The task asks for rank-minimal. We should try to go lower.
            # But finding lower is harder.
            # Let's continue searching for lower ranks if time permits.
            # However, if we found 35, 34 is unlikely to be found quickly.
            # We can just return the first one found to ensure valid output.
            # But "rank-minimal" implies we should try to find the lowest.
            # Given time constraints, returning the first valid improvement is a good strategy.
            # Let's return immediately upon finding any rank < 36.
            return best_result
            
    return best_result
