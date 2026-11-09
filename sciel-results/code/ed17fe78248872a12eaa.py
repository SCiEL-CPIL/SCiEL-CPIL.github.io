import time
import numpy as np

def construct():
    n, m, p = 2, 4, 7
    nm, mp, np_ = n * m, m * p, p * n
    
    # Build tensor
    T = np.zeros((nm, mp, np_), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    start_time = time.time()
    DEADLINE = 140
    
    def verify(U, V, W, tol=1e-6):
        """Verify if U, V, W exactly decompose T."""
        # U: nm x r, V: mp x r, W: np_ x r
        r = U.shape[1]
        recon = np.zeros((nm, mp, np_), dtype=np.float64)
        for r_idx in range(r):
            recon += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        if np.allclose(recon, T, atol=tol):
            return True
        return False

    def try_rank(r, max_restarts=20, max_iter=1000):
        """Try to find a rank-r decomposition."""
        for restart in range(max_restarts):
            if time.time() - start_time > DEADLINE - 10:
                return None
            
            np.random.seed(42 + r * 100 + restart)
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(np_, r)
            
            best_residual = float('inf')
            best_U, best_V, best_W = None, None, None
            
            for it in range(max_iter):
                if time.time() - start_time > DEADLINE - 5:
                    break
                
                # Update U
                # T[a,:,:] = sum_r U[a,r] * (V[:,r] * W[:,r])
                # Let M_r = outer(V[:,r], W[:,r]).flatten()
                # T[a,:,:].flatten() = M @ U[a,:]
                # Solve M^T M U[a,:] = M^T T[a,:,:].flatten()
                
                # Construct Design Matrix for U update
                # Shape: (mp * np_, r)
                # This is expensive if done from scratch every time.
                # But r is small (28), mp*np_ is 392.
                D_U = np.zeros((mp * np_, r), dtype=np.float64)
                for r_idx in range(r):
                    D_U[:, r_idx] = np.outer(V[:, r_idx], W[:, r_idx]).flatten()
                
                DT_D_U = D_U.T @ D_U
                try:
                    inv_DT_D_U = np.linalg.inv(DT_D_U)
                except np.linalg.LinAlgError:
                    inv_DT_D_U = np.linalg.pinv(DT_D_U)
                
                for a in range(nm):
                    target = T[a, :, :].flatten()
                    rhs = D_U.T @ target
                    U[a, :] = inv_DT_D_U @ rhs
                    
                # Update V
                # T[:,b,:] = sum_r V[b,r] * (U[:,r] * W[:,r])
                D_V = np.zeros((nm * np_, r), dtype=np.float64)
                for r_idx in range(r):
                    D_V[:, r_idx] = np.outer(U[:, r_idx], W[:, r_idx]).flatten()
                
                DT_D_V = D_V.T @ D_V
                try:
                    inv_DT_D_V = np.linalg.inv(DT_D_V)
                except np.linalg.LinAlgError:
                    inv_DT_D_V = np.linalg.pinv(DT_D_V)
                
                for b in range(mp):
                    target = T[:, b, :].flatten()
                    rhs = D_V.T @ target
                    V[b, :] = inv_DT_D_V @ rhs
                    
                # Update W
                # T[:,:,c] = sum_r W[c,r] * (U[:,r] * V[:,r])
                D_W = np.zeros((nm * mp, r), dtype=np.float64)
                for r_idx in range(r):
                    D_W[:, r_idx] = np.outer(U[:, r_idx], V[:, r_idx]).flatten()
                
                DT_D_W = D_W.T @ D_W
                try:
                    inv_DT_D_W = np.linalg.inv(DT_D_W)
                except np.linalg.LinAlgError:
                    inv_DT_D_W = np.linalg.pinv(DT_D_W)
                
                for c in range(np_):
                    target = T[:, :, c].flatten()
                    rhs = D_W.T @ target
                    W[c, :] = inv_DT_D_W @ rhs
                
                # Check convergence
                if it % 10 == 0:
                    recon = np.zeros((nm, mp, np_), dtype=np.float64)
                    for r_idx in range(r):
                        recon += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
                    residual = np.sum((T - recon) ** 2)
                    if residual < best_residual:
                        best_residual = residual
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if residual < 1e-8:
                        break
            
            if best_U is not None:
                # Try rounding to integers
                U_int = np.round(best_U)
                V_int = np.round(best_V)
                W_int = np.round(best_W)
                if verify(U_int, V_int, W_int, tol=1e-5):
                    return U_int, V_int, W_int
                
                # Try rounding to half-integers
                U_half = np.round(best_U * 2) / 2
                V_half = np.round(best_V * 2) / 2
                W_half = np.round(best_W * 2) / 2
                if verify(U_half, V_half, W_half, tol=1e-5):
                    return U_half, V_half, W_half
                
                # If floating point solution is good enough, return it?
                # The task asks for integer/half-integer entries.
                # If we can't round, we might fail.
                # But if residual is 1e-8, rounding should work if the true solution is integer.
        
        return None

    # Try ranks starting from 28
    for r in range(28, 57):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        result = try_rank(r, max_restarts=50 if r == 28 else 10, max_iter=500)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
