import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Target tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
                
    # Schoolbook solution
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U_sb[u, col] = 1
                V_sb[v, col] = 1
                W_sb[w, col] = 1
                col += 1
                
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    best_rank = rank_schoolbook
    
    start_time = time.time()
    deadline = start_time + 100
    
    # Try ranks from 47 down to 40
    for r in range(47, 39, -1):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for trial in range(5):
            if time.time() > deadline:
                break
                
            np.random.seed(123 + trial)
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # ALS
            for it in range(100):
                if time.time() > deadline:
                    break
                    
                # Update U
                # T_u = sum_alpha U[u, alpha] (V_alpha \otimes W_alpha)
                # M_u is (m*p*p*n) x r
                # We can build M_u once for each u?
                # M_u[:, alpha] = (V[:, alpha] * W[:, alpha])? No, it's the outer product of the vectors V_alpha and W_alpha?
                # No, V_alpha is a vector of size m*p. W_alpha is a vector of size p*n.
                # The term is V_alpha \otimes W_alpha?
                # The tensor term is U[u, alpha] * V[:, alpha] * W[:, alpha]?
                # No, the tensor is T[u, v, w] = sum_alpha U[u, alpha] V[v, alpha] W[w, alpha].
                # So for fixed u, T[u, v, w] = sum_alpha U[u, alpha] (V[v, alpha] W[w, alpha]).
                # Let Z_alpha[v, w] = V[v, alpha] W[w, alpha].
                # Then T[u, :, :] = sum_alpha U[u, alpha] Z_alpha.
                # We want to find U[u, :] such that this holds.
                # This is a linear system: M_u X = T_u_vec
                # M_u[:, alpha] = vec(Z_alpha)
                
                for u in range(n*m):
                    # Build M_u
                    M_u = np.zeros((m*p*p*n, r))
                    for alpha in range(r):
                        # Z_alpha is (m*p, p*n)
                        # vec(Z_alpha) is (m*p*p*n,)
                        Z_alpha = np.outer(V[:, alpha], W[:, alpha])
                        M_u[:, alpha] = Z_alpha.flatten()
                        
                    T_u_vec = T[u, :, :].flatten()
                    # Solve M_u U[u, :] = T_u_vec
                    U[u, :], residuals, rank_u, sv = np.linalg.lstsq(M_u, T_u_vec, rcond=None)
                
                # Update V
                for v in range(m*p):
                    M_v = np.zeros((n*m*p*n, r))
                    for alpha in range(r):
                        Z_alpha = np.outer(U[:, alpha], W[:, alpha])
                        M_v[:, alpha] = Z_alpha.flatten()
                    T_v_vec = T[:, v, :].flatten()
                    V[v, :], residuals, rank_v, sv = np.linalg.lstsq(M_v, T_v_vec, rcond=None)
                
                # Update W
                for w in range(p*n):
                    M_w = np.zeros((n*m*m*p, r))
                    for alpha in range(r):
                        Z_alpha = np.outer(U[:, alpha], V[:, alpha])
                        M_w[:, alpha] = Z_alpha.flatten()
                    T_w_vec = T[:, :, w].flatten()
                    W[w, :], residuals, rank_w, sv = np.linalg.lstsq(M_w, T_w_vec, rcond=None)
                
                # Check error
                # T_approx = sum_alpha U[:, alpha] \otimes V[:, alpha] \otimes W[:, alpha]
                # Error = || T - T_approx ||_F
                # This is expensive to compute fully.
                # We can compute the Frobenius norm of the residual.
                # || T - sum U V W ||_F^2
                # We can estimate it.
                
                # If error is small, we might have a solution.
                # But we need exact integer/half-integer entries?
                # The task says "integer/half-integer entries".
                # The ALS solution will be floating point.
                # We need to round to the nearest half-integer.
                # Then check if the rounded solution is exact.
                
                # Let's check the error after rounding.
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Compute error of rounded solution
                # T_approx = sum_alpha U_rounded[:, alpha] \otimes V_rounded[:, alpha] \otimes W_rounded[:, alpha]
                # We can compute this for a few random indices to check.
                # Or compute the full tensor.
                
                # Full tensor computation
                T_approx = np.zeros((n*m, m*p, p*n))
                for alpha in range(r):
                    T_approx += np.einsum('u,v,w->uvw', U_rounded[:, alpha], V_rounded[:, alpha], W_rounded[:, alpha])
                
                error = np.linalg.norm(T - T_approx)
                
                if error < 1e-9:
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    best_rank = r
                    break
            else:
                continue
            break
            
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
