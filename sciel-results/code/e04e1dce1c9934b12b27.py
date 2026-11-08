import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s safety margin (150s limit)
    
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[(i,j), (j2,k), (k2,i2)] = 1 if j==j2 and i==i2 and k==k2
    # W index: k*n+i, so output index c = k*n+i means i=c%n, k=c//n
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i*m + j
                b_idx = j*p + k
                c_idx = k*n + i  # W[k*n+i]
                T[a_idx, b_idx, c_idx] = 1.0
    
    T_norm = np.linalg.norm(T)
    
    def check_convergence(U, V, W, tol=1e-10):
        """Check if the decomposition is accurate enough."""
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T - T_recon) / T_norm
        return err < tol, err
    
    def polish(U, V, W, max_iters=100, tol=1e-12):
        """High-precision polish pass."""
        for it in range(max_iters):
            if time.time() > deadline:
                break
            # Update U
            M = (V.T @ V) * (W.T @ W)
            rhs = np.einsum('abc,br,cr->ar', T, V, W)
            try:
                U = np.linalg.solve(M.T, rhs.T).T
            except np.linalg.LinAlgError:
                U = np.linalg.lstsq(M.T, rhs.T, rcond=None)[0].T
            
            # Update V
            M_v = (U.T @ U) * (W.T @ W)
            rhs_v = np.einsum('abc,ar,cr->br', T, U, W)
            try:
                V = np.linalg.solve(M_v.T, rhs_v.T).T
            except np.linalg.LinAlgError:
                V = np.linalg.lstsq(M_v.T, rhs_v.T, rcond=None)[0].T
            
            # Update W
            M_w = (U.T @ U) * (V.T @ V)
            rhs_w = np.einsum('abc,ar,br->cr', T, U, V)
            try:
                W = np.linalg.solve(M_w.T, rhs_w.T).T
            except np.linalg.LinAlgError:
                W = np.linalg.lstsq(M_w.T, rhs_w.T, rcond=None)[0].T
            
            conv, err = check_convergence(U, V, W, tol)
            if conv:
                break
        
        return U, V, W
    
    def try_rank(r, max_iters=300, n_restarts=5):
        """Try to find CP decomposition of rank r with multiple initialization strategies."""
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Different initialization strategies
            if restart == 0:
                # Random Gaussian
                U = np.random.randn(n*m, r)
                V = np.random.randn(m*p, r)
                W = np.random.randn(p*n, r)
            elif restart == 1:
                # Random with sparsity encouragement
                U = np.random.randn(n*m, r) * (np.random.rand(n*m, r) > 0.3)
                V = np.random.randn(m*p, r) * (np.random.rand(m*p, r) > 0.3)
                W = np.random.randn(p*n, r) * (np.random.rand(p*n, r) > 0.3)
            elif restart == 2:
                # Binary-like initialization
                U = np.random.choice([-1, 1], size=(n*m, r))
                V = np.random.choice([-1, 1], size=(m*p, r))
                W = np.random.choice([-1, 1], size=(p*n, r))
            elif restart == 3:
                # Mixed: some structured, some random
                U = np.random.randn(n*m, r)
                V = np.random.randn(m*p, r)
                W = np.zeros((p*n, r))
                # Initialize W with pattern based on tensor structure
                for i in range(n):
                    for k in range(p):
                        c_idx = k*n + i
                        W[c_idx, :] = np.random.randn(r)
            else:
                # Scaled random
                scale = 1.0 / np.sqrt(r)
                U = np.random.randn(n*m, r) * scale
                V = np.random.randn(m*p, r) * scale
                W = np.random.randn(p*n, r) * scale
            
            prev_err = np.inf
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Update U
                M = (V.T @ V) * (W.T @ W)
                rhs = np.einsum('abc,br,cr->ar', T, V, W)
                try:
                    U_new = np.linalg.solve(M.T, rhs.T).T
                except np.linalg.LinAlgError:
                    U_new = np.linalg.lstsq(M.T, rhs.T, rcond=None)[0].T
                
                # Update V
                M_v = (U_new.T @ U_new) * (W.T @ W)
                rhs_v = np.einsum('abc,ar,cr->br', T, U_new, W)
                try:
                    V_new = np.linalg.solve(M_v.T, rhs_v.T).T
                except np.linalg.LinAlgError:
                    V_new = np.linalg.lstsq(M_v.T, rhs_v.T, rcond=None)[0].T
                
                # Update W
                M_w = (U_new.T @ U_new) * (V_new.T @ V_new)
                rhs_w = np.einsum('abc,ar,br->cr', T, U_new, V_new)
                try:
                    W_new = np.linalg.solve(M_w.T, rhs_w.T).T
                except np.linalg.LinAlgError:
                    W_new = np.linalg.lstsq(M_w.T, rhs_w.T, rcond=None)[0].T
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence every 10 iterations to save time
                if it % 10 == 0 or it == max_iters - 1:
                    conv, err = check_convergence(U, V, W, 1e-10)
                    if conv:
                        # Polish
                        U, V, W = polish(U, V, W, max_iters=50, tol=1e-12)
                        conv_final, err_final = check_convergence(U, V, W, 1e-10)
                        if conv_final:
                            return U, V, W, True
                    if err < prev_err * 0.5:
                        prev_err = err
                    elif it > 50 and err > prev_err * 1.5:
                        # Diverging, break early
                        break
            
            # Final check after loop
            conv, err = check_convergence(U, V, W, 1e-10)
            if conv:
                U, V, W = polish(U, V, W, max_iters=50, tol=1e-12)
                conv_final, _ = check_convergence(U, V, W, 1e-10)
                if conv_final:
                    return U, V, W, True
        
        return None, None, None, False
    
    # Try ranks from low to high
    # For 3x3x3, best known is 21. For 3x3x5, estimate minimum rank.
    # Start from a reasonable lower bound and go up.
    
    best_result = None
    # Try a focused range - start higher to avoid wasting time on impossible ranks
    for r in range(20, 46):
        if time.time() > deadline:
            break
        U, V, W, success = try_rank(r, max_iters=200, n_restarts=4)
        if success:
            best_result = (r, U, V, W)
            break
    
    if best_result is None:
        # Fallback to schoolbook
        r = n * m * p
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        r = n * m * p
    else:
        r, U, V, W = best_result
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
