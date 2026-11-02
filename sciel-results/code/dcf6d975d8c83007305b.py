import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    u_rows = n * m  # 9
    v_rows = m * p  # 15
    w_rows = p * n  # 15

    # Build the multiplication tensor T
    # T[u, v, w] = 1 if u = i*m+j, v = j*p+k, w = k*n+i for some i,j,k
    T = np.zeros((u_rows, v_rows, w_rows), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    def check_exact(U, V, W, tol=1e-8):
        """Check if U, V, W exactly decompose T."""
        T_recon = np.einsum('uc,vc,wc->uvw', U, V, W, optimize=True)
        return np.allclose(T_recon, T, atol=tol, rtol=0)

    def compute_error(U, V, W):
        """Compute max absolute error."""
        T_recon = np.einsum('uc,vc,wc->uvw', U, V, W, optimize=True)
        return np.max(np.abs(T_recon - T))

    def discrete_local_search(U, V, W, max_iters=100):
        """
        Local search in the discrete space of half-integers.
        """
        U = np.array(U, dtype=float).copy()
        V = np.array(V, dtype=float).copy()
        W = np.array(W, dtype=float).copy()
        
        # Round to nearest half-integer
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        current_err = compute_error(U, V, W)
        
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            if current_err < 1e-9:
                break
                
            improved = False
            
            # Try adjusting each entry of U
            for i in range(u_rows):
                if improved:
                    break
                for j in range(U.shape[1]):
                    if time.time() > deadline - 3:
                        break
                    old_val = U[i, j]
                    for delta in [-0.5, 0.5]:
                        new_val = old_val + delta
                        if abs(new_val) > 5:
                            continue
                        U[i, j] = new_val
                        err = compute_error(U, V, W)
                        if err < current_err:
                            current_err = err
                            improved = True
                        else:
                            U[i, j] = old_val
                    if improved:
                        break
            
            if improved:
                continue
            
            # Try adjusting each entry of V
            for i in range(v_rows):
                if improved:
                    break
                for j in range(V.shape[1]):
                    if time.time() > deadline - 3:
                        break
                    old_val = V[i, j]
                    for delta in [-0.5, 0.5]:
                        new_val = old_val + delta
                        if abs(new_val) > 5:
                            continue
                        V[i, j] = new_val
                        err = compute_error(U, V, W)
                        if err < current_err:
                            current_err = err
                            improved = True
                        else:
                            V[i, j] = old_val
                    if improved:
                        break
            
            if improved:
                continue
            
            # Try adjusting each entry of W
            for i in range(w_rows):
                if improved:
                    break
                for j in range(W.shape[1]):
                    if time.time() > deadline - 3:
                        break
                    old_val = W[i, j]
                    for delta in [-0.5, 0.5]:
                        new_val = old_val + delta
                        if abs(new_val) > 5:
                            continue
                        W[i, j] = new_val
                        err = compute_error(U, V, W)
                        if err < current_err:
                            current_err = err
                            improved = True
                        else:
                            W[i, j] = old_val
                    if improved:
                        break
            
            if not improved:
                break
                
        if current_err < 1e-9:
            return U, V, W
        return None

    def try_rank(r, seed=42, max_iter=200):
        """Alternating minimization with multiple initialization strategies."""
        if time.time() > deadline - 5:
            return None

        init_strategies = ['random_small', 'random_medium', 'structured']
        
        for strategy in init_strategies:
            if time.time() > deadline - 5:
                break
                
            np.random.seed(seed)
            
            if strategy == 'random_small':
                U = np.random.randn(u_rows, r) * 0.1
                V = np.random.randn(v_rows, r) * 0.1
                W = np.random.randn(w_rows, r) * 0.1
            elif strategy == 'random_medium':
                U = np.random.randn(u_rows, r) * 0.5
                V = np.random.randn(v_rows, r) * 0.5
                W = np.random.randn(w_rows, r) * 0.5
            else:  # structured
                U = np.zeros((u_rows, r))
                V = np.zeros((v_rows, r))
                W = np.zeros((w_rows, r))
                for c in range(min(r, n * m * p)):
                    i = c % n
                    j = (c // n) % m
                    k = (c // (n * m)) % p
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                # Add small random perturbation
                U += np.random.randn(u_rows, r) * 0.01
                V += np.random.randn(v_rows, r) * 0.01
                W += np.random.randn(w_rows, r) * 0.01

            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break

                # Update U: fix V, W
                # T_recon[u,v,w] = sum_c U[u,c]*V[v,c]*W[w,c]
                # For fixed V, W: minimize sum_{v,w} (T[u,v,w] - sum_c U[u,c]*V[v,c]*W[w,c])^2
                # d/du: -2 * sum_{v,w} T[u,v,w]*V[v,c]*W[w,c] + 2 * sum_{v,w,c'} U[u,c']*V[v,c']*W[w,c']*V[v,c]*W[w,c]
                TW = V[:, None, :] * W[None, :, :]  # (v_rows, w_rows, r)
                h_U = np.einsum('uvw,vwc->uc', T, TW, optimize=True)  # (u_rows, r)
                G_U = np.einsum('vwc,vwd->cd', TW, TW, optimize=True)  # (r, r)
                G_U += 1e-10 * np.eye(r)
                try:
                    U = np.linalg.solve(G_U, h_U.T).T
                except np.linalg.LinAlgError:
                    U = np.linalg.lstsq(G_U, h_U.T, rcond=None)[0].T

                # Update V: fix U, W
                UW = U[:, None, :] * W[None, :, :]  # (u_rows, w_rows, r)
                h_V = np.einsum('uvw,uwc->vc', T, UW, optimize=True)  # (v_rows, r)
                G_V = np.einsum('uwc,uwd->cd', UW, UW, optimize=True)  # (r, r)
                G_V += 1e-10 * np.eye(r)
                try:
                    V = np.linalg.solve(G_V, h_V.T).T
                except np.linalg.LinAlgError:
                    V = np.linalg.lstsq(G_V, h_V.T, rcond=None)[0].T

                # Update W: fix U, V
                UV = U[:, None, :] * V[None, :, :]  # (u_rows, v_rows, r)
                h_W = np.einsum('uvw,uvd->wc', T, UV, optimize=True)  # (w_rows, r)
                G_W = np.einsum('uvd,uve->de', UV, UV, optimize=True)  # (r, r)
                G_W += 1e-10 * np.eye(r)
                try:
                    W = np.linalg.solve(G_W, h_W.T).T
                except np.linalg.LinAlgError:
                    W = np.linalg.lstsq(G_W, h_W.T, rcond=None)[0].T

                # Early exit check
                if it % 10 == 0:
                    T_recon = np.einsum('uc,vc,wc->uvw', U, V, W, optimize=True)
                    if np.allclose(T_recon, T, atol=1e-7, rtol=0):
                        break

            # Try rounding to half-integers and then discrete local search
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            # Check if already exact
            if check_exact(U_r, V_r, W_r):
                return (r, U_r, V_r, W_r)

            # Try discrete local search to fix the error
            result = discrete_local_search(U_r, V_r, W_r)
            if result is not None:
                return (r, result[0], result[1], result[2])

        return None

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(u_rows)]
        V = [[0] * rank for _ in range(v_rows)]
        W = [[0] * rank for _ in range(w_rows)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = schoolbook()
    best_rank = n * m * p

    # Lower bound: max(n*p, n*m, m*p) = max(15, 9, 15) = 15
    # Try ranks from 15 up to schoolbook - 1
    ranks_to_try = list(range(15, best_rank))

    seeds = [42, 0, 1, 2, 3, 7, 13, 99, 123, 456, 789, 1000, 2024, 3141, 2718,
             100, 200, 300, 400, 500, 600, 700, 800, 900]

    for r in ranks_to_try:
        if time.time() > deadline - 5:
            break
        if r >= best_rank:
            break

        for seed in seeds:
            if time.time() > deadline - 5:
                break
            result = try_rank(r, seed)
            if result is not None:
                r_found, U_f, V_f, W_f = result
                if r_found < best_rank:
                    best_rank = r_found
                    best_result = {
                        "U": U_f.tolist(),
                        "V": V_f.tolist(),
                        "W": W_f.tolist()
                    }
                    break  # Found this rank, move to next lower rank

    return best_result
