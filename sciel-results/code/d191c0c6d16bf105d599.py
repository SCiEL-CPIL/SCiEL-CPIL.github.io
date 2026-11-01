import numpy as np
import time
from itertools import product, combinations

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Seed solution (Rank 24) - schoolbook
    rank_seed = n * m * p
    U_seed = np.zeros((n * m, rank_seed), dtype=np.float64)
    V_seed = np.zeros((m * p, rank_seed), dtype=np.float64)
    W_seed = np.zeros((p * n, rank_seed), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i * m + j, col] = 1
                V_seed[j * p + k, col] = 1
                W_seed[k * n + i, col] = 1
                col += 1

    best_U, best_V, best_W = U_seed.copy(), V_seed.copy(), W_seed.copy()
    best_rank = rank_seed

    # Constraints: (u_idx, v_idx, w_idx)
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                constraints.append((u_idx, v_idx, w_idx))
                
    num_constraints = len(constraints)

    # Pre-compute indices for vectorization
    u_idxs = np.array([c[0] for c in constraints])
    v_idxs = np.array([c[1] for c in constraints])
    w_idxs = np.array([c[2] for c in constraints])

    def error_function(U, V, W):
        vals = np.sum(U[u_idxs, :] * V[v_idxs, :] * W[w_idxs, :], axis=1)
        return np.sum((vals - 1.0) ** 2)

    def check_exact(U, V, W):
        vals = np.sum(U[u_idxs, :] * V[v_idxs, :] * W[w_idxs, :], axis=1)
        return np.allclose(vals, 1.0, atol=1e-9)

    def normalize_columns(X):
        norms = np.linalg.norm(X, axis=0)
        norms[norms < 1e-12] = 1.0
        return X / norms

    def find_valid_scaling(U, V, W, r):
        """
        For each column k, find scalings (s_k, t_k, u_k) such that:
        - s_k * U[:,k], t_k * V[:,k], u_k * W[:,k] are all half-integers
        - s_k * t_k * u_k = 1
        Returns (U_scaled, V_scaled, W_scaled) or None.
        """
        U_out = np.zeros_like(U)
        V_out = np.zeros_like(V)
        W_out = np.zeros_like(W)
        
        # Try a wider range of scaling factors
        scales = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
        
        for k in range(r):
            u_col = U[:, k]
            v_col = V[:, k]
            w_col = W[:, k]
            
            # If column is zero, it's a valid trivial solution for that rank component?
            # No, rank components must contribute. But if ALS found a zero column, it's redundant.
            # We can just leave it as 0.
            if np.allclose(u_col, 0) and np.allclose(v_col, 0) and np.allclose(w_col, 0):
                continue
            
            found = False
            for s in scales:
                u_scaled = s * u_col
                # Check if u_scaled is close to half-integers
                u_rounded = np.round(u_scaled * 2) / 2
                if not np.allclose(u_scaled, u_rounded, atol=1e-5):
                    continue
                
                # If u_rounded is all zeros, skip (unless u_col was zero, handled above)
                if np.allclose(u_rounded, 0):
                    continue

                for t in scales:
                    v_scaled = t * v_col
                    v_rounded = np.round(v_scaled * 2) / 2
                    if not np.allclose(v_scaled, v_rounded, atol=1e-5):
                        continue
                    if np.allclose(v_rounded, 0):
                        continue

                    u_factor = 1.0 / (s * t)
                    w_scaled = u_factor * w_col
                    w_rounded = np.round(w_scaled * 2) / 2
                    if np.allclose(w_scaled, w_rounded, atol=1e-5):
                        if np.allclose(w_rounded, 0):
                            continue
                        
                        U_out[:, k] = u_rounded
                        V_out[:, k] = v_rounded
                        W_out[:, k] = w_rounded
                        found = True
                        break
                if found:
                    break
            
            if not found:
                return None
        
        return U_out, V_out, W_out

    def try_rank_als(r, max_iters=50, num_restarts=100, seed_base=42):
        if time.time() > deadline - 10.0:
            return None
            
        for restart in range(num_restarts):
            if time.time() > deadline - 5.0:
                return None
                
            np.random.seed(seed_base + restart)
            
            # Structured initialization: use the algebraic structure
            # Initialize with small perturbations of a structured pattern
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            # Add some structure: make some entries larger
            for _ in range(3):
                idx = np.random.randint(0, n*m)
                U[idx, :] += 0.5
                idx = np.random.randint(0, m*p)
                V[idx, :] += 0.5
                idx = np.random.randint(0, p*n)
                W[idx, :] += 0.5
            
            # Normalize initial guesses
            U = normalize_columns(U)
            V = normalize_columns(V)
            W = normalize_columns(W)
            
            for it in range(max_iters):
                if time.time() > deadline - 5.0:
                    return None
                
                # Build linear systems for U, V, W
                A_U = np.zeros((num_constraints, n * m * r))
                A_V = np.zeros((num_constraints, m * p * r))
                A_W = np.zeros((num_constraints, p * n * r))
                
                for c in range(num_constraints):
                    u_c, v_c, w_c = constraints[c]
                    idx_u = u_c * r + np.arange(r)
                    idx_v = v_c * r + np.arange(r)
                    idx_w = w_c * r + np.arange(r)
                    
                    A_U[c, idx_u] = V[v_c, :] * W[w_c, :]
                    A_V[c, idx_v] = U[u_c, :] * W[w_c, :]
                    A_W[c, idx_w] = U[u_c, :] * V[v_c, :]
                
                # Solve least squares
                try:
                    U_flat, _, _, _ = np.linalg.lstsq(A_U, np.ones(num_constraints), rcond=None)
                    V_flat, _, _, _ = np.linalg.lstsq(A_V, np.ones(num_constraints), rcond=None)
                    W_flat, _, _, _ = np.linalg.lstsq(A_W, np.ones(num_constraints), rcond=None)
                except np.linalg.LinAlgError:
                    continue
                
                U = U_flat.reshape(n * m, r)
                V = V_flat.reshape(m * p, r)
                W = W_flat.reshape(p * n, r)
                
                # Normalize to keep values in check
                U = normalize_columns(U)
                V = normalize_columns(V)
                W = normalize_columns(W)
                
                # Check error
                err = error_function(U, V, W)
                
                if err < 1e-4:
                    # Try to find valid half-integer decomposition
                    # First, try direct rounding
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    if check_exact(U_r, V_r, W_r):
                        return U_r, V_r, W_r
                    
                    # Try column-wise scaling search
                    scaled = find_valid_scaling(U, V, W, r)
                    if scaled is not None:
                        U_s, V_s, W_s = scaled
                        if check_exact(U_s, V_s, W_s):
                            return U_s, V_s, W_s
                    
                    # Try a few more ALS iterations to refine
                    for _ in range(5):
                        if time.time() > deadline - 5.0:
                            return None
                        A_U = np.zeros((num_constraints, n * m * r))
                        A_V = np.zeros((num_constraints, m * p * r))
                        A_W = np.zeros((num_constraints, p * n * r))
                        
                        for c in range(num_constraints):
                            u_c, v_c, w_c = constraints[c]
                            idx_u = u_c * r + np.arange(r)
                            idx_v = v_c * r + np.arange(r)
                            idx_w = w_c * r + np.arange(r)
                            
                            A_U[c, idx_u] = V[v_c, :] * W[w_c, :]
                            A_V[c, idx_v] = U[u_c, :] * W[w_c, :]
                            A_W[c, idx_w] = U[u_c, :] * V[v_c, :]
                        
                        try:
                            U_flat, _, _, _ = np.linalg.lstsq(A_U, np.ones(num_constraints), rcond=None)
                            V_flat, _, _, _ = np.linalg.lstsq(A_V, np.ones(num_constraints), rcond=None)
                            W_flat, _, _, _ = np.linalg.lstsq(A_W, np.ones(num_constraints), rcond=None)
                        except np.linalg.LinAlgError:
                            break
                        
                        U = U_flat.reshape(n * m, r)
                        V = V_flat.reshape(m * p, r)
                        W = W_flat.reshape(p * n, r)
                        
                        U = normalize_columns(U)
                        V = normalize_columns(V)
                        W = normalize_columns(W)
                        
                        err = error_function(U, V, W)
                        if err < 1e-6:
                            U_r = np.round(U * 2) / 2
                            V_r = np.round(V * 2) / 2
                            W_r = np.round(W * 2) / 2
                            if check_exact(U_r, V_r, W_r):
                                return U_r, V_r, W_r
                            scaled = find_valid_scaling(U, V, W, r)
                            if scaled is not None:
                                U_s, V_s, W_s = scaled
                                if check_exact(U_s, V_s, W_s):
                                    return U_s, V_s, W_s
                            break

        return None

    # Try lower ranks
    for r in [20, 21, 22, 23]:
        if time.time() > deadline - 10.0:
            break
            
        sol = try_rank_als(r, max_iters=40, num_restarts=80, seed_base=r*100)
        if sol:
            best_U, best_V, best_W = sol
            best_rank = r
            break
            
    # If we didn't find a better rank, return the seed
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
