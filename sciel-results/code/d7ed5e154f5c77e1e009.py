import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
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
    
    start_time = time.time()
    timeout = 130.0  # Safety margin
    
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
        # Vectorized error calculation
        vals = np.sum(U[u_idxs, :] * V[v_idxs, :] * W[w_idxs, :], axis=1)
        return np.sum((vals - 1.0) ** 2)

    def normalize_columns(X):
        # Normalize columns to unit norm to handle scaling invariance
        norms = np.linalg.norm(X, axis=0)
        norms[norms < 1e-12] = 1.0
        return X / norms

    def try_rank_als(r, time_limit=10.0, seed=42, max_iters=100, num_restarts=20):
        np.random.seed(seed)
        
        # Precompute matrices for linear solves
        # We want to minimize sum_c (U[u_c, :] * V[v_c, :] * W[w_c, :] - 1)^2
        # Let's view this as a linear least squares problem for each variable.
        
        for restart in range(num_restarts):
            if time.time() - start_time > timeout - 5.0:
                return None
            
            # Random initialization
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            # Normalize initial guesses
            U = normalize_columns(U)
            V = normalize_columns(V)
            W = normalize_columns(W)
            
            for it in range(max_iters):
                if time.time() - start_time > timeout - 5.0:
                    return None
                
                # 1. Solve for U
                # For each column c in 0..r-1, we want to find U[:, c] such that
                # sum_c (U[u_idx, c] * (V[v_idx, c] * W[w_idx, c]) - 1)^2 is minimized?
                # Actually, the objective is sum over all constraints c of (sum_k U[u_c, k] V[v_c, k] W[w_c, k] - 1)^2
                # This is not a simple linear LS for U if we sum over k inside the square.
                # However, if we fix V and W, the term inside the square is linear in U.
                # Let y_c = 1. Let z_c = sum_k U[u_c, k] * (V[v_c, k] * W[w_c, k]).
                # We want to minimize sum_c (z_c - 1)^2.
                # This is a least squares problem: A U = b, where A is the constraint matrix.
                # But U has multiple columns. The term is sum_k U[u_c, k] * Q_c_k where Q_c_k = V[v_c, k] * W[w_c, k].
                # So for each constraint c, we have a linear combination of U[u_c, :].
                # We can form a matrix M where M[c, (u_c, k)] = V[v_c, k] * W[w_c, k].
                # Then the predicted value for constraint c is sum_k U[u_c, k] M[c, (u_c, k)]? No, indices are tricky.
                
                # Let's construct the linear system properly.
                # Variables are U[:, k] for k=0..r-1.
                # For a specific constraint c (with indices u, v, w), the contribution to the sum is:
                # sum_{k=0}^{r-1} U[u, k] * (V[v, k] * W[w, k])
                # Let coeff_c_k = V[v_c, k] * W[w_c, k].
                # Then pred_c = sum_k U[u_c, k] * coeff_c_k.
                # We want pred_c = 1.
                # This is a linear system: A * U_flat = 1, where U_flat is U reshaped?
                # No, U is (n*m, r). The variable for constraint c uses row u_c of U.
                # So we can solve for U by building a matrix A of shape (num_constraints, n*m * r)?
                # A[c, u_c * r + k] = coeff_c_k.
                # Then A @ U_flat = 1.
                # Since num_constraints (24) is small, and n*m*r is at most 6*24=144, this is feasible.
                # But we have multiple restarts and iterations.
                
                # Let's build A_U, A_V, A_W once? No, they depend on V, W, U.
                
                # Solve for U:
                # A_U[c, u_c * r + k] = V[v_c, k] * W[w_c, k]
                # A_U is (24, 6*r). U_flat is (6*r,).
                # We solve A_U @ U_flat = 1.
                
                A_U = np.zeros((num_constraints, n * m * r))
                A_V = np.zeros((num_constraints, m * p * r))
                A_W = np.zeros((num_constraints, p * n * r))
                
                # Fill A_U
                for c in range(num_constraints):
                    u_c, v_c, w_c = constraints[c]
                    for k in range(r):
                        A_U[c, u_c * r + k] = V[v_c, k] * W[w_c, k]
                        A_V[c, v_c * r + k] = U[u_c, k] * W[w_c, k]
                        A_W[c, w_c * r + k] = U[u_c, k] * V[v_c, k]
                
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
                
                # Normalize
                U = normalize_columns(U)
                V = normalize_columns(V)
                W = normalize_columns(W)
                
                # Check error
                err = error_function(U, V, W)
                if err < 1e-8:
                    # Round to nearest 0.5
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Verify exactness
                    if error_function(U_rounded, V_rounded, W_rounded) < 1e-10:
                        return U_rounded, V_rounded, W_rounded
                    
                    # If rounding failed, try to polish with gradient descent? 
                    # Or just return the float solution if it's very close? 
                    # The task asks for integer/half-integer. So we must round.
                    # If rounding doesn't work, this solution is invalid for the final output.
                    # But maybe we can use this as a start for a more precise search?
                    # For now, break and try next restart.
                    break
                    
            # If loop finishes without finding a rounded solution, continue to next restart
        return None

    # Try lower ranks
    for r in [23, 22, 21, 20, 19, 18]:
        sol = try_rank_als(r, time_limit=10.0, max_iters=50, num_restarts=50)
        if sol:
            best_U, best_V, best_W = sol
            best_rank = r
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
