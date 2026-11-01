import numpy as np
import time
import random

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    target_rank = 50 # Start with schoolbook as fallback
    
    # Precompute valid indices
    valid_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                valid_indices.append((u, v, w))
    
    # Precompute invalid indices (sample for verification)
    all_indices = [(u, v, w) for u in range(nm) for v in range(mp) for w in range(pn)]
    invalid_indices = [(u, v, w) for (u, v, w) in all_indices if (u, v, w) not in set(valid_indices)]
    
    # Deadline guard
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    random.seed(42)
    np.random.seed(42)

    def verify_fast(U, V, W):
        """Fast verification using only valid and sampled invalid indices."""
        r = U.shape[1]
        
        # Check valid indices
        for (u, v, w) in valid_indices:
            s = 0
            for t in range(r):
                s += U[u, t] * V[v, t] * W[w, t]
            if abs(s - 1) > 1e-9:
                return False
        
        # Check a subset of invalid indices (all if small, else sample)
        n_check = min(len(invalid_indices), 500)
        if n_check == len(invalid_indices):
            check_set = invalid_indices
        else:
            check_set = random.sample(invalid_indices, n_check)
        
        for (u, v, w) in check_set:
            s = 0
            for t in range(r):
                s += U[u, t] * V[v, t] * W[w, t]
            if abs(s) > 1e-9:
                return False
        
        return True

    def solve_for_W(U, V, r):
        """Solve for W given U, V using least squares grouped by w."""
        W_sol = np.zeros((pn, r), dtype=float)
        
        for w in range(pn):
            # Build A_w (nm*mp) x r and b_w (nm*mp)
            A_w = np.zeros((nm * mp, r), dtype=float)
            b_w = np.zeros(nm * mp, dtype=float)
            
            idx = 0
            for u in range(nm):
                for v in range(mp):
                    i1, j1 = divmod(u, m)
                    j2, k1 = divmod(v, p)
                    k2, i2 = divmod(w, n)
                    
                    if i1 == i2 and j1 == j2 and k1 == k2:
                        target = 1.0
                    else:
                        target = 0.0
                    
                    b_w[idx] = target
                    for t in range(r):
                        A_w[idx, t] = U[u, t] * V[v, t]
                    idx += 1
            
            sol, residuals, rank, sv = np.linalg.lstsq(A_w, b_w, rcond=None)
            W_sol[w, :] = sol
        
        return W_sol

    def round_to_half_integer(X):
        X2 = np.round(X * 2)
        return X2 / 2.0

    def try_rank(r, max_trials):
        """Try to find a decomposition of rank r."""
        for trial in range(max_trials):
            if time.time() > deadline:
                return None
            
            # Choose initialization strategy
            strategy = trial % 4
            
            if strategy == 0:
                # Random integers in {-1, 0, 1}
                U = np.random.randint(-1, 2, size=(nm, r)).astype(float)
                V = np.random.randint(-1, 2, size=(mp, r)).astype(float)
            elif strategy == 1:
                # Random Gaussian
                U = np.random.randn(nm, r)
                V = np.random.randn(mp, r)
            elif strategy == 2:
                # Random integers in {-2, -1, 0, 1, 2}
                U = np.random.randint(-2, 3, size=(nm, r)).astype(float)
                V = np.random.randint(-2, 3, size=(mp, r)).astype(float)
            else:
                # Structured: some columns are basis vectors
                U = np.zeros((nm, r))
                V = np.zeros((mp, r))
                for t in range(r):
                    if t < nm:
                        U[t, t] = 1.0
                    if t < mp:
                        V[t, t] = 1.0
                    # Add some randomness
                    U[:, t] += np.random.randn(nm) * 0.1
                    V[:, t] += np.random.randn(mp) * 0.1
            
            # Solve for W
            W_float = solve_for_W(U, V, r)
            
            # Round to half-integers
            W_r = round_to_half_integer(W_float)
            
            # Also try rounding U and V to half-integers and re-solving
            U_r = round_to_half_integer(U)
            V_r = round_to_half_integer(V)
            W_r2 = solve_for_W(U_r, V_r, r)
            W_r2 = round_to_half_integer(W_r2)
            
            # Verify both candidates
            if verify_fast(U_r, V_r, W_r2):
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r2.tolist()
                }
            
            if verify_fast(U, V, W_r):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W_r.tolist()
                }
        
        return None

    # Search from known lower bound upward
    # Known lower bound for 2x5x5 is 28 (from literature)
    # Try ranks from 28 to 49
    for r in range(28, 50):
        if time.time() > deadline:
            break
        
        # Adaptive trial count based on time remaining
        time_left = deadline - time.time()
        # Estimate time per trial (~50ms for small r, more for large r)
        est_time_per_trial = 0.05 + r * 0.001
        max_trials = max(10, int(time_left / est_time_per_trial))
        max_trials = min(max_trials, 500)  # Cap to avoid overshooting
        
        result = try_rank(r, max_trials)
        if result is not None:
            return result
    
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

# EVOLVE-BLOCK-END
