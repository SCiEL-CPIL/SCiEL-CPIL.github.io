import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    n, m, p = 3, 3, 3
    nm, mp, pn = 9, 9, 9
    
    # Precompute the target tensor indices
    ones_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                ones_indices.append((a, b, c))
    
    ones_set = set(ones_indices)
    all_indices = [(a, b, c) for a in range(nm) for b in range(mp) for c in range(pn)]
    zero_indices = [idx for idx in all_indices if idx not in ones_set]
    # Combine into a single list for vectorized residual calculation
    # Target vector T of size 729
    T = np.zeros(729)
    idx = 0
    for a, b, c in all_indices:
        if (a, b, c) in ones_set:
            T[idx] = 1.0
        idx += 1
        
    # Map (a,b,c) to flat index
    def flat_idx(a, b, c):
        return a * mp * pn + b * pn + c

    # Precompute indices for the 27 ones
    ones_flat = [flat_idx(a, b, c) for a, b, c in ones_indices]
    zeros_flat = [flat_idx(a, b, c) for a, b, c in zero_indices]
    
    def verify(U, V, W, tol=1e-9):
        """
        Verify if U, V, W form a valid decomposition.
        """
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Check ones
        for a, b, c in ones_indices:
            s = np.sum(W[c, :] * U[a, :] * V[b, :])
            if abs(s - 1.0) > tol:
                return False
        # Check zeros
        for a, b, c in zero_indices:
            s = np.sum(W[c, :] * U[a, :] * V[b, :])
            if abs(s) > tol:
                return False
        return True

    def residual(x, R):
        U_f = x[:nm*R].reshape(nm, R)
        V_f = x[nm*R:2*nm*R].reshape(mp, R)
        W_f = x[2*nm*R:3*nm*R].reshape(pn, R)
        
        res = np.zeros(729)
        # Efficient calculation:
        # res[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # We can compute this using einsum or loops.
        # For speed, we can use:
        # T_decomposed = np.einsum('ar,br,cr->abc', U_f, V_f, W_f)
        # But 9x9x9 is small, so einsum is fast.
        T_dec = np.einsum('ar,br,cr->abc', U_f, V_f, W_f)
        res[:] = T_dec.flatten() - T
        return res

    def try_optimize_rank(R, U_init, V_init, W_init, time_budget):
        """
        Attempt to find a valid decomposition of rank R.
        """
        if time.time() > start_time + time_budget:
            return None
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        best_cost = float('inf')
        best_x = x0
        
        max_iter = 3
        for i in range(max_iter):
            if time.time() > start_time + time_budget:
                break
            
            try:
                sol = least_squares(residual, x0, args=(R,), method='trf', max_nfev=2000, ftol=1e-15, xtol=1e-15, gtol=1e-15)
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_x = sol.x
                    if sol.cost < 1e-12:
                        break
            except Exception:
                pass
            
            # Perturb for next restart
            if i < max_iter - 1:
                x0 = best_x + np.random.randn(len(best_x)) * 0.01

        if best_cost < 1e-10:
            U_sol = best_x[:nm*R].reshape(nm, R)
            V_sol = best_x[nm*R:2*nm*R].reshape(mp, R)
            W_sol = best_x[2*nm*R:3*nm*R].reshape(pn, R)
            
            # Try to snap to half-integers
            U_r = np.round(U_sol * 2) / 2
            V_r = np.round(V_sol * 2) / 2
            W_r = np.round(W_sol * 2) / 2
            
            if verify(U_r, V_r, W_r, tol=1e-9):
                return U_r, V_r, W_r
            
            # Try integers
            U_i = np.round(U_sol)
            V_i = np.round(V_sol)
            W_i = np.round(W_sol)
            if verify(U_i, V_i, W_i, tol=1e-9):
                return U_i, V_i, W_i
                
        return None

    # --- Step 1: Construct Rank-26 Solution ---
    R26 = 26
    U26 = np.zeros((nm, R26))
    V26 = np.zeros((mp, R26))
    W26 = np.zeros((pn, R26))
    
    def idx_u(i, j): return i * m + j
    def idx_v(j, k): return j * p + k
    def idx_w(k, i): return k * n + i
    
    # Strassen terms
    U26[idx_u(0,0), 0] = 1; U26[idx_u(1,1), 0] = 1
    V26[idx_v(0,0), 0] = 1; V26[idx_v(1,1), 0] = 1
    W26[idx_w(0,0), 0] = 1; W26[idx_w(1,1), 0] = 1
    
    U26[idx_u(1,0), 1] = 1; U26[idx_u(1,1), 1] = 1
    V26[idx_v(0,0), 1] = 1
    W26[idx_w(0,1), 1] = 1; W26[idx_w(1,1), 1] = -1
    
    U26[idx_u(0,0), 2] = 1
    V26[idx_v(0,1), 2] = 1; V26[idx_v(1,1), 2] = -1
    W26[idx_w(1,0), 2] = 1; W26[idx_w(1,1), 2] = 1
    
    U26[idx_u(1,1), 3] = 1
    V26[idx_v(1,0), 3] = 1; V26[idx_v(0,0), 3] = -1
    W26[idx_w(0,0), 3] = 1; W26[idx_w(0,1), 3] = 1
    
    U26[idx_u(0,0), 4] = 1; U26[idx_u(0,1), 4] = 1
    V26[idx_v(1,1), 4] = 1
    W26[idx_w(0,0), 4] = -1; W26[idx_w(1,0), 4] = 1
    
    U26[idx_u(1,0), 5] = 1; U26[idx_u(0,0), 5] = -1
    V26[idx_v(0,0), 5] = 1; V26[idx_v(0,1), 5] = 1
    W26[idx_w(1,1), 5] = 1
    
    U26[idx_u(0,1), 6] = 1; U26[idx_u(1,1), 6] = -1
    V26[idx_v(1,0), 6] = 1; V26[idx_v(1,1), 6] = 1
    W26[idx_w(0,0), 6] = 1
    
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U26[idx_u(i,j), col] = 1
                V26[idx_v(j,k), col] = 1
                W26[idx_w(k,i), col] = 1
                col += 1
                
    assert col == 26
    
    if not verify(U26, V26, W26):
        # Fallback
        R27 = 27
        U27 = np.zeros((nm, R27))
        V27 = np.zeros((mp, R27))
        W27 = np.zeros((pn, R27))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U27[idx_u(i,j), c] = 1
                    V27[idx_v(j,k), c] = 1
                    W27[idx_w(k,i), c] = 1
                    c += 1
        return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
    
    best_U, best_V, best_W = U26, V26, W26
    best_rank = 26
    
    # --- Step 2: Try to find Rank 25, 24, 23 ---
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        for R in [23, 24, 25]:
            if time.time() > deadline - 5:
                break
                
            found = False
            # Number of restarts
            num_restarts = max(1, int((deadline - time.time()) / 8.0))
            
            for restart in range(num_restarts):
                if time.time() > deadline - 5:
                    break
                    
                np.random.seed(restart * 1000 + R)
                
                # Strategy: Start from Rank 26 solution, project to Rank R
                # Or random init
                if R < 26:
                    # Take first R columns of Rank 26 solution and optimize
                    U_init = U26[:, :R].copy()
                    V_init = V26[:, :R].copy()
                    W_init = W26[:, :R].copy()
                    # Add small noise to avoid exact saddle points
                    U_init += np.random.randn(nm, R) * 0.1
                    V_init += np.random.randn(mp, R) * 0.1
                    W_init += np.random.randn(pn, R) * 0.1
                else:
                    U_init = np.random.randn(nm, R) * 0.5
                    V_init = np.random.randn(mp, R) * 0.5
                    W_init = np.random.randn(pn, R) * 0.5
                    
                res = try_optimize_rank(R, U_init, V_init, W_init, time_budget=6.0)
                if res is not None:
                    best_U, best_V, best_W = res
                    best_rank = R
                    found = True
                    break
                    
            if found:
                break
                
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
