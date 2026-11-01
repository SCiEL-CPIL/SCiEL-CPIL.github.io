import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    U_size = n * m  # 6
    V_size = m * p  # 9
    W_size = p * n  # 6
    
    # Build target tensor
    T = np.zeros((U_size, V_size, W_size))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    # Precompute index mappings
    U_indices = []
    for i in range(n):
        for j in range(m):
            u = i * m + j
            eqs = []
            for k in range(p):
                v = j * p + k
                w = k * n + i
                eqs.append((v, w))
            U_indices.append((i, j, u, eqs))
            
    V_indices = []
    for j in range(m):
        for k in range(p):
            v = j * p + k
            eqs = []
            for i in range(n):
                u = i * m + j
                w = k * n + i
                eqs.append((u, w))
            V_indices.append((j, k, v, eqs))
            
    W_indices = []
    for k in range(p):
        for i in range(n):
            w = k * n + i
            eqs = []
            for j in range(m):
                u = i * m + j
                v = j * p + k
                eqs.append((u, v))
            W_indices.append((k, i, w, eqs))

    def verify_solution(U, V, W):
        # Check if U, V, W are half-integers
        if not np.allclose(U * 2, np.round(U * 2), atol=1e-9): return False
        if not np.allclose(V * 2, np.round(V * 2), atol=1e-9): return False
        if not np.allclose(W * 2, np.round(W * 2), atol=1e-9): return False
        
        # Check exact decomposition
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[u, r] * V[v, r] * W[w, r]
                    if not np.isclose(s, 1.0, atol=1e-9):
                        return False
        return True

    def compute_error(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[u, r] * V[v, r] * W[w, r]
                    err += (s - 1.0) ** 2
        return err

    def als_decompose(U_init, V_init, W_init, R, max_iters=300, tol=1e-12):
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        prev_err = np.inf
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            # Update U
            for i, j, u, eqs in U_indices:
                B = np.zeros((len(eqs), R))
                for idx, (v, w) in enumerate(eqs):
                    B[idx, :] = V[v, :] * W[w, :]
                U_u, _, _, _ = np.linalg.lstsq(B, np.ones(len(eqs)), rcond=None)
                U[u, :] = U_u
                
            # Update V
            for j, k, v, eqs in V_indices:
                C = np.zeros((len(eqs), R))
                for idx, (u, w) in enumerate(eqs):
                    C[idx, :] = U[u, :] * W[w, :]
                V_v, _, _, _ = np.linalg.lstsq(C, np.ones(len(eqs)), rcond=None)
                V[v, :] = V_v
                
            # Update W
            for k, i, w, eqs in W_indices:
                D = np.zeros((len(eqs), R))
                for idx, (u, v) in enumerate(eqs):
                    D[idx, :] = U[u, :] * V[v, :]
                W_w, _, _, _ = np.linalg.lstsq(D, np.ones(len(eqs)), rcond=None)
                W[w, :] = W_w
            
            # Calculate error
            err = compute_error(U, V, W)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        
        U, V, W = best_U, best_V, best_W
        
        # Round to half-integers and verify
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        
        if verify_solution(U_round, V_round, W_round):
            return U_round, V_round, W_round
            
        return None

    def discrete_local_search(U, V, W, R, max_iters=500, time_budget=5.0):
        """Local search on half-integer values"""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        best_err = compute_error(U, V, W)
        start_time = time.time()
        
        # Define possible values: -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2
        possible_vals = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
        
        for it in range(max_iters):
            if time.time() - start_time > time_budget:
                break
            
            # Pick a random matrix and entry
            mat_idx = np.random.randint(3)
            if mat_idx == 0:
                mat = U
                size = U_size
            elif mat_idx == 1:
                mat = V
                size = V_size
            else:
                mat = W
                size = W_size
            
            row = np.random.randint(size)
            col = np.random.randint(R)
            
            old_val = mat[row, col]
            # Try changing to a nearby half-integer
            if np.random.random() < 0.5:
                # Small perturbation
                delta = np.random.choice([-0.5, 0.5])
                new_val = old_val + delta
            else:
                # Try a random value
                new_val = np.random.choice(possible_vals)
            
            if abs(new_val) > 2.0:
                continue
            
            mat[row, col] = new_val
            new_err = compute_error(U, V, W)
            
            if new_err < best_err:
                best_err = new_err
            else:
                # Revert
                mat[row, col] = old_val
        
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        
        if verify_solution(U_round, V_round, W_round):
            return U_round, V_round, W_round
        return None

    def try_rank(R, time_budget=15.0):
        """Try to find a decomposition of rank R"""
        start_time = time.time()
        
        # Multiple initialization strategies
        strategies = [
            'random',
            'structured',
            'sparse_init'
        ]
        
        for strat in strategies:
            if time.time() - start_time > time_budget:
                break
            
            num_restarts = 20 if R <= 15 else 10
            
            for restart in range(num_restarts):
                if time.time() - start_time > time_budget:
                    break
                
                np.random.seed(hash((R, strat, restart)) % (2**32))
                
                if strat == 'random':
                    U_init = np.random.randn(U_size, R) * 0.3
                    V_init = np.random.randn(V_size, R) * 0.3
                    W_init = np.random.randn(W_size, R) * 0.3
                elif strat == 'structured':
                    # Initialize with some structure based on the tensor
                    U_init = np.zeros((U_size, R))
                    V_init = np.zeros((V_size, R))
                    W_init = np.zeros((W_size, R))
                    for r in range(R):
                        # Random assignment of which tensor entries this rank covers
                        entries = np.random.choice(18, size=max(1, 18//R), replace=False)
                        for e in entries:
                            i, j, k = divmod(e, 9)
                            j, k = divmod(j, 3)
                            i, j = divmod(i, 3)
                            # Actually let's just use random indices
                        u_idx = np.random.randint(U_size)
                        v_idx = np.random.randint(V_size)
                        w_idx = np.random.randint(W_size)
                        U_init[u_idx, r] = 1.0
                        V_init[v_idx, r] = 1.0
                        W_init[w_idx, r] = 1.0
                else:  # sparse_init
                    U_init = np.zeros((U_size, R))
                    V_init = np.zeros((V_size, R))
                    W_init = np.zeros((W_size, R))
                    for r in range(R):
                        # Each rank-1 term covers a few entries
                        num_entries = max(1, 18 // R)
                        for _ in range(num_entries):
                            i = np.random.randint(n)
                            j = np.random.randint(m)
                            k = np.random.randint(p)
                            U_init[i*m+j, r] = 1.0
                            V_init[j*p+k, r] = 1.0
                            W_init[k*n+i, r] = 1.0
                
                res = als_decompose(U_init, V_init, W_init, R, max_iters=200)
                if res is not None:
                    return res
                
                # If ALS didn't work, try discrete local search from the best ALS result
                if res is None and time.time() - start_time < time_budget - 2:
                    # Run ALS to get a good starting point
                    U_ls, V_ls, W_ls = U_init.copy(), V_init.copy(), W_init.copy()
                    prev_err = np.inf
                    for it in range(100):
                        for i, j, u, eqs in U_indices:
                            B = np.zeros((len(eqs), R))
                            for idx, (v, w) in enumerate(eqs):
                                B[idx, :] = V_ls[v, :] * W_ls[w, :]
                            U_u, _, _, _ = np.linalg.lstsq(B, np.ones(len(eqs)), rcond=None)
                            U_ls[u, :] = U_u
                        for j, k, v, eqs in V_indices:
                            C = np.zeros((len(eqs), R))
                            for idx, (u, w) in enumerate(eqs):
                                C[idx, :] = U_ls[u, :] * W_ls[w, :]
                            V_v, _, _, _ = np.linalg.lstsq(C, np.ones(len(eqs)), rcond=None)
                            V_ls[v, :] = V_v
                        for k, i, w, eqs in W_indices:
                            D = np.zeros((len(eqs), R))
                            for idx, (u, v) in enumerate(eqs):
                                D[idx, :] = U_ls[u, :] * V_ls[v, :]
                            W_w, _, _, _ = np.linalg.lstsq(D, np.ones(len(eqs)), rcond=None)
                            W_ls[w, :] = W_w
                        err = compute_error(U_ls, V_ls, W_ls)
                        if abs(prev_err - err) < 1e-10:
                            break
                        prev_err = err
                    
                    res = discrete_local_search(U_ls, V_ls, W_ls, R, max_iters=300, time_budget=3.0)
                    if res is not None:
                        return res
        
        return None

    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((U_size, rank_school))
    V_school = np.zeros((V_size, rank_school))
    W_school = np.zeros((W_size, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    start_time = time.time()
    deadline = start_time + 130  # 20s margin before 150s limit

    # Try ranks from 14 upwards
    for R in [14, 15, 16, 17]:
        if time.time() > deadline:
            break
        
        # Allocate more time for lower ranks
        time_budget = min(30.0, deadline - time.time())
        
        res = try_rank(R, time_budget)
        if res is not None:
            U, V, W = res
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }

    # Fallback to schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
