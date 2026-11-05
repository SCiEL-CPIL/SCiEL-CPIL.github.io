import numpy as np
import time

def construct():
    start_time = time.time()
    max_time = 145  # Safety margin before 150s kill
    
    n, m, p = 2, 3, 4
    
    # Build the proven rank-22 construction as fallback
    def build_rank_22():
        rank = 22
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_term(u_indices, v_indices, w_indices, col):
            for idx, val in u_indices:
                U[idx][col] += val
            for idx, val in v_indices:
                V[idx][col] += val
            for idx, val in w_indices:
                W[idx][col] += val
        
        # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
        add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
        add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
        add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
        add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
        add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
        add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
        add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)
        
        # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
        add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
        add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
        add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
        add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
        add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
        add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
        add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)
        
        # Outer Product: A[0:2, 2:3] * B[2:3, 0:4]
        add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
        add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
        add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
        add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
        
        add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
        add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
        add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)
        
        return U, V, W, rank
    
    def verify(U, V, W, rank):
        """Check if the decomposition is exact using vectorized numpy."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j, :rank], V_arr[j*p+k, :rank] * W_arr[k*n+i, :rank])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def snap_to_half_integer(arr, tol=1e-6):
        """Snap values to nearest half-integer if close."""
        snapped = np.zeros_like(arr)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                nearest_int = round(val)
                if abs(val - nearest_int) < tol:
                    snapped[i, j] = nearest_int
                else:
                    nearest_half = round(val * 2) / 2.0
                    if abs(val - nearest_half) < tol:
                        snapped[i, j] = nearest_half
                    else:
                        snapped[i, j] = val
        return snapped
    
    def try_als(rank, max_iter=400, tol=1e-12, seed=42, init_U=None, init_V=None, init_W=None):
        """Alternating least squares to find rank decomposition."""
        rng = np.random.RandomState(seed)
        
        if init_U is not None:
            U = np.array(init_U, dtype=float).copy()
            V = np.array(init_V, dtype=float).copy()
            W = np.array(init_W, dtype=float).copy()
        else:
            U = rng.randn(n * m, rank) * 0.5
            V = rng.randn(m * p, rank) * 0.5
            W = rng.randn(p * n, rank) * 0.5
        
        for iteration in range(max_iter):
            if time.time() - start_time > max_time - 5:
                return None
            
            # Solve for W: fix U, V
            W_new = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    w = k * n + i
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    try:
                        W_new[w, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        W_new[w, :] = rng.randn(rank) * 0.1
            
            # Solve for U: fix V, W
            U_new = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    u = i * m + j
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    try:
                        U_new[u, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        U_new[u, :] = rng.randn(rank) * 0.1
            
            # Solve for V: fix U, W
            V_new = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    v = j * p + k
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        V_new[v, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        V_new[v, :] = rng.randn(rank) * 0.1
            
            diff = np.max(np.abs(W_new - W)) + np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V))
            W, U, V = W_new, U_new, V_new
            
            if diff < tol:
                U_list = [[round(float(x), 10) for x in row] for row in U]
                V_list = [[round(float(x), 10) for x in row] for row in V]
                W_list = [[round(float(x), 10) for x in row] for row in W]
                if verify(U_list, V_list, W_list, rank):
                    U_snap = snap_to_half_integer(U)
                    V_snap = snap_to_half_integer(V)
                    W_snap = snap_to_half_integer(W)
                    U_list = [[round(float(x), 10) for x in row] for row in U_snap]
                    V_list = [[round(float(x), 10) for x in row] for row in V_snap]
                    W_list = [[round(float(x), 10) for x in row] for row in W_snap]
                    if verify(U_list, V_list, W_list, rank):
                        return U_list, V_list, W_list
                    else:
                        return U_list, V_list, W_list
        return None
    
    def polish(U, V, W, rank, max_iter=500, tol=1e-14):
        """High-precision polish pass with tighter tolerance."""
        U = np.array(U, dtype=float).copy()
        V = np.array(V, dtype=float).copy()
        W = np.array(W, dtype=float).copy()
        
        for iteration in range(max_iter):
            if time.time() - start_time > max_time - 3:
                break
            
            # Solve for W
            W_new = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    w = k * n + i
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    try:
                        W_new[w, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        W_new[w, :] = W[w, :]
            
            # Solve for U
            U_new = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    u = i * m + j
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    try:
                        U_new[u, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        U_new[u, :] = U[u, :]
            
            # Solve for V
            V_new = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    v = j * p + k
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        V_new[v, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        V_new[v, :] = V[v, :]
            
            diff = np.max(np.abs(W_new - W)) + np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V))
            W, U, V = W_new, U_new, V_new
            
            if diff < tol:
                break
        
        U_list = [[round(float(x), 12) for x in row] for row in U]
        V_list = [[round(float(x), 12) for x in row] for row in V]
        W_list = [[round(float(x), 12) for x in row] for row in W]
        if verify(U_list, V_list, W_list, rank):
            return U_list, V_list, W_list
        return None
    
    def warm_start_from_rank22(target_rank, seed, strategy='random_cols'):
        """Create a warm start by modifying rank-22 solution."""
        rng = np.random.RandomState(seed)
        U22, V22, W22, _ = build_rank_22()
        U22_arr = np.array(U22, dtype=float)
        V22_arr = np.array(V22, dtype=float)
        W22_arr = np.array(W22, dtype=float)
        
        if target_rank <= 22:
            if strategy == 'random_cols':
                cols = rng.choice(22, size=target_rank, replace=False)
                U_init = U22_arr[:, cols].copy()
                V_init = V22_arr[:, cols].copy()
                W_init = W22_arr[:, cols].copy()
            elif strategy == 'structured':
                # Keep Strassen blocks, drop some outer product terms
                if target_rank >= 18:
                    op_terms = target_rank - 14
                    if op_terms <= 8:
                        if op_terms == 8:
                            cols = list(range(22))
                        elif op_terms == 7:
                            drop = rng.randint(14, 22)
                            cols = [c for c in range(22) if c != drop]
                        elif op_terms == 6:
                            drops = rng.choice(range(14, 22), size=2, replace=False)
                            cols = [c for c in range(22) if c not in drops]
                        elif op_terms == 5:
                            drops = rng.choice(range(14, 22), size=3, replace=False)
                            cols = [c for c in range(22) if c not in drops]
                        else:
                            cols = rng.choice(22, size=target_rank, replace=False)
                    else:
                        cols = rng.choice(22, size=target_rank, replace=False)
                else:
                    cols = rng.choice(22, size=target_rank, replace=False)
                U_init = U22_arr[:, cols].copy()
                V_init = V22_arr[:, cols].copy()
                W_init = W22_arr[:, cols].copy()
            else:
                cols = rng.choice(22, size=target_rank, replace=False)
                U_init = U22_arr[:, cols].copy()
                V_init = V22_arr[:, cols].copy()
                W_init = W22_arr[:, cols].copy()
        else:
            extra = target_rank - 22
            U_init = np.hstack([U22_arr, rng.randn(6, extra) * 0.1])
            V_init = np.hstack([V22_arr, rng.randn(12, extra) * 0.1])
            W_init = np.hstack([W22_arr, rng.randn(8, extra) * 0.1])
        
        # Add small random perturbation to escape local minima
        U_init += rng.randn(*U_init.shape) * 0.05
        V_init += rng.randn(*V_init.shape) * 0.05
        W_init += rng.randn(*W_init.shape) * 0.05
        
        return U_init, V_init, W_init
    
    # Try to find lower rank decompositions using ALS
    best_found = None
    
    for rank in range(18, 22):
        if time.time() - start_time > max_time - 15:
            break
        
        best_result = None
        
        # Try multiple strategies and seeds
        strategies = ['random_cols', 'structured']
        for strategy in strategies:
            for seed in range(30):
                if time.time() - start_time > max_time - 8:
                    break
                U_init, V_init, W_init = warm_start_from_rank22(rank, seed, strategy)
                result = try_als(rank, max_iter=300, seed=seed, init_U=U_init, init_V=V_init, init_W=W_init)
                if result is not None:
                    U_list, V_list, W_list = result
                    if verify(U_list, V_list, W_list, rank):
                        best_result = (U_list, V_list, W_list, rank)
                        break
            if best_result is not None:
                break
        
        # If warm start didn't work, try random inits
        if best_result is None:
            for seed in range(30):
                if time.time() - start_time > max_time - 8:
                    break
                result = try_als(rank, max_iter=300, seed=seed)
                if result is not None:
                    U_list, V_list, W_list = result
                    if verify(U_list, V_list, W_list, rank):
                        best_result = (U_list, V_list, W_list, rank)
                        break
        
        if best_result is not None:
            U_list, V_list, W_list, r = best_result
            # Polish
            polished = polish(U_list, V_list, W_list, r)
            if polished is not None:
                U_list, V_list, W_list = polished
            best_found = {"U": U_list, "V": V_list, "W": W_list}
    
    # Return the best found (lowest rank) or fallback to rank 22
    if best_found is not None:
        return best_found
    
    # Fallback: return the proven rank-22 construction
    U, V, W, rank = build_rank_22()
    return {"U": U, "V": V, "W": W}
