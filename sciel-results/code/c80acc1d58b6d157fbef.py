import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def _build_target_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T

def _reconstruct(U, V, W):
    return np.einsum('ur,vr,wr->uvw', U, V, W)

def _error(U, V, W, T):
    return np.linalg.norm(_reconstruct(U, V, W) - T)

def _local_search(U, V, W, T, values, max_iter, timeout, temperature=1.0, cooling=0.999):
    """
    Discrete Simulated Annealing / Local Search.
    Optimized for speed by using flat indexing and minimal overhead.
    """
    start_time = time.time()
    current_err = _error(U, V, W, T)
    best_err = current_err
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    n_u, n_v, n_w = U.size, V.size, W.size
    total_vars = n_u + n_v + n_w
    
    for it in range(max_iter):
        if time.time() - start_time > timeout:
            break
        if current_err < 1e-12:
            break
        
        # Pick a random variable
        var_idx = random.randint(0, total_vars - 1)
        
        if var_idx < n_u:
            mat, idx = U, var_idx
        elif var_idx < n_u + n_v:
            mat, idx = V, var_idx - n_u
        else:
            mat, idx = W, var_idx - n_u - n_v
        
        old_val = mat.flat[idx]
        
        # Find best new value among candidates
        # We check all values in 'values'
        best_new_val = old_val
        best_new_err = current_err
        
        for new_val in values:
            if abs(new_val - old_val) < 1e-9:
                continue
            mat.flat[idx] = new_val
            e = _error(U, V, W, T)
            if e < best_new_err:
                best_new_err = e
                best_new_val = new_val
            # Restore
            mat.flat[idx] = old_val
            
        # Acceptance
        delta = best_new_err - current_err
        if delta < 0:
            mat.flat[idx] = best_new_val
            current_err = best_new_err
            if current_err < best_err:
                best_err = current_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        else:
            if temperature > 0.01:
                # Avoid overflow in exp
                if delta < 50 * temperature: 
                    if random.random() < np.exp(-delta / temperature):
                        mat.flat[idx] = best_new_val
                        current_err = best_new_err
            temperature *= cooling
            
    return best_err, best_U, best_V, best_W

def _try_rank(R, n, m, p, T, time_limit):
    """
    Try to find a rank-R decomposition.
    """
    start_time = time.time()
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # Focus on integers first, as they are most common in optimal small decompositions
    values = [-1.0, 0.0, 1.0]
    
    # Increase restarts significantly
    num_restarts = 50
    time_per_restart = time_limit / num_restarts
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        # Random sparse initialization
        # Density 0.5
        for col in range(R):
            for i in range(n * m):
                if random.random() < 0.5:
                    U[i, col] = random.choice([-1.0, 1.0])
            for i in range(m * p):
                if random.random() < 0.5:
                    V[i, col] = random.choice([-1.0, 1.0])
            for i in range(p * n):
                if random.random() < 0.5:
                    W[i, col] = random.choice([-1.0, 1.0])
        
        # Run local search
        err, U, V, W = _local_search(U, V, W, T, values, max_iter=20000, timeout=time_per_restart, temperature=2.0, cooling=0.998)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U, V, W
            
        if best_err < 1e-12:
            break
            
    if best_err < 1e-12:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)
    
    # Schoolbook fallback: rank 24
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    best_result = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    # Prioritize Rank 17 (theoretical minimum for 2x3x4)
    # If 17 is found, we are optimal.
    
    # Try Rank 17 with most of the time
    time_for_17 = 100.0
    if time.time() - start_time < deadline - time_for_17:
        result = _try_rank(17, n, m, p, T, time_for_17)
        if result is not None:
            U, V, W = result
            if _error(U, V, W, T) < 1e-12:
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                return best_result

    # Try Rank 18
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    if remaining > 5.0:
        time_for_18 = min(30.0, remaining)
        result = _try_rank(18, n, m, p, T, time_for_18)
        if result is not None:
            U, V, W = result
            if _error(U, V, W, T) < 1e-12:
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                return best_result

    # Try Rank 19
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    if remaining > 5.0:
        time_for_19 = min(15.0, remaining)
        result = _try_rank(19, n, m, p, T, time_for_19)
        if result is not None:
            U, V, W = result
            if _error(U, V, W, T) < 1e-12:
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                return best_result

    return best_result
