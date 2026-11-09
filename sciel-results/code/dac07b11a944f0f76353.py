import numpy as np
import time
import random
import sys

DIMENSIONS = [3, 4, 5]
DEADLINE = 140.0

def _schoolbook():
    """
    Construct the standard schoolbook rank-60 decomposition.
    Iterates over output indices (i, k) and inner dimension j.
    W follows the k*n+i convention.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(W[0]) if W else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Vectorized: for each c, outer product
    T = np.zeros((n, m, p))
    for c in range(rank):
        u_col = U_arr[:, c].reshape(n, m)
        v_col = V_arr[:, c].reshape(m, p)
        w_col = W_arr[:, c].reshape(p, n)
        # T[i,j,k] += u_col[i,j] * v_col[j,k] * w_col[k,i]
        for i in range(n):
            for j in range(m):
                T[i, j, :] += u_col[i, j] * v_col[j, :] * w_col[:, i]
    
    if np.max(np.abs(T - 1.0)) > tol:
        return False
    return True

def _residual_vectorized(U_arr, V_arr, W_arr, n, m, p):
    """Compute sum of squared residuals efficiently."""
    rank = U_arr.shape[1]
    T = np.zeros((n, m, p))
    for c in range(rank):
        u_col = U_arr[:, c].reshape(n, m)
        v_col = V_arr[:, c].reshape(m, p)
        w_col = W_arr[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                T[i, j, :] += u_col[i, j] * v_col[j, :] * w_col[:, i]
    return np.sum((T - 1.0) ** 2)

def _residual_single_term(U_arr, V_arr, W_arr, n, m, p, c):
    """Compute contribution of a single term to the tensor."""
    u_col = U_arr[:, c].reshape(n, m)
    v_col = V_arr[:, c].reshape(m, p)
    w_col = W_arr[:, c].reshape(p, n)
    T_c = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            T_c[i, j, :] = u_col[i, j] * v_col[j, :] * w_col[:, i]
    return T_c

def _gradient_descent_smooth(U, V, W, n, m, p, lr=0.01, max_iters=2000, time_limit=10.0, seed=0):
    """
    Smooth optimization using gradient descent on the continuous relaxation.
    Uses the fact that the objective is smooth in each factor.
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    U_arr = np.array(U, dtype=float).copy()
    V_arr = np.array(V, dtype=float).copy()
    W_arr = np.array(W, dtype=float).copy()
    rank = U_arr.shape[1]
    
    # Precompute current tensor
    T = np.zeros((n, m, p))
    for c in range(rank):
        T += _residual_single_term(U_arr, V_arr, W_arr, n, m, p, c)
    
    best_res = _residual_vectorized(U_arr, V_arr, W_arr, n, m, p)
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random term
        c = rng.randint(0, rank)
        
        # Compute gradient w.r.t. U[:, c]
        # d(T[i,j,k])/d(U[i*m+j, c]) = V[j*p+k, c] * W[k*n+i, c]
        # So grad_U[i*m+j] = sum_k V[j*p+k,c] * W[k*n+i,c] * 2*(T[i,j,k]-1)
        grad_U = np.zeros(n * m)
        grad_V = np.zeros(m * p)
        grad_W = np.zeros(p * n)
        
        u_col = U_arr[:, c].reshape(n, m)
        v_col = V_arr[:, c].reshape(m, p)
        w_col = W_arr[:, c].reshape(p, n)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    err = 2.0 * (T[i, j, k] - 1.0)
                    grad_U[i * m + j] += err * v_col[j, k] * w_col[k, i]
                    grad_V[j * p + k] += err * u_col[i, j] * w_col[k, i]
                    grad_W[k * n + i] += err * u_col[i, j] * v_col[j, k]
        
        # Adaptive learning rate
        scale = lr / (1.0 + 0.01 * it)
        
        U_arr[:, c] -= scale * grad_U
        V_arr[:, c] -= scale * grad_V
        W_arr[:, c] -= scale * grad_W
        
        # Update T incrementally
        # Delta T = sum of changes
        # For simplicity, recompute T for this term and update
        old_T_c = _residual_single_term(U_arr, V_arr, W_arr, n, m, p, c)
        # Actually we need to subtract old and add new
        # Let's just recompute the full T periodically
        if it % 50 == 0:
            T = np.zeros((n, m, p))
            for cc in range(rank):
                T += _residual_single_term(U_arr, V_arr, W_arr, n, m, p, cc)
        
        new_res = _residual_vectorized(U_arr, V_arr, W_arr, n, m, p)
        if new_res < best_res:
            best_res = new_res
            best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
        
        if best_res < 1e-10:
            break
    
    return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_res

def _round_and_verify(U, V, W, n, m, p, tol=1e-6):
    """Try rounding to integers/half-integers and verify."""
    # Try rounding to nearest integer
    U_int = np.round(np.array(U, dtype=float))
    V_int = np.round(np.array(V, dtype=float))
    W_int = np.round(np.array(W, dtype=float))
    
    if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist(), n, m, p, tol):
        return U_int.tolist(), V_int.tolist(), W_int.tolist()
    
    # Try rounding to nearest half-integer
    U_half = np.round(np.array(U, dtype=float) * 2) / 2
    V_half = np.round(np.array(V, dtype=float) * 2) / 2
    W_half = np.round(np.array(W, dtype=float) * 2) / 2
    
    if _verify(U_half.tolist(), V_half.tolist(), W_half.tolist(), n, m, p, tol):
        return U_half.tolist(), V_half.tolist(), W_half.tolist()
    
    return None

def _try_rank_reduction(U, V, W, n, m, p, target_rank, time_limit=15.0, seed=0):
    """
    Try to reduce rank by dropping columns and optimizing the remaining ones.
    """
    start_time = time.time()
    r_current = len(W[0])
    
    if r_current <= target_rank:
        return U, V, W
    
    U_arr = np.array(U, dtype=float).copy()
    V_arr = np.array(V, dtype=float).copy()
    W_arr = np.array(W, dtype=float).copy()
    
    best_res = _residual_vectorized(U_arr, V_arr, W_arr, n, m, p)
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    
    # Greedy: try removing one column at a time
    columns = list(range(r_current))
    random.Random(seed).shuffle(columns)
    
    for col_idx in columns:
        if time.time() - start_time > time_limit:
            break
        if r_current - 1 < target_rank:
            break
        
        # Save column
        saved_U = U_arr[:, col_idx].copy()
        saved_V = V_arr[:, col_idx].copy()
        saved_W = W_arr[:, col_idx].copy()
        
        # Zero out column
        U_arr[:, col_idx] = 0
        V_arr[:, col_idx] = 0
        W_arr[:, col_idx] = 0
        
        new_res = _residual_vectorized(U_arr, V_arr, W_arr, n, m, p)
        
        if new_res < 1e-6:
            # Can keep it zeroed
            r_current -= 1
            if new_res < best_res:
                best_res = new_res
                best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
        else:
            # Restore
            U_arr[:, col_idx] = saved_U
            V_arr[:, col_idx] = saved_V
            W_arr[:, col_idx] = saved_W
    
    if r_current <= target_rank and best_res < 1e-6:
        # Find non-zero columns
        non_zero = []
        for c in range(r_current):
            if np.any(best_U[:, c] != 0) or np.any(best_V[:, c] != 0) or np.any(best_W[:, c] != 0):
                non_zero.append(c)
        
        if len(non_zero) <= target_rank:
            return best_U[:, non_zero].tolist(), best_V[:, non_zero].tolist(), best_W[:, non_zero].tolist()
    
    return None

def _construct_from_seed(n, m, p, r, seed):
    """
    Construct a decomposition using a structured approach.
    Initialize with random values and optimize.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize with small random values
    U = rng.randn(n * m, r) * 0.5
    V = rng.randn(m * p, r) * 0.5
    W = rng.randn(p * n, r) * 0.5
    
    return U.tolist(), V.tolist(), W.tolist()

def _optimize_and_round(U, V, W, n, m, p, time_limit=10.0, seed=0):
    """
    Optimize continuously then try to round to valid values.
    """
    start_time = time.time()
    
    # Run gradient descent
    U_opt, V_opt, W_opt, res = _gradient_descent_smooth(U, V, W, n, m, p, 
                                                         lr=0.005, max_iters=3000,
                                                         time_limit=time_limit, seed=seed)
    
    if time.time() - start_time > time_limit:
        return None
    
    # Try to round
    result = _round_and_verify(U_opt, V_opt, W_opt, n, m, p)
    if result is not None:
        return result
    
    # If continuous solution is very good, try local integer search around it
    if res < 1e-4:
        U_rounded = np.round(np.array(U_opt, dtype=float))
        V_rounded = np.round(np.array(V_opt, dtype=float))
        W_rounded = np.round(np.array(W_opt, dtype=float))
        
        # Local search: flip individual entries
        best_U, best_V, best_W = U_rounded.copy(), V_rounded.copy(), W_rounded.copy()
        best_res = _residual_vectorized(best_U, best_V, best_W, n, m, p)
        
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 0.5:
            for _ in range(200):
                if time.time() - start_time > time_limit:
                    break
                
                factor = np.random.randint(0, 3)
                if factor == 0:
                    idx = np.random.randint(0, n * m)
                    arr = best_U
                elif factor == 1:
                    idx = np.random.randint(0, m * p)
                    arr = best_V
                else:
                    idx = np.random.randint(0, p * n)
                    arr = best_W
                
                col = np.random.randint(0, len(arr[0]))
                old_val = arr[idx, col]
                new_val = old_val + np.random.choice([-1, 1])
                if abs(new_val) > 3:
                    continue
                
                arr[idx, col] = new_val
                new_res = _residual_vectorized(best_U, best_V, best_W, n, m, p)
                
                if new_res < best_res:
                    best_res = new_res
                else:
                    arr[idx, col] = old_val
                
                if best_res < 1e-10:
                    break
        
        if best_res < 1e-8:
            if _verify(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p):
                return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook as fallback
    best = _schoolbook()
    best_rank = n * m * p
    
    # Strategy 1: Try to reduce rank from schoolbook
    U_sb, V_sb, W_sb = best["U"], best["V"], best["W"]
    
    for target_r in range(best_rank - 1, max(40, best_rank - 10), -1):
        if time.time() - start_time > DEADLINE - 40:
            break
        time_remaining = DEADLINE - 40 - (time.time() - start_time)
        if time_remaining < 2.0:
            break
        
        time_per_try = min(8.0, time_remaining / 3.0)
        result = _try_rank_reduction(U_sb, V_sb, W_sb, n, m, p, target_r, 
                                      time_limit=time_per_try, seed=target_r)
        if result is not None:
            U, V, W = result
            best = {"U": U, "V": V, "W": W}
            best_rank = len(W[0])
            U_sb, V_sb, W_sb = U, V, W
    
    # Strategy 2: Try to find decompositions at lower ranks using optimization
    for r in range(min(best_rank - 1, 55), max(35, best_rank - 15), -1):
        if time.time() - start_time > DEADLINE - 20:
            break
        time_remaining = DEADLINE - 20 - (time.time() - start_time)
        if time_remaining < 1.0:
            break
        
        time_per_rank = min(5.0, time_remaining / 2.0)
        
        for attempt in range(3):
            if time.time() - start_time > DEADLINE - 15:
                break
            
            U_init, V_init, W_init = _construct_from_seed(n, m, p, r, seed=r * 100 + attempt)
            result = _optimize_and_round(U_init, V_init, W_init, n, m, p,
                                          time_limit=time_per_rank / 3.0,
                                          seed=r * 100 + attempt)
            if result is not None:
                U, V, W = result
                best = {"U": U, "V": V, "W": W}
                best_rank = r
                break
    
    # Final safety check
    if time.time() - start_time > DEADLINE - 5:
        return _schoolbook()
    
    return best
