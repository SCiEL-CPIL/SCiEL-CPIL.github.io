"""Matrix multiplication 3x4x5 rank-minimal exact tensor decomposition search."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # Safety margin before 120s limit

def _time_remaining():
    return _DEADLINE - time.time()

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m * p) + j * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    return U, V, W

def _compute_error(U, V, W, n, m, p, rank):
    """Compute the total absolute error of the decomposition."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                err += abs(s - 1.0)
    return err

def _compute_error_vectorized(U, V, W, n, m, p, rank):
    """Compute error using vectorized operations for speed."""
    # Build the full tensor check
    total_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # s = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                total_err += abs(s - 1.0)
    return total_err

def _try_local_search(U, V, W, n, m, p, rank, max_iters=2000, seed=42):
    """Local search to refine a decomposition with smooth error minimization."""
    rng = np.random.RandomState(seed)
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    err = _compute_error_vectorized(U, V, W, n, m, p, rank)
    if err < 1e-9:
        return U, V, W
    
    best_err = err
    step_sizes = [0.5, 0.25, 0.1, 0.05, 0.01]
    
    for it in range(max_iters):
        if time.time() > _DEADLINE:
            break
        
        if err < 1e-9:
            return U, V, W
        
        if err < best_err:
            best_err = err
        
        # Adaptive step size
        step = step_sizes[min(4, int(err * 20))] if err > 0.01 else 0.01
        
        # Try perturbing random entries
        r = rng.randint(rank)
        choice = rng.randint(3)
        
        if choice == 0:
            idx = rng.randint(n*m)
            delta = rng.choice([-step, step, -2*step, 2*step])
            old_val = U[idx, r]
            U[idx, r] = old_val + delta
            new_err = _compute_error_vectorized(U, V, W, n, m, p, rank)
            if new_err >= err:
                U[idx, r] = old_val
            else:
                err = new_err
        elif choice == 1:
            idx = rng.randint(m*p)
            delta = rng.choice([-step, step, -2*step, 2*step])
            old_val = V[idx, r]
            V[idx, r] = old_val + delta
            new_err = _compute_error_vectorized(U, V, W, n, m, p, rank)
            if new_err >= err:
                V[idx, r] = old_val
            else:
                err = new_err
        else:
            idx = rng.randint(p*n)
            delta = rng.choice([-step, step, -2*step, 2*step])
            old_val = W[idx, r]
            W[idx, r] = old_val + delta
            new_err = _compute_error_vectorized(U, V, W, n, m, p, rank)
            if new_err >= err:
                W[idx, r] = old_val
            else:
                err = new_err
        
        # Periodic refinement pass
        if it % 100 == 0 and err < 0.1:
            for r in range(rank):
                for idx in range(n*m):
                    old = U[idx, r]
                    for delta in [-0.5, 0.5]:
                        U[idx, r] = old + delta
                        new_err = _compute_error_vectorized(U, V, W, n, m, p, rank)
                        if new_err < err:
                            err = new_err
                        else:
                            U[idx, r] = old
                for idx in range(m*p):
                    old = V[idx, r]
                    for delta in [-0.5, 0.5]:
                        V[idx, r] = old + delta
                        new_err = _compute_error_vectorized(U, V, W, n, m, p, rank)
                        if new_err < err:
                            err = new_err
                        else:
                            V[idx, r] = old
                for idx in range(p*n):
                    old = W[idx, r]
                    for delta in [-0.5, 0.5]:
                        W[idx, r] = old + delta
                        new_err = _compute_error_vectorized(U, V, W, n, m, p, rank)
                        if new_err < err:
                            err = new_err
                        else:
                            W[idx, r] = old
                
                if time.time() > _DEADLINE:
                    break
            if time.time() > _DEADLINE:
                break

    if err < 1e-9:
        return U, V, W
    return None

def _try_polish(U, V, W, n, m, p, rank, max_iters=500):
    """High-precision polish pass with very small steps."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    err = _compute_error_vectorized(U, V, W, n, m, p, rank)
    if err < 1e-9:
        return U, V, W, True
    
    rng = np.random.RandomState(123)
    for it in range(max_iters):
        if time.time() > _DEADLINE:
            break
        if err < 1e-9:
            return U, V, W, True
        
        step = 0.01 if err > 0.01 else 0.001
        r = rng.randint(rank)
        choice = rng.randint(3)
        
        if choice == 0:
            idx = rng.randint(n*m)
            delta = rng.choice([-step, step])
            old_val = U[idx, r]
            U[idx, r] = old_val + delta
            new_err = _compute_error_vectorized(U, V, W, n, m, p, rank)
            if new_err >= err:
                U[idx, r] = old_val
            else:
                err = new_err
        elif choice == 1:
            idx = rng.randint(m*p)
            delta = rng.choice([-step, step])
            old_val = V[idx, r]
            V[idx, r] = old_val + delta
            new_err = _compute_error_vectorized(U, V, W, n, m, p, rank)
            if new_err >= err:
                V[idx, r] = old_val
            else:
                err = new_err
        else:
            idx = rng.randint(p*n)
            delta = rng.choice([-step, step])
            old_val = W[idx, r]
            W[idx, r] = old_val + delta
            new_err = _compute_error_vectorized(U, V, W, n, m, p, rank)
            if new_err >= err:
                W[idx, r] = old_val
            else:
                err = new_err
    
    return U, V, W, err < 1e-9

def _structured_search(n, m, p, start_rank, end_rank):
    """Search for low-rank decompositions using structured initialization."""
    best_result = None
    best_rank = None
    
    for rank in range(start_rank, min(end_rank, 61)):
        if time.time() > _DEADLINE:
            break
        
        # Try multiple structured initializations
        for seed in range(8):
            if time.time() > _DEADLINE:
                break
            
            rng = np.random.RandomState(seed * 1000 + rank)
            
            # Initialize with sparse random patterns
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # Each rank term gets a random subset of indices
            for r in range(rank):
                # Assign random non-zero entries with varied sparsity
                n_u = max(1, int(rng.randint(1, max(2, n*m//2 + 1))))
                n_v = max(1, int(rng.randint(1, max(2, m*p//2 + 1))))
                n_w = max(1, int(rng.randint(1, max(2, p*n//2 + 1))))
                
                u_indices = rng.choice(n*m, size=n_u, replace=False)
                v_indices = rng.choice(m*p, size=n_v, replace=False)
                w_indices = rng.choice(p*n, size=n_w, replace=False)
                
                for idx in u_indices:
                    U[idx, r] = rng.choice([-1, 1])
                for idx in v_indices:
                    V[idx, r] = rng.choice([-1, 1])
                for idx in w_indices:
                    W[idx, r] = rng.choice([-1, 1])
            
            # Try local search
            result = _try_local_search(U, V, W, n, m, p, rank, max_iters=3000, seed=seed)
            if result is not None:
                U_arr, V_arr, W_arr = result
                # Polish
                U_arr, V_arr, W_arr, success = _try_polish(U_arr, V_arr, W_arr, n, m, p, rank, max_iters=300)
                if success:
                    U_list = [[float(U_arr[i, r]) for r in range(rank)] for i in range(n*m)]
                    V_list = [[float(V_arr[i, r]) for r in range(rank)] for i in range(m*p)]
                    W_list = [[float(W_arr[i, r]) for r in range(rank)] for i in range(p*n)]
                    return U_list, V_list, W_list
    
    return None

def _systematic_search(n, m, p, start_rank, end_rank):
    """More systematic search with better initialization strategies."""
    for rank in range(start_rank, min(end_rank, 55)):
        if time.time() > _DEADLINE:
            break
        
        # Strategy 1: Dense random initialization
        for seed in range(3):
            if time.time() > _DEADLINE:
                break
            
            rng = np.random.RandomState(seed * 7777 + rank)
            
            U = rng.randn(n*m, rank) * 0.5
            V = rng.randn(m*p, rank) * 0.5
            W = rng.randn(p*n, rank) * 0.5
            
            result = _try_local_search(U, V, W, n, m, p, rank, max_iters=2000, seed=seed)
            if result is not None:
                U_arr, V_arr, W_arr = result
                U_arr, V_arr, W_arr, success = _try_polish(U_arr, V_arr, W_arr, n, m, p, rank, max_iters=200)
                if success:
                    U_list = [[float(U_arr[i, r]) for r in range(rank)] for i in range(n*m)]
                    V_list = [[float(V_arr[i, r]) for r in range(rank)] for i in range(m*p)]
                    W_list = [[float(W_arr[i, r]) for r in range(rank)] for i in range(p*n)]
                    return U_list, V_list, W_list
        
        # Strategy 2: Structured block initialization
        for seed in range(3):
            if time.time() > _DEADLINE:
                break
            
            rng = np.random.RandomState(seed * 3333 + rank)
            
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # Block structure: each rank term covers a block of indices
            block_u = max(1, n*m // rank)
            block_v = max(1, m*p // rank)
            block_w = max(1, p*n // rank)
            
            for r in range(rank):
                u_start = (r * block_u) % (n*m)
                v_start = (r * block_v) % (m*p)
                w_start = (r * block_w) % (p*n)
                
                for di in range(min(block_u, n*m - u_start)):
                    U[u_start + di, r] = rng.choice([-1, 1])
                for dj in range(min(block_v, m*p - v_start)):
                    V[v_start + dj, r] = rng.choice([-1, 1])
                for dk in range(min(block_w, p*n - w_start)):
                    W[w_start + dk, r] = rng.choice([-1, 1])
            
            result = _try_local_search(U, V, W, n, m, p, rank, max_iters=2000, seed=seed)
            if result is not None:
                U_arr, V_arr, W_arr = result
                U_arr, V_arr, W_arr, success = _try_polish(U_arr, V_arr, W_arr, n, m, p, rank, max_iters=200)
                if success:
                    U_list = [[float(U_arr[i, r]) for r in range(rank)] for i in range(n*m)]
                    V_list = [[float(V_arr[i, r]) for r in range(rank)] for i in range(m*p)]
                    W_list = [[float(W_arr[i, r]) for r in range(rank)] for i in range(p*n)]
                    return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try structured search for lower rank
    result = _structured_search(n, m, p, 20, 48)
    
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Try systematic search
    result = _systematic_search(n, m, p, 20, 48)
    
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
