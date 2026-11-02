import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def _build_target_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    return T

def _verify(U, V, W, T):
    """Check if U, V, W exactly decompose T."""
    rank = U.shape[1]
    # Compute sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # This is a tensor of shape (n*m, m*p, p*n)
    # We can compute it as: for each r, outer product
    # But that's expensive. Instead, use einsum-like approach.
    # Actually, let's just do it directly for small sizes.
    result = np.zeros(T.shape, dtype=np.int64)
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        # Outer product
        for a in range(len(u)):
            if u[a] != 0:
                for b in range(len(v)):
                    if v[b] != 0:
                        for c in range(len(w)):
                            if w[c] != 0:
                                result[a, b, c] += u[a] * v[b] * w[c]
    return np.array_equal(result, T)

def _verify_fast(U, V, W, T):
    """Faster verification using numpy."""
    rank = U.shape[1]
    # Use tensordot or einsum
    # sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # This equals np.einsum('ar,br,cr->abc', U, V, W)
    result = np.einsum('ar,br,cr->abc', U, V, W)
    return np.array_equal(result, T)

def _try_rank_decomposition(T, rank, max_iters=100, timeout=None):
    """Try to find a rank decomposition using random initialization and optimization."""
    n, m, p = DIMENSIONS
    shape_u = n * m
    shape_v = m * p
    shape_w = p * n
    
    start_time = time.time()
    
    for it in range(max_iters):
        if timeout and (time.time() - start_time) > timeout:
            break
        
        # Random initialization
        U = np.random.randint(-2, 3, size=(shape_u, rank))
        V = np.random.randint(-2, 3, size=(shape_v, rank))
        W = np.random.randint(-2, 3, size=(shape_w, rank))
        
        if _verify_fast(U, V, W, T):
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_target_tensor()
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school), dtype=np.int64)
    V_school = np.zeros((m*p, rank_school), dtype=np.int64)
    W_school = np.zeros((p*n, rank_school), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    start_time = time.time()
    deadline = 140  # Leave 10s margin
    
    # Try to find lower rank
    # Lower bound is max(n*m, m*p, p*n) = 20
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try ranks from lower bound up
    lower_bound = max(n*m, m*p, p*n)
    for rank in range(lower_bound, rank_school):
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 1:
            break
        max_iters = min(50, int(remaining * 10))  # Scale iterations with time
        result = _try_rank_decomposition(T, rank, max_iters=max_iters, timeout=remaining - 1)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = rank
            break  # Found the minimal rank
    
    # Convert to lists for return
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
