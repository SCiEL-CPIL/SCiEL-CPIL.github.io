import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 145.0

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U,V,W decompose the matmul tensor exactly."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check all target entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U_arr[a, :], np.multiply(V_arr[b, :], W_arr[c, :]))
                if abs(val - 1.0) > tol:
                    return False
    
    # Check that non-target entries are 0
    target_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_set.add((a, b, c))
    
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) not in target_set:
                    val = np.dot(U_arr[a, :], np.multiply(V_arr[b, :], W_arr[c, :]))
                    if abs(val) > tol:
                        return False
    return True

def _construct_rank_36(n, m, p):
    """Schoolbook rank-36 decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _gradient_decompose(n, m, p, r, n_restarts=5, max_iters=500, seed=42):
    """
    Use gradient descent with vectorized operations to find rank-r decomposition.
    Returns (U, V, W) if successful, else None.
    """
    np.random.seed(seed)
    
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))
    
    target_set = set(triples)
    
    # Precompute all non-target indices
    non_targets = []
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) not in target_set:
                    non_targets.append((a, b, c))
    
    best = None
    best_loss = float('inf')
    
    for attempt in range(n_restarts):
        if _time_left() < 5:
            break
        
        U = np.random.randn(n * m, r) * 0.3
        V = np.random.randn(m * p, r) * 0.3
        W = np.random.randn(p * n, r) * 0.3
        
        lr = 0.01
        prev_loss = float('inf')
        
        for it in range(max_iters):
            if _time_left() < 3:
                break
            
            # Compute gradient vectorized
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            loss = 0.0
            
            # Target entries: should be 1
            for (a, b, c) in triples:
                vw = np.multiply(V[b, :], W[c, :])
                u = U[a, :]
                val = np.dot(u, vw)
                err = val - 1.0
                loss += err * err
                grad_U[a, :] += err * vw
                grad_V[b, :] += err * np.multiply(u, W[c, :])
                grad_W[c, :] += err * np.multiply(u, V[b, :])
            
            # Non-target entries: should be 0
            for (a, b, c) in non_targets:
                vw = np.multiply(V[b, :], W[c, :])
                u = U[a, :]
                val = np.dot(u, vw)
                loss += val * val
                grad_U[a, :] += val * vw
                grad_V[b, :] += val * np.multiply(u, W[c, :])
                grad_W[c, :] += val * np.multiply(u, V[b, :])
            
            # Update with learning rate decay
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            if loss < 1e-14:
                break
            if abs(prev_loss - loss) < 1e-12:
                break
            prev_loss = loss
            lr *= 0.995
        
        # Verify
        def _verify_local(Ub, Vb, Wb, n, m, p, tol=1e-9):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        val = np.dot(Ub[a, :], np.multiply(Vb[b, :], Wb[c, :]))
                        if abs(val - 1.0) > tol:
                            return False
            ts = set()
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        ts.add((i*m+j, j*p+k, k*n+i))
            for a in range(n*m):
                for b in range(m*p):
                    for c in range(p*n):
                        if (a,b,c) not in ts:
                            val = np.dot(Ub[a, :], np.multiply(Vb[b, :], Wb[c, :]))
                            if abs(val) > tol:
                                return False
            return True
        
        if _verify_local(U, V, W, n, m, p):
            return (U, V, W)
        
        if loss < best_loss:
            best_loss = loss
            best = (U.copy(), V.copy(), W.copy())
    
    return best

def _construct_direct(n, m, p):
    """
    Try to find a low-rank decomposition directly for the full 2x3x6 tensor.
    """
    # Try ranks from low to high
    for r in range(12, 19):
        if _time_left() < 15:
            break
        result = _gradient_decompose(n, m, p, r, n_restarts=5, max_iters=300, seed=42+r)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def construct():
    n, m, p = 2, 3, 6
    
    # Try direct low-rank decomposition
    if _time_left() > 30:
        result = _construct_direct(n, m, p)
        if result is not None and _verify(result["U"], result["V"], result["W"], n, m, p, tol=1e-9):
            return result
    
    # Fallback to schoolbook rank 36
    return _construct_rank_36(n, m, p)
