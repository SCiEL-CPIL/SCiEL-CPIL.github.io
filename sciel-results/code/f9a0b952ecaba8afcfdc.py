import time
import numpy as np

DIMENSIONS = [3, 3, 5]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition exactly reproduces the matrix multiplication tensor."""
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Build the full tensor T
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for c in range(rank):
        T += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
    
    # Build expected tensor
    T_expected = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_expected[u, v, w] = 1.0
    
    if not np.allclose(T, T_expected, atol=tol):
        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _try_rank(r, n, m, p, deadline, max_iter=500):
    """Try to find a decomposition of rank r using random initialization + gradient descent."""
    U = np.random.randn(n*m, r) * 0.1
    V = np.random.randn(m*p, r) * 0.1
    W = np.random.randn(p*n, r) * 0.1
    
    lr = 0.05
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        # Compute T
        T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for c in range(r):
            T += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        
        # T_target
        T_target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for c in range(r):
            dU[:, c] = np.einsum('ijk,jk->i', T - T_target, np.einsum('j,k->jk', V[:, c], W[:, c]))
            dV[:, c] = np.einsum('ijk,ik->j', T - T_target, np.einsum('i,k->ik', U[:, c], W[:, c]))
            dW[:, c] = np.einsum('ijk,ij->k', T - T_target, np.einsum('i,j->ij', U[:, c], V[:, c]))
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        if it % 50 == 0:
            lr *= 0.95
        
        if np.max(np.abs(T - T_target)) < 1e-6:
            return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Fallback to schoolbook
    U_s, V_s, W_s = _schoolbook(n, m, p)
    best_result = (U_s.tolist(), V_s.tolist(), W_s.tolist(), n*m*p)
    
    # Try to find rank 17
    # We don't have the hardcoded BLS, so we try to search.
    # If search fails, we return schoolbook.
    
    # To improve chances, we might try to use the block structure if we had it.
    # Since we don't, we rely on the search.
    
    # Given the difficulty of finding 17 via search, and the risk of timeout,
    # we might just return the schoolbook if we can't find a better one quickly.
    # However, the task asks for a stronger algorithm.
    
    # Let's try a few random restarts for rank 20, 18, 17.
    for r in [17, 18, 19, 20]:
        if time.time() > deadline - 10:
            break
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            result = _try_rank(r, n, m, p, deadline)
            if result is not None:
                U, V, W = result
                if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    best_result = (U.tolist(), V.tolist(), W.tolist(), r)
                    break
        if best_result[3] < n*m*p:
            break
            
    U, V, W, rank = best_result
    return {"U": U, "V": V, "W": W}
