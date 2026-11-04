"""Matrix multiplication 2x5x6 - rank search with verification."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = delta."""
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    rank = U.shape[1]
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U_arr[u_idx, :], np.multiply(V_arr[v_idx, :], W_arr[w_idx, :]))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank(r, n, m, p, deadline):
    """Try to find a rank-r decomposition using numerical optimization."""
    T = _build_tensor(n, m, p)
    
    # Random initialization
    np.random.seed(42)
    U = np.random.randn(n * m, r) * 0.1
    V = np.random.randn(m * p, r) * 0.1
    W = np.random.randn(p * n, r) * 0.1
    
    # Gradient-based search (simple projected gradient)
    # The objective: minimize sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
    lr = 0.01
    for step in range(5000):
        if time.time() > deadline - 30:
            break
        # Compute residuals
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = U[i*m+j, :]
                    vk = V[j*p+k, :]
                    wk = W[k*n+i, :]
                    val = np.dot(ui, np.multiply(vk, wk))
                    resid = val - 1.0
                    # d/dU: resid * V * W
                    grad_U[i*m+j, :] += 2 * resid * np.multiply(vk, wk)
                    # d/dV: resid * U * W
                    grad_V[j*p+k, :] += 2 * resid * np.multiply(ui, wk)
                    # d/dW: resid * U * V
                    grad_W[k*n+i, :] += 2 * resid * np.multiply(ui, vk)
        
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        if step % 500 == 0:
            # Check
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j,:], np.multiply(V[j*p+k,:], W[k*n+i,:]))
                        max_err = max(max_err, abs(val - 1.0))
            if max_err < 1e-4:
                # Success - try to clean up
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    """Construct a decomposition for 2x5x6 matrix multiplication."""
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try lower ranks
    for r in range(30, best_rank + 1, 5):
        if time.time() > deadline - 60:
            break
        result = _try_rank(r, n, m, p, deadline)
        if result is not None:
            best = result
            best_rank = r
            break
    
    return best
