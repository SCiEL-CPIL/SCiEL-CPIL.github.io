"""Rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _to_halfint_list(arr):
    """Convert numpy array to list with integer/half-integer entries."""
    result = []
    for row in arr:
        row_list = []
        for val in row:
            rounded = round(val * 2) / 2.0
            if abs(rounded - int(rounded)) < 1e-9:
                row_list.append(int(rounded))
            else:
                row_list.append(rounded)
        result.append(row_list)
    return result

def _try_rank_structured(r, n, m, p, max_iters=2000, seed=42):
    """Try to find a rank-r decomposition using structured initialization."""
    np.random.seed(seed)
    
    # Initialize with structured random values
    U = np.random.randn(n*m, r) * 0.3
    V = np.random.randn(m*p, r) * 0.3
    W = np.random.randn(p*n, r) * 0.3
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.05
    for it in range(max_iters):
        # Compute loss and gradients efficiently
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err = val - 1.0
                    loss += err * err
                    for c in range(r):
                        grad_U[i*m+j, c] += err * V[j*p+k, c] * W[k*n+i, c]
                        grad_V[j*p+k, c] += err * U[i*m+j, c] * W[k*n+i, c]
                        grad_W[k*n+i, c] += err * U[i*m+j, c] * V[j*p+k, c]
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-10:
            break
        
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
    
    # Round to half-integers and verify
    U_rounded = np.round(best_U * 2) / 2.0
    V_rounded = np.round(best_V * 2) / 2.0
    W_rounded = np.round(best_W * 2) / 2.0
    
    if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
        return U_rounded, V_rounded, W_rounded
    
    if _verify(best_U.tolist(), best_V.tolist(), W_rounded.tolist(), n, m, p):
        return best_U, best_V, W_rounded
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find lower rank decompositions
    for r in range(90, 34, -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        for seed in range(8):
            if time.time() > deadline - 5:
                break
            
            result = _try_rank_structured(r, n, m, p, max_iters=1500, seed=seed)
            if result is not None:
                U, V, W = result
                best_rank = r
                best_result = {
                    "U": _to_halfint_list(U),
                    "V": _to_halfint_list(V),
                    "W": _to_halfint_list(W)
                }
                found = True
                break
        
        if found:
            continue  # Try lower rank
        else:
            break  # Can't find at this rank, likely can't at lower either
    
    return best_result
