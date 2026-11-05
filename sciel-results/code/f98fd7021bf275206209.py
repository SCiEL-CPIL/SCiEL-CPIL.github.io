"""Matrix multiplication 3x3 by 3x7 - rank minimization."""
DIMENSIONS = [3, 3, 7]

import time
import numpy as np

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that the decomposition exactly matches matrix multiplication."""
    U = np.asarray(U)
    V = np.asarray(V)
    W = np.asarray(W)
    R = U.shape[1]
    
    # Check tensor equation: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _build_schoolbook(n, m, p):
    """Build schoolbook decomposition with rank n*m*p."""
    R = n * m * p
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _try_rank_search(n, m, p, target_rank, deadline, seed=42):
    """Try to find a rank-target_rank decomposition using random init + optimization."""
    np.random.seed(seed)
    R = target_rank
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute index triples
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i * m + j, j * p + k, k * n + i))
    
    U = np.random.randn(nm, R) * 0.5
    V = np.random.randn(mp, R) * 0.5
    W = np.random.randn(pn, R) * 0.5
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.01
    for it in range(300):
        if time.time() > deadline:
            break
        
        # Compute loss and gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        loss = 0.0
        
        for (iu, iv, iw) in triples:
            val = np.dot(U[iu, :], V[iv, :] * W[iw, :])
            err = val - 1.0
            loss += err * err
            dU[iu, :] += 2 * err * (V[iv, :] * W[iw, :])
            dV[iv, :] += 2 * err * (U[iu, :] * W[iw, :])
            dW[iw, :] += 2 * err * (U[iu, :] * V[iv, :])
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Adaptive LR
        if it > 0 and loss > best_loss * 1.2:
            lr *= 0.5
        elif it > 5 and loss < best_loss * 0.8:
            lr *= 1.1
        
        if loss < 1e-12:
            break
    
    # Try rounding to half-integers
    for scale in [2, 1]:  # try half-integers and integers
        U_r = np.round(best_U * scale) / scale
        V_r = np.round(best_V * scale) / scale
        W_r = np.round(best_W * scale) / scale
        
        if _verify_decomposition(n, m, p, U_r, V_r, W_r):
            return U_r, V_r, W_r
    
    # Try with the unrounded values if they're close enough
    if best_loss < 1e-8:
        if _verify_decomposition(n, m, p, best_U, best_V, best_W):
            return best_U, best_V, best_W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    # Start with schoolbook as fallback
    best_U, best_V, best_W = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower-rank decompositions
    # Lower bound: rank >= max(n, m, p) = 7, but realistically higher
    # Try a range of ranks
    ranks_to_try = [21, 24, 27, 30, 33, 36, 40, 45, 50, 54, 57, 60]
    
    for r in ranks_to_try:
        if r >= best_rank:
            break
        if time.time() > deadline - 5:
            break
        
        result = _try_rank_search(n, m, p, r, deadline, seed=42 + r)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = r
            break  # Found a valid decomposition, stop
    
    # Verify and return
    if _verify_decomposition(n, m, p, best_U, best_V, best_W):
        return {
            "U": np.asarray(best_U).tolist(),
            "V": np.asarray(best_V).tolist(),
            "W": np.asarray(best_W).tolist()
        }
    
    # Fallback to schoolbook
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
