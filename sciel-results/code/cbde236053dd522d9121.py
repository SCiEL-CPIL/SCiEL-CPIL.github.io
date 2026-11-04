import time
import numpy as np

DIMENSIONS = [2, 3, 9]
DIMENSIONS_233 = [2, 3, 3]

def _schoolbook():
    """Return schoolbook decomposition with rank 54."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, tol=1e-8):
    """Verify that the decomposition exactly matches the matmul tensor."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U_arr[idx_u, :], np.multiply(V_arr[idx_v, :], W_arr[idx_w, :]))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _verify_233(U, V, W, tol=1e-8):
    """Verify decomposition for 2x3x3."""
    n, m, p = DIMENSIONS_233
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U_arr[idx_u, :], np.multiply(V_arr[idx_v, :], W_arr[idx_w, :]))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank_233(rank, max_iters=2000, tol=1e-7, seed=42):
    """
    Try to find a decomposition of 2x3x3 of given rank.
    """
    n, m, p = DIMENSIONS_233
    nm, mp, pn = n * m, m * p, p * n
    
    np.random.seed(seed)
    
    scale = 1.0 / np.sqrt(rank)
    U = np.random.randn(nm, rank) * scale
    V = np.random.randn(mp, rank) * scale
    W = np.random.randn(pn, rank) * scale
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.05
    momentum = 0.9
    
    vel_U = np.zeros_like(U)
    vel_V = np.zeros_like(V)
    vel_W = np.zeros_like(W)
    
    for it in range(max_iters):
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        loss = 0.0
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    
                    val = np.sum(U[idx_u, :] * V[idx_v, :] * W[idx_w, :])
                    err = val - 1.0
                    loss += err * err
                    
                    grad_U[idx_u, :] += 2.0 * err * V[idx_v, :] * W[idx_w, :]
                    grad_V[idx_v, :] += 2.0 * err * U[idx_u, :] * W[idx_w, :]
                    grad_W[idx_w, :] += 2.0 * err * U[idx_u, :] * V[idx_v, :]
        
        loss /= (n * m * p)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < tol:
            break
        
        vel_U = momentum * vel_U - lr * grad_U
        vel_V = momentum * vel_V - lr * grad_V
        vel_W = momentum * vel_W - lr * grad_W
        
        U += vel_U
        V += vel_V
        W += vel_W
        
        if it % 500 == 499:
            lr *= 0.8
            momentum *= 0.95
    
    if best_loss < 1e-5:
        for mult in [1, 2, 4, 8]:
            U_rounded = np.round(best_U * mult) / mult
            V_rounded = np.round(best_V * mult) / mult
            W_rounded = np.round(best_W * mult) / mult
            
            if _verify_233(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), tol=1e-8):
                return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
        
        if _verify_233(best_U.tolist(), best_V.tolist(), best_W.tolist(), tol=1e-8):
            return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def _expand_233_to_239(U_233, V_233, W_233):
    """
    Expand a 2x3x3 decomposition to a 2x3x9 decomposition.
    """
    n, m, p = DIMENSIONS
    n_233, m_233, p_233 = DIMENSIONS_233
    r_233 = len(U_233[0])
    
    U_239 = [[0] * (3 * r_233) for _ in range(n * m)]
    V_239 = [[0] * (3 * r_233) for _ in range(m * p)]
    W_239 = [[0] * (3 * r_233) for _ in range(p * n)]
    
    for s in range(3):
        for beta in range(r_233):
            alpha = s * r_233 + beta
            
            # U
            for i in range(n):
                for j in range(m):
                    idx_u = i * m + j
                    U_239[idx_u][alpha] = U_233[idx_u][beta]
            
            # V
            for j in range(m):
                for k_prime in range(p_233):
                    k = s * p_233 + k_prime
                    idx_v = j * p + k
                    idx_v_233 = j * p_233 + k_prime
                    V_239[idx_v][alpha] = V_233[idx_v_233][beta]
            
            # W
            for k_prime in range(p_233):
                for i in range(n):
                    k = s * p_233 + k_prime
                    idx_w = k * n + i
                    idx_w_233 = k_prime * n + i
                    W_239[idx_w][alpha] = W_233[idx_w_233][beta]
    
    return U_239, V_239, W_239

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    best_result = _schoolbook()
    best_rank = 54
    
    # Try ranks for 2x3x3 from 9 to 18
    for r_233 in range(9, 19):
        if time.time() > deadline - 10:
            break
            
        for seed in [42, 123, 456, 789, 1024, 2048, 4096]:
            if time.time() > deadline - 5:
                break
                
            res_233 = _try_rank_233(r_233, max_iters=1000, tol=1e-7, seed=seed)
            if res_233 is not None:
                U_233, V_233, W_233 = res_233
                U_239, V_239, W_239 = _expand_233_to_239(U_233, V_233, W_233)
                
                if _verify(U_239, V_239, W_239):
                    rank_239 = 3 * r_233
                    if rank_239 < best_rank:
                        best_result = {"U": U_239, "V": V_239, "W": W_239}
                        best_rank = rank_239
                    break # Found a valid decomposition for this rank, try lower rank for 233
    
    return best_result
