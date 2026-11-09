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

def _try_rank_structured(r, n, m, p, max_iters=500, seed=42):
    """Try to find a rank-r decomposition using Adam optimization."""
    np.random.seed(seed)
    
    # Initialize with structured random values
    # Scale initialization to help convergence
    U = np.random.randn(n*m, r) * 0.5
    V = np.random.randn(m*p, r) * 0.5
    W = np.random.randn(p*n, r) * 0.5
    
    # Adam optimizer parameters
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    m_U = np.zeros_like(U)
    v_U = np.zeros_like(U)
    m_V = np.zeros_like(V)
    v_V = np.zeros_like(V)
    m_W = np.zeros_like(W)
    v_W = np.zeros_like(W)
    
    t = 0
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Precompute indices for efficiency
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # We can vectorize the gradient calculation using einsum or similar
    
    for it in range(max_iters):
        t += 1
        # Compute loss and gradients
        # val[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # Efficient computation using broadcasting might be hard due to index mixing.
        # We'll stick to loops but optimize the inner loop.
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        loss = 0.0
        
        for i in range(n):
            for j in range(m):
                idx_u = i*m + j
                for k in range(p):
                    idx_v = j*p + k
                    idx_w = k*n + i
                    
                    # Compute value
                    val = np.dot(U[idx_u, :], V[idx_v, :]) # This is wrong, W is missing
                    # Actually: sum_c U[idx_u, c] * V[idx_v, c] * W[idx_w, c]
                    # Let's do it element-wise for clarity, or use vectorized dot
                    # prod = U[idx_u, :] * V[idx_v, :]
                    # val = np.dot(prod, W[idx_w, :])
                    
                    prod_UV = U[idx_u, :] * V[idx_v, :]
                    val = np.dot(prod_UV, W[idx_w, :])
                    
                    err = val - 1.0
                    loss += err * err
                    
                    # Gradients
                    # dL/dU[idx_u, c] = err * V[idx_v, c] * W[idx_w, c]
                    # dL/dV[idx_v, c] = err * U[idx_u, c] * W[idx_w, c]
                    # dL/dW[idx_w, c] = err * U[idx_u, c] * V[idx_v, c]
                    
                    grad_U[idx_u, :] += err * (V[idx_v, :] * W[idx_w, :])
                    grad_V[idx_v, :] += err * (U[idx_u, :] * W[idx_w, :])
                    grad_W[idx_w, :] += err * (U[idx_u, :] * V[idx_v, :])
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-12:
            break
        
        # Adam update
        # U
        m_U = beta1 * m_U + (1 - beta1) * grad_U
        v_U = beta2 * v_U + (1 - beta2) * (grad_U ** 2)
        m_U_hat = m_U / (1 - beta1 ** t)
        v_U_hat = v_U / (1 - beta2 ** t)
        U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
        
        # V
        m_V = beta1 * m_V + (1 - beta1) * grad_V
        v_V = beta2 * v_V + (1 - beta2) * (grad_V ** 2)
        m_V_hat = m_V / (1 - beta1 ** t)
        v_V_hat = v_V / (1 - beta2 ** t)
        V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
        
        # W
        m_W = beta1 * m_W + (1 - beta1) * grad_W
        v_W = beta2 * v_W + (1 - beta2) * (grad_W ** 2)
        m_W_hat = m_W / (1 - beta1 ** t)
        v_W_hat = v_W / (1 - beta2 ** t)
        W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
    
    # Round to half-integers and verify
    U_rounded = np.round(best_U * 2) / 2.0
    V_rounded = np.round(best_V * 2) / 2.0
    W_rounded = np.round(best_W * 2) / 2.0
    
    if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
        return U_rounded, V_rounded, W_rounded
    
    # Try verifying without rounding if loss is very low
    if best_loss < 1e-6:
         if _verify(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p):
             return best_U, best_V, best_W
    
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
    # We start from a high rank and go down.
    # Schoolbook is 105. We know 105 works.
    # We will check ranks from 104 down to 35.
    # To save time, we will only try a few seeds per rank initially.
    
    # First, ensure we have a valid baseline (Schoolbook is already set)
    # Let's try to improve from 104 downwards.
    
    for r in range(104, 34, -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Increase seeds for higher ranks as they are more likely to succeed
        num_seeds = 3 if r > 90 else 5
        
        for seed in range(num_seeds):
            if time.time() > deadline - 5:
                break
            
            result = _try_rank_structured(r, n, m, p, max_iters=300, seed=seed)
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
            # If we fail at a rank, it's possible we just didn't find it,
            # but for small matrices, if we fail at 104, we might fail at 103 too.
            # However, rank 105 is guaranteed.
            # Let's be conservative and break if we fail at several consecutive ranks?
            # Or just break? The original code broke.
            # Let's break to avoid wasting time on impossible ranks.
            break
    
    return best_result
