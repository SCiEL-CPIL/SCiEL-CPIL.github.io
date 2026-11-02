import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    deadline = start_time + 140  # safety margin
    
    # First, build the rank-42 Strassen solution
    rank42 = _strassen_rank42(n, m, p)
    U42 = np.array(rank42['U'], dtype=np.float64)
    V42 = np.array(rank42['V'], dtype=np.float64)
    W42 = np.array(rank42['W'], dtype=np.float64)
    
    # Try to find a rank-40 solution
    best_U, best_V, best_W = U42, V42, W42
    best_rank = 42
    
    if time.time() < deadline:
        result = _try_reduce_rank(U42, V42, W42, n, m, p, deadline)
        if result is not None:
            U, V, W, rank = result
            best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Convert to lists of integers/half-integers
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _strassen_rank42(n, m, p):
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(i_start, j_start, k_start, col_offset):
        a_idx = (i_start * m + j_start, i_start * m + j_start + 1,
                 (i_start + 1) * m + j_start, (i_start + 1) * m + j_start + 1)
        e_idx = (j_start * p + k_start, j_start * p + k_start + 1,
                 (j_start + 1) * p + k_start, (j_start + 1) * p + k_start + 1)
        c11_idx = k_start * n + i_start
        c12_idx = (k_start + 1) * n + i_start
        c21_idx = k_start * n + (i_start + 1)
        c22_idx = (k_start + 1) * n + (i_start + 1)
        
        col = col_offset
        # M1 = (a+d)(e+h)
        U[a_idx[0]][col] = 1; U[a_idx[3]][col] = 1
        V[e_idx[0]][col] = 1; V[e_idx[3]][col] = 1
        W[c11_idx][col] = 1; W[c22_idx][col] = 1
        col += 1
        # M2 = (c+d)e
        U[a_idx[2]][col] = 1; U[a_idx[3]][col] = 1
        V[e_idx[0]][col] = 1
        W[c21_idx][col] = 1; W[c22_idx][col] = -1
        col += 1
        # M3 = a(f-h)
        U[a_idx[0]][col] = 1
        V[e_idx[1]][col] = 1; V[e_idx[3]][col] = -1
        W[c12_idx][col] = 1; W[c22_idx][col] = 1
        col += 1
        # M4 = d(g-e)
        U[a_idx[3]][col] = 1
        V[e_idx[2]][col] = 1; V[e_idx[0]][col] = -1
        W[c11_idx][col] = 1; W[c21_idx][col] = 1
        col += 1
        # M5 = (a+b)h
        U[a_idx[0]][col] = 1; U[a_idx[1]][col] = 1
        V[e_idx[3]][col] = 1
        W[c11_idx][col] = -1; W[c12_idx][col] = 1
        col += 1
        # M6 = (c-a)(e+f)
        U[a_idx[2]][col] = 1; U[a_idx[0]][col] = -1
        V[e_idx[0]][col] = 1; V[e_idx[1]][col] = 1
        W[c22_idx][col] = 1
        col += 1
        # M7 = (b-d)(g+h)
        U[a_idx[1]][col] = 1; U[a_idx[3]][col] = -1
        V[e_idx[2]][col] = 1; V[e_idx[3]][col] = 1
        W[c11_idx][col] = 1
        col += 1
        return col
    
    col = 0
    col = add_strassen_block(0, 0, 0, col)
    col = add_strassen_block(0, 0, 2, col)
    col = add_strassen_block(0, 0, 4, col)
    col = add_strassen_block(0, 2, 0, col)
    col = add_strassen_block(0, 2, 2, col)
    col = add_strassen_block(0, 2, 4, col)
    
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(U42, V42, W42, n, m, p, deadline):
    """Try to find a lower rank decomposition by optimization."""
    n, m, p = n, m, p
    nm = n * m  # 8
    mp = m * p  # 24
    pn = p * n  # 12
    
    # Build the target tensor
    # T[i,j,k] = 1 for all valid indices
    # We need: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    
    # Try rank 40, then 39
    for target_rank in [40, 39]:
        if time.time() > deadline - 20:
            break
        
        # Initialize: take first target_rank columns of the rank-42 solution
        # and try to optimize
        U0 = U42[:, :target_rank].copy()
        V0 = V42[:, :target_rank].copy()
        W0 = W42[:, :target_rank].copy()
        
        # Add small random perturbation
        np.random.seed(42)
        U0 += 0.01 * np.random.randn(nm, target_rank)
        V0 += 0.01 * np.random.randn(mp, target_rank)
        W0 += 0.01 * np.random.randn(pn, target_rank)
        
        # Optimize using gradient descent
        U, V, W = _optimize_decomposition(U0, V0, W0, n, m, p, deadline)
        
        if U is not None:
            # Verify
            if _verify_decomposition(U, V, W, n, m, p):
                return U, V, W, target_rank
    
    return None

def _optimize_decomposition(U, V, W, n, m, p, deadline):
    """Optimize the decomposition to minimize reconstruction error."""
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    
    # Precompute index mappings
    # For each (i,j,k) triple, compute the contribution
    # T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    
    # Build sparse structure: for each t, which (i,j) pairs are nonzero in U,
    # which (j,k) pairs are nonzero in V, which (k,i) pairs are nonzero in W
    
    def compute_error(U, V, W):
        """Compute the sum of squared errors."""
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for t in range(rank):
                        val += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                    error += (val - 1.0) ** 2
        return error
    
    # Use simple gradient descent with line search
    lr = 0.01
    U_curr, V_curr, W_curr = U.copy(), V.copy(), W.copy()
    curr_err = compute_error(U_curr, V_curr, W_curr)
    
    max_iter = 500
    for it in range(max_iter):
        if time.time() > deadline - 10:
            break
        if curr_err < 1e-10:
            break
        
        # Compute gradients
        # dE/dU[i*m+j, t] = 2 * sum over (i,j,k) where this U is used:
        #   (val(i,j,k) - 1) * V[j*p+k, t] * W[k*n+i, t]
        
        dU = np.zeros_like(U_curr)
        dV = np.zeros_like(V_curr)
        dW = np.zeros_like(W_curr)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for t in range(rank):
                        val += U_curr[i * m + j, t] * V_curr[j * p + k, t] * W_curr[k * n + i, t]
                    err = val - 1.0
                    for t in range(rank):
                        dU[i * m + j, t] += 2 * err * V_curr[j * p + k, t] * W_curr[k * n + i, t]
                        dV[j * p + k, t] += 2 * err * U_curr[i * m + j, t] * W_curr[k * n + i, t]
                        dW[k * n + i, t] += 2 * err * U_curr[i * m + j, t] * V_curr[j * p + k, t]
        
        # Gradient step
        U_curr -= lr * dU
        V_curr -= lr * dV
        W_curr -= lr * dW
        
        new_err = compute_error(U_curr, V_curr, W_curr)
        if new_err < curr_err:
            curr_err = new_err
        else:
            lr *= 0.5
            U_curr += lr * dU
            V_curr += lr * dV
            W_curr += lr * dW
    
    if curr_err < 1e-8:
        return U_curr, V_curr, W_curr
    return None

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    nm, mp, pn = n * m, m * p, p * n
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for t in range(U.shape[1]):
                    val += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                if abs(val - 1.0) > 1e-6:
                    return False
    return True
