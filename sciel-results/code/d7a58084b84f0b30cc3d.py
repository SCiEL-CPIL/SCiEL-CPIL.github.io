"""Bootstrap SEARCH for rank-minimal exact tensor decomposition of 2x5 * 5x6."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def _verify_rank(U, V, W, n=2, m=5, p=6):
    """Check if given factor matrices produce correct matmul tensor."""
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # T[i*m+j, j*p+k, k*n+i] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check off-diagonal: T[a, b, c] = 0 unless indices align
    # a = i*m+j, b = j'*p+k, c = k'*n+i'
    # Non-zero only if j==j', k==k', i==i'
    for a in range(n*m):
        for b in range(m*p):
            i_a = a // m
            j_a = a % m
            j_b = b // p
            k_b = b % p
            if j_a != j_b:
                continue
            for c in range(p*n):
                k_c = c // n
                i_c = c % n
                if k_b != k_c or i_a != i_c:
                    val = np.sum(U_np[a] * V_np[b] * W_np[c])
                    if abs(val) > 1e-9:
                        return False
    return True

def _schoolbook_rank():
    """Return the standard schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_rank_optimization(R, n, m, p, max_iters=50, time_budget=10):
    """Try to find a rank-R decomposition using random restarts and optimization."""
    import random
    from scipy.optimize import minimize
    
    start_time = time.time()
    
    # Target tensor indices
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_indices.append((i*m+j, j*p+k, k*n+i))
    
    U_size = n * m * R
    V_size = m * p * R
    W_size = p * n * R
    total_vars = U_size + V_size + W_size
    
    def objective(x):
        U_flat = x[:U_size]
        V_flat = x[U_size:U_size+V_size]
        W_flat = x[U_size+V_size:]
        
        U_np = U_flat.reshape(n*m, R)
        V_np = V_flat.reshape(m*p, R)
        W_np = W_flat.reshape(p*n, R)
        
        # Compute error on non-zero entries
        err = 0.0
        for (a, b, c) in target_indices:
            val = np.dot(U_np[a], V_np[b] * W_np[c])
            err += (val - 1.0) ** 2
        
        # Compute error on zero entries (sample a subset for speed)
        # We check entries where the indices don't align
        zero_err = 0.0
        for a in range(n*m):
            i_a = a // m
            j_a = a % m
            for b in range(m*p):
                j_b = b // p
                k_b = b % p
                if j_a != j_b:
                    continue
                for c in range(p*n):
                    k_c = c // n
                    i_c = c % n
                    if k_b != k_c or i_a != i_c:
                        val = np.dot(U_np[a], V_np[b] * W_np[c])
                        zero_err += val ** 2
        
        return err + zero_err
    
    # Try multiple random restarts
    random.seed(42)
    for restart in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        # Random initialization
        x0 = np.random.randn(total_vars) * 0.1
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                             options={'maxiter': 100, 'ftol': 1e-10})
            
            if result.fun < 1e-8:
                # Found a solution, verify it
                U_flat = result.x[:U_size]
                V_flat = result.x[U_size:U_size+V_size]
                W_flat = result.x[U_size+V_size:]
                
                U_np = U_flat.reshape(n*m, R)
                V_np = V_flat.reshape(m*p, R)
                W_np = W_flat.reshape(p*n, R)
                
                U_list = U_np.tolist()
                V_list = V_np.tolist()
                W_list = W_np.tolist()
                
                if _verify_rank(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list}
        except Exception:
            continue
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    best = _schoolbook_rank()
    best_rank = n * m * p  # 60
    
    # Verify schoolbook
    if not _verify_rank(best["U"], best["V"], best["W"]):
        return best

    # Try to find a lower rank solution
    # Start from rank 59 and go up to 60
    for R in range(best_rank - 1, 0, -1):
        if time.time() > deadline - 20:  # Leave 20s buffer
            break
        
        result = _try_rank_optimization(R, n, m, p, max_iters=30, time_budget=15)
        if result is not None:
            best = result
            best_rank = R
            break
    
    return best

# EVOLVE-BLOCK-END
