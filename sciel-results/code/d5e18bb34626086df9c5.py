import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    target_rank = 50  # Schoolbook rank
    best_solution = None
    best_rank = target_rank
    
    # Pre-compute target tensor
    # T[i, j, j_prime, k] = 1 if j == j_prime else 0
    # Shape: (n, m, m, p)
    target = np.zeros((n, m, m, p), dtype=np.float64)
    for j in range(m):
        for i in range(n):
            for k in range(p):
                target[i, j, j, k] = 1.0

    def compute_residual(params, R):
        # Unpack params into U, V, W
        # U: (n, m, R), V: (m, p, R), W: (p, n, R)
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U = params[:size_U].reshape(n, m, R)
        V = params[size_U:size_U + size_V].reshape(m, p, R)
        W = params[size_U + size_V:].reshape(p, n, R)
        
        # Compute T[i, j, j_prime, k] = sum_r U[i,j,r] * V[j_prime,k,r] * W[k,i,r]
        # einsum: U[i,j,r] * V[j_prime,k,r] * W[k,i,r] -> T[i, j, j_prime, k]
        # Indices: U: i j r, V: a k r (a is j_prime), W: k i r
        T = np.einsum('ijr,akr,kir->iajk', U, V, W)
        
        # Residual is T - target
        # Flatten for least_squares
        return (T - target).flatten()

    def check_solution(U, V, W, R):
        # Round to nearest 0.5
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Check if entries are effectively integer or half-integer
        # np.round(x*2)/2 ensures this.
        
        # Compute T exactly with rounded values
        T = np.einsum('ijr,akr,kir->iajk', U_r, V_r, W_r)
        
        # Check if T matches target
        if np.allclose(T, target, atol=1e-9):
            return True, U_r, V_r, W_r
        return False, None, None, None

    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    # Try ranks from 49 down to 30
    for R in range(49, 29, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize params
        # Random initialization
        np.random.seed(42 + R)
        params0 = np.random.randn(n*m*R + m*p*R + p*n*R) * 0.5
        
        # Try a few restarts for each R
        for restart in range(5):
            if time.time() - start_time > time_limit:
                break
                
            if restart == 0:
                p_init = params0
            else:
                p_init = np.random.randn(n*m*R + m*p*R + p*n*R) * 0.5
            
            try:
                res = opt.least_squares(compute_residual, p_init, args=(R,), method='lm', max_nfev=1000)
            except Exception:
                continue
                
            if res.success or res.cost < 1e-6:
                # Unpack
                size_U = n * m * R
                size_V = m * p * R
                U = res.x[:size_U].reshape(n, m, R)
                V = res.x[size_U:size_U + size_V].reshape(m, p, R)
                W = res.x[size_U + size_V:].reshape(p, n, R)
                
                valid, U_r, V_r, W_r = check_solution(U, V, W, R)
                if valid:
                    best_rank = R
                    best_solution = (U_r, V_r, W_r)
                    break # Found valid for this R, try lower R? Or return immediately?
                    # Since we want min rank, and we are going down, we can return immediately if we trust the search.
                    # But maybe a lower rank is possible?
                    # Given the time limit, finding one is good.
                    # Let's return the first valid one found (highest R that is valid? No, lowest R).
                    # We are iterating 49, 48... so the first valid is the lowest R found so far.
                    # We can break and return.
                    
    if best_solution is not None:
        U, V, W = best_solution
        # Convert to list of lists as required by interface
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
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
# EVOLVE-BLOCK-END
