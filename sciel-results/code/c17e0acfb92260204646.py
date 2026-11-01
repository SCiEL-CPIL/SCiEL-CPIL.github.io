import numpy as np
import time

DIMENSIONS = [2, 2, 12]

def construct():
    n, m, p = DIMENSIONS
    # Target rank to try
    target_rank = 22
    
    # Deadline guard
    start_time = time.time()
    max_time = 100  # Leave 20s margin
    
    # Initialize best result as schoolbook (rank 48)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    best_error = 0.0
    
    # Try to find a lower rank decomposition
    # We will try rank 22
    # Structure: Two blocks of 2x2x6
    # We will search for a rank-11 decomposition for 2x2x6
    # and replicate it for the two halves of the 12 columns.
    
    # Parameters for 2x2x6 block
    n_b, m_b, p_b = 2, 2, 6
    rank_b = 11
    
    # Initialize random block decomposition
    # U_b: 4 x 11, V_b: 12 x 11, W_b: 12 x 11
    # We use a heuristic initialization based on Strassen for 2x2x2 and extending
    # For simplicity, random initialization with scaling
    np.random.seed(42)
    
    # We will optimize the block decomposition
    # Flatten the parameters
    # We want to minimize sum (sum_r U[i,j,r] V[j,k,r] W[k,i,r] - 1)^2 for all i,j,k in 2x2x6
    
    def compute_error_block(U_b, V_b, W_b):
        # U_b: 4 x rank_b
        # V_b: 12 x rank_b
        # W_b: 12 x rank_b
        # Indices: i in 0..1, j in 0..1, k in 0..5
        # U index: i*2 + j
        # V index: j*6 + k
        # W index: k*2 + i
        errors = []
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    u_idx = i * m_b + j
                    v_idx = j * p_b + k
                    w_idx = k * n_b + i
                    val = np.dot(U_b[u_idx, :], np.multiply(V_b[v_idx, :], W_b[w_idx, :]))
                    errors.append(val - 1.0)
        return np.sum(np.array(errors) ** 2)
    
    # Optimization loop
    U_b = np.random.randn(n_b * m_b, rank_b) * 0.5
    V_b = np.random.randn(m_b * p_b, rank_b) * 0.5
    W_b = np.random.randn(p_b * n_b, rank_b) * 0.5
    
    # Use simple gradient descent or coordinate descent
    # Since scipy is available, we can use least_squares
    def residual_func(params):
        # params: flattened U_b, V_b, W_b
        size_U = n_b * m_b * rank_b
        size_V = m_b * p_b * rank_b
        size_W = p_b * n_b * rank_b
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:size_U + size_V + size_W]
        
        U_curr = U_flat.reshape(n_b * m_b, rank_b)
        V_curr = V_flat.reshape(m_b * p_b, rank_b)
        W_curr = W_flat.reshape(p_b * n_b, rank_b)
        
        res = []
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    u_idx = i * m_b + j
                    v_idx = j * p_b + k
                    w_idx = k * n_b + i
                    val = np.dot(U_curr[u_idx, :], np.multiply(V_curr[v_idx, :], W_curr[w_idx, :]))
                    res.append(val - 1.0)
        return np.array(res)
    
    initial_params = np.concatenate([U_b.flatten(), V_b.flatten(), W_b.flatten()])
    
    try:
        # Use scipy.optimize.least_squares
        from scipy.optimize import least_squares
        result = least_squares(residual_func, initial_params, method='lm', max_nfev=1000)
        
        if result.success or np.linalg.norm(result.fun) < 1e-6:
            # Extract solution
            U_sol = result.x[:n_b * m_b * rank_b].reshape(n_b * m_b, rank_b)
            V_sol = result.x[n_b * m_b * rank_b:n_b * m_b * rank_b + m_b * p_b * rank_b].reshape(m_b * p_b, rank_b)
            W_sol = result.x[n_b * m_b * rank_b + m_b * p_b * rank_b:].reshape(p_b * n_b, rank_b)
            
            err = np.sum(result.fun ** 2)
            if err < 1e-8:
                # Construct full 2x2x12 decomposition
                U_full = np.zeros((n * m, 2 * rank_b))
                V_full = np.zeros((m * p, 2 * rank_b))
                W_full = np.zeros((p * n, 2 * rank_b))
                
                # Block 1 (cols 0-5 of B)
                U_full[:, 0:rank_b] = U_sol
                V_full[0:m_b * p_b, 0:rank_b] = V_sol
                W_full[0:p_b * n_b, 0:rank_b] = W_sol
                
                # Block 2 (cols 6-11 of B)
                U_full[:, rank_b:2 * rank_b] = U_sol
                # V indices: j*12 + k. For block 2, k is 6..11.
                # V_sol has indices j*6 + k' where k' is 0..5.
                # So V_full[j*12 + 6 + k', rank_b:2*rank_b] = V_sol[j*6 + k', 0:rank_b]
                for j in range(m_b):
                    for k in range(p_b):
                        V_full[j * p + 6 + k, rank_b:2 * rank_b] = V_sol[j * p_b + k, 0:rank_b]
                
                # W indices: k*2 + i. For block 2, k is 6..11.
                # W_full[(6+k')*2 + i, rank_b:2*rank_b] = W_sol[k'*2 + i, 0:rank_b]
                for k in range(p_b):
                    for i in range(n_b):
                        W_full[(6 + k) * n_b + i, rank_b:2 * rank_b] = W_sol[k * n_b + i, 0:rank_b]
                
                # Verify full decomposition
                # Check error
                full_errors = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            val = np.dot(U_full[u_idx, :], np.multiply(V_full[v_idx, :], W_full[w_idx, :]))
                            full_errors.append(val - 1.0)
                            
                full_err = np.sum(np.array(full_errors) ** 2)
                if full_err < 1e-8:
                    best_U, best_V, best_W = U_full, V_full, W_full
                    best_rank = 2 * rank_b
                    best_error = full_err
                    
    except Exception as e:
        pass # Fallback to schoolbook
    
    # Convert to lists of lists as required by interface
    U_out = [[0] * best_rank for _ in range(n * m)]
    V_out = [[0] * best_rank for _ in range(m * p)]
    W_out = [[0] * best_rank for _ in range(p * n)]
    
    for i in range(n * m):
        for r in range(best_rank):
            U_out[i][r] = best_U[i][r]
    for j in range(m * p):
        for r in range(best_rank):
            V_out[j][r] = best_V[j][r]
    for k in range(p * n):
        for r in range(best_rank):
            W_out[k][r] = best_W[k][r]
            
    return {"U": U_out, "V": V_out, "W": W_out}
