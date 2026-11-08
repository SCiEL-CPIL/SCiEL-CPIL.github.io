import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Target ranks to try, from low to high
    # We know schoolbook is 36.
    # Try 30 to 35.
    target_ranks = [30, 31, 32, 33, 34, 35]
    
    best_result = None
    best_rank = rank_schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    for R in target_ranks:
        if time.time() > deadline:
            break
            
        # Define variables
        # U: (n*m, R)
        # V: (m*p, R)
        # W: (p*n, R)
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        total_size = size_U + size_V + size_W
        
        # Residual function
        def residuals(x):
            U = x[:size_U].reshape(n * m, R)
            V = x[size_U:size_U + size_V].reshape(m * p, R)
            W = x[size_U + size_V:].reshape(p * n, R)
            
            # Compute sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            # We need to check for all i, j, k
            errors = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Sum over r
                        # U[i*m + j, :] is a vector of length R
                        # V[j*p + k, :] is a vector of length R
                        # W[k*n + i, :] is a vector of length R
                        # Dot product of (U * V) with W?
                        # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                        # This is not a simple dot product of two vectors.
                        # It is the sum of element-wise product of three vectors.
                        # np.sum(U_vec * V_vec * W_vec)
                        U_vec = U[i * m + j, :]
                        V_vec = V[j * p + k, :]
                        W_vec = W[k * n + i, :]
                        val = np.sum(U_vec * V_vec * W_vec)
                        errors.append(val - 1.0)
            return np.array(errors)
        
        # Initial guess
        # Use schoolbook decomposition if R <= 36?
        # If R < 36, we can't use full schoolbook.
        # Let's try random initialization with small values.
        # Or, use a structured initialization:
        # Set first R columns of schoolbook? No, schoolbook is 36.
        # If R < 36, random is better to avoid local minima of "missing terms".
        # If R >= 36, we can use schoolbook directly (but we are searching for < 36).
        
        # Let's try a few random restarts for each R
        found = False
        num_restarts = 5 if R > 32 else 10
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization
            np.random.seed(42 + R + restart) # Deterministic for reproducibility in testing, but random for actual search?
            # Better to use random seed based on time for actual exploration?
            # But for stability, let's use a mix.
            # Let's use random normal distribution scaled by 1.0
            x0 = np.random.normal(0, 1, total_size)
            
            # Try optimization
            try:
                res = opt.least_squares(residuals, x0, method='trf', max_nfev=1000)
            except Exception as e:
                continue
                
            if res.success or res.cost < 1e-9:
                # Check if error is small enough
                if res.cost < 1e-9:
                    # Extract U, V, W
                    U_opt = res.x[:size_U].reshape(n * m, R)
                    V_opt = res.x[size_U:size_U + size_V].reshape(m * p, R)
                    W_opt = res.x[size_U + size_V:].reshape(p * n, R)
                    
                    # Snap to half-integers
                    # Check if snapping preserves exactness
                    U_snap = np.round(U_opt * 2) / 2
                    V_snap = np.round(V_opt * 2) / 2
                    W_snap = np.round(W_opt * 2) / 2
                    
                    # Verify snapped solution
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                U_vec = U_snap[i * m + j, :]
                                V_vec = V_snap[j * p + k, :]
                                W_vec = W_snap[k * n + i, :]
                                val = np.sum(U_vec * V_vec * W_vec)
                                if abs(val - 1.0) > 1e-9:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        # Found a valid solution with rank R
                        # Convert to lists of floats/ints
                        U_list = U_snap.tolist()
                        V_list = V_snap.tolist()
                        W_list = W_snap.tolist()
                        return {"U": U_list, "V": V_list, "W": W_list}
                        
        # If no valid snapped solution found, maybe the float solution is good enough?
        # Task says "integer/half-integer entries". So we must snap.
        
    # Fallback: Schoolbook
    rank = rank_schoolbook
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
