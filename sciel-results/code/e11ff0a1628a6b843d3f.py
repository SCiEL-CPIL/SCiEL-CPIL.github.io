"""Find rank-minimal tensor decomposition for 2x3 * 3x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x4 matrix.
    
    The output tensor W has indices W[k*n + i, r] where:
    - i is the row index of A (0..n-1)
    - k is the column index of B (0..p-1) 
    - r is the rank index
    """
    n, m, p = 2, 3, 4
    start_time = time.time()
    deadline = start_time + 115.0 # 5s safety margin
    
    # Precompute indices for the tensor equation
    # T[i, j, k] = 1
    # Indices:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # We want to solve for U, V, W
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    def check_decomposition(U, V, W):
        """Check if U, V, W form a valid exact decomposition."""
        rank = U.shape[1]
        # U: (n*m, rank)
        # V: (m*p, rank)
        # W: (p*n, rank)
        
        # Efficient check using einsum or loops
        # Since dimensions are small, loops are fine, but let's try to be slightly faster.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    # Sum over rank
                    # Vectorized dot product
                    total = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    def try_rank(r, max_attempts=200):
        """
        Try to find a rank-r decomposition using local optimization and rounding.
        """
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Initialize with small random values
            # Scale: The product U*V*W should be ~1.
            # If we use values around 1, the sum of r terms will be ~r.
            # So we should scale down by 1/sqrt(r) or similar?
            # Or just use small values and let GD adjust.
            
            # Let's try initializing with integers/half-integers in a small range
            # and then doing a polish.
            
            # Strategy:
            # 1. Random init in [-1, 1]
            # 2. Gradient descent for a few steps
            # 3. Round to nearest 0.5
            # 4. Check
            
            U = np.random.uniform(-1, 1, size=(n*m, r))
            V = np.random.uniform(-1, 1, size=(m*p, r))
            W = np.random.uniform(-1, 1, size=(p*n, r))
            
            # Gradient Descent
            # Objective: E = sum_{i,j,k} (1 - sum_r U_{u,r} V_{v,r} W_{w,r})^2
            # dE/dU_{u,r} = -2 * sum_{i,j,k: u=i*m+j} (1 - S_{i,j,k}) * V_{v,r} * W_{w,r}
            # where S_{i,j,k} = sum_r U_{u,r} V_{v,r} W_{w,r}
            
            lr = 0.01
            for step in range(50):
                if time.time() > deadline:
                    return None
                
                # Compute S for all i,j,k
                # S[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                S = np.zeros((n, m, p))
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            
                            # S = dot(U[u], V[v] * W[w])
                            s_val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                            S[i, j, k] = s_val
                            
                            error = 1.0 - s_val
                            
                            # Gradients
                            # dE/dU[u,r] += -2 * error * V[v,r] * W[w,r]
                            grad_U[u_idx, :] += -2.0 * error * (V[v_idx, :] * W[w_idx, :])
                            grad_V[v_idx, :] += -2.0 * error * (U[u_idx, :] * W[w_idx, :])
                            grad_W[w_idx, :] += -2.0 * error * (U[u_idx, :] * V[v_idx, :])
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Optional: check if we are getting close to integers/half-integers
                # and if the error is small, try rounding early?
                # No, let's finish the steps.
            
            # Round to nearest half-integer
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Check
            if check_decomposition(U_rounded, V_rounded, W_rounded):
                # Convert to int representation (multiply by 2) to ensure exactness in check?
                # The check uses floats. 0.5 is exact in binary.
                return U_rounded, V_rounded, W_rounded
                
            # Also try pure random integer/half-integer search as a fallback for this attempt
            # This helps if GD gets stuck in a bad local minimum
            U_int = np.random.randint(-4, 5, size=(n*m, r))
            V_int = np.random.randint(-4, 5, size=(m*p, r))
            W_int = np.random.randint(-4, 5, size=(p*n, r))
            
            U_rand = U_int / 2.0
            V_rand = V_int / 2.0
            W_rand = W_int / 2.0
            
            if check_decomposition(U_rand, V_rand, W_rand):
                return U_rand, V_rand, W_rand
                
        return None

    # Construct schoolbook rank 24 as fallback
    rank_schoolbook = n * m * p
    U_school = np.zeros((n*m, rank_schoolbook), dtype=float)
    V_school = np.zeros((m*p, rank_schoolbook), dtype=float)
    W_school = np.zeros((p*n, rank_schoolbook), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1.0
                V_school[j*p + k, col] = 1.0
                W_school[k*n + i, col] = 1.0
                col += 1
                
    best_result = (U_school, V_school, W_school)
    best_rank = rank_schoolbook
    
    # Try to find lower ranks
    # We prioritize lower ranks.
    # If we find rank 20, we stop.
    # If we find 21, we try 20 for remaining time.
    
    ranks_to_try = [20, 21, 22, 23]
    
    for r in ranks_to_try:
        if time.time() > deadline - 5.0:
            break
            
        # Allocate more attempts to lower ranks?
        # Or just a fixed number.
        # Let's say 100 attempts per rank.
        attempts = 100
        if r == 20:
            attempts = 150 # Give more chance to the best
        
        result = try_rank(r, attempts)
        if result is not None:
            U, V, W = result
            best_result = (U, V, W)
            best_rank = r
            # If we found the minimum we care about (20), we can break
            # But maybe we can find 19? Unlikely for 2x3x4.
            # Lower bound for 2x2x2 is 7. For 2x3x4, it's likely > 20.
            # Let's just break if we find 20.
            if r == 20:
                break
        
    U, V, W = best_result
    
    # Convert to lists of lists
    # Ensure values are exact floats (0.5, 1.0, etc.)
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
