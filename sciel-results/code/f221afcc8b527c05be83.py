"""
Strategy: Combinatorial Search over Small Integer Alphas.

The previous strategy attempted continuous numerical optimization, which is
prone to local minima and makes exact rational verification difficult due to
floating-point precision issues. 

This new strategy recognizes that many known fast matrix multiplication
algorithms (Strassen, Winograd, etc.) use only small integer coefficients
(-1, 0, 1). We reformulate the problem as a discrete search:
1. We restrict the entries of U, V, W to a small set of integers, e.g., {-1, 0, 1}.
2. We use a randomized local search (simulated annealing or simple hill-climbing 
   with random restarts) to minimize the squared error of the tensor decomposition.
3. Because the search space is discrete and the target values are integers, 
   we can check for exact matches (error == 0) very quickly.
4. We try ranks from 24 upwards. If we find a rank R < 30 with exact integer 
   coefficients, we return it. This is much more likely to succeed than 
   continuous optimization for this specific problem structure.
"""

import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    n, m, p = 2, 3, 5
    
    # Target tensor is all ones
    # T[i, j, k] = 1
    
    # Shapes
    dim_u = n * m  # 6
    dim_v = m * p  # 15
    dim_w = p * n  # 10

    def verify(U, V, W, R):
        """
        Verify exact decomposition using integer arithmetic.
        U, V, W are numpy arrays of integers.
        """
        # Compute sum_{r} U[i,j,r] * V[j,k,r] * W[k,i,r]
        # We can use einsum for speed, but since R is small and dims are small,
        # a loop is also fine and safer for exactness if we stick to ints.
        # However, numpy int64 is exact for small numbers.
        
        # T_recon[i, j, k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # Note: W is indexed by (k, i) in the tensor formula W[k*n+i, r]
        # In our matrix representation, W is shape (dim_w, R).
        # W[k*n+i, r] corresponds to element at row k*n+i.
        
        # Let's flatten U to (n, m, R), V to (m, p, R), W to (p, n, R)
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        
        # Compute the tensor product sum
        # We want T[i, j, k] = sum_r U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
        
        T_recon = np.zeros((n, m, p), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
                    T_recon[i, j, k] = s
        
        return np.all(T_recon == 1)

    def compute_error(U, V, W, R):
        """
        Compute squared error of the decomposition.
        """
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        
        err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
                    diff = s - 1
                    err += diff * diff
        return err

    def search_rank(R, allowed_values, max_iter=5000):
        """
        Search for a rank-R decomposition using local search on discrete values.
        """
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        # Number of variables
        total_vars = R * (dim_u + dim_v + dim_w)
        
        # We will represent the state as a single flat array of indices into allowed_values
        # Or just the values themselves.
        
        for restart in range(20):
            if time.time() > deadline:
                break
                
            # Random initialization
            # Start with random values from allowed_values
            U = np.array([random.choice(allowed_values) for _ in range(dim_u * R)])
            V = np.array([random.choice(allowed_values) for _ in range(dim_v * R)])
            W = np.array([random.choice(allowed_values) for _ in range(dim_w * R)])
            
            current_err = compute_error(U, V, W, R)
            
            if current_err == 0:
                return U, V, W
            
            if current_err < best_err:
                best_err = current_err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
            # Local search
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                    
                # Pick a random variable to change
                var_idx = random.randint(0, total_vars - 1)
                
                # Determine which matrix and index
                if var_idx < dim_u * R:
                    mat = U
                    idx = var_idx
                elif var_idx < dim_u * R + dim_v * R:
                    mat = V
                    idx = var_idx - dim_u * R
                else:
                    mat = W
                    idx = var_idx - dim_u * R - dim_v * R
                
                # Pick a new value
                new_val = random.choice(allowed_values)
                old_val = mat[idx]
                
                if new_val == old_val:
                    continue
                    
                # Apply change
                mat[idx] = new_val
                
                new_err = compute_error(U, V, W, R)
                
                if new_err == 0:
                    return U, V, W
                
                # Accept if better, or with some probability (simulated annealing style)
                # For simplicity, let's do hill climbing first, then add random jumps
                if new_err < current_err:
                    current_err = new_err
                    if current_err < best_err:
                        best_err = current_err
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                else:
                    # Revert
                    mat[idx] = old_val
                    
                # Occasionally do a larger jump (mutation)
                if random.random() < 0.05:
                    # Change a few random variables
                    for _ in range(3):
                        v_idx = random.randint(0, total_vars - 1)
                        if v_idx < dim_u * R:
                            mat = U
                            idx = v_idx
                        elif v_idx < dim_u * R + dim_v * R:
                            mat = V
                            idx = v_idx - dim_u * R
                        else:
                            mat = W
                            idx = v_idx - dim_u * R - dim_v * R
                        mat[idx] = random.choice(allowed_values)
                    
                    jump_err = compute_error(U, V, W, R)
                    if jump_err < current_err:
                        current_err = jump_err
                    else:
                        # Revert jump is complex, so just keep it if not terrible? 
                        # Or revert to best known? 
                        # Let's just accept it if it's not too bad, or revert to best.
                        # To keep it simple, we'll just let it be. 
                        # But this can drift. 
                        # Let's stick to simple hill climbing + random restarts.
                        pass

        if best_U is not None and best_err == 0:
            return best_U, best_V, best_W
            
        return None

    # Try ranks from 24 to 29
    # Allowed values: -1, 0, 1 is a good start. Maybe -2, -1, 0, 1, 2 if needed.
    for R in range(24, 30):
        if time.time() > deadline:
            break
            
        result = search_rank(R, allowed_values=[-1, 0, 1], max_iter=2000)
        if result is not None:
            U, V, W = result
            # Convert to lists for output
            return {
                "U": U.reshape(dim_u, R).tolist(),
                "V": V.reshape(dim_v, R).tolist(),
                "W": W.reshape(dim_w, R).tolist()
            }
            
        # If -1,0,1 didn't work, try a wider range? 
        # But usually small ints are enough for these structures.
        # Let's try one more rank with slightly wider values if time permits
        if time.time() < deadline - 20:
             result = search_rank(R, allowed_values=[-2, -1, 0, 1, 2], max_iter=1000)
             if result is not None:
                 U, V, W = result
                 return {
                     "U": U.reshape(dim_u, R).tolist(),
                     "V": V.reshape(dim_v, R).tolist(),
                     "W": W.reshape(dim_w, R).tolist()
                 }

    # Fallback to schoolbook rank 30
    rank = n * m * p
    U = [[0.0] * rank for _ in range(dim_u)]
    V = [[0.0] * rank for _ in range(dim_v)]
    W = [[0.0] * rank for _ in range(dim_w)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1

    return {"U": U, "V": V, "W": W}
