import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target rank search
    # Lower bound is 25.
    # Let's try ranks from 25 to 30.
    
    # Precompute target tensor
    # T[i, j, k, i_p, k_p] = 1 if i==i_p and k==k_p else 0
    # Shape: (n, m, p, n, p)
    T_target = np.zeros((n, m, p, n, p))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                T_target[i, j, k, i, k] = 1.0
                
    # Flatten target for least_squares
    # We need to map (i, j, k, i_p, k_p) to a linear index
    # But least_squares expects a function that returns a vector of residuals.
    # We can just return the difference M - T_target flattened.
    
    def residuals(params):
        # params: flattened U, V, W
        # U: (n*m, r)
        # V: (m*p, r)
        # W: (p*n, r)
        
        size_U = n * m
        size_V = m * p
        size_W = p * n
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U+size_V]
        W_flat = params[size_U+size_V:]
        
        # Reshape
        U = U_flat.reshape((n, m))
        V = V_flat.reshape((m, p))
        W = W_flat.reshape((p, n))
        
        # Compute M[i, j, k, i_p, k_p] = sum_c U[i,j,c] V[j,k,c] W[k_p, i_p, c]
        # Wait, W is (p, n). Index is (k_p, i_p).
        # My previous derivation: W[k*n+i]. So row k, col i.
        # So W[k_p, i_p] is correct.
        
        # We need to sum over c. But params is for a single rank-1 term?
        # No, params contains all r columns.
        # So U is (n, m, r)?
        # Let's define params as containing r sets of (U_c, V_c, W_c).
        
        r = len(params) // (size_U + size_V + size_W)
        
        M = np.zeros((n, m, p, n, p))
        
        for c in range(r):
            U_c = params[c*(size_U+size_V+size_W) : c*(size_U+size_V+size_W)+size_U].reshape((n, m))
            V_c = params[c*(size_U+size_V+size_W)+size_U : (c+1)*(size_U+size_V+size_W)].reshape((m, p))
            W_c = params[(c+1)*(size_U+size_V+size_W)-size_W : (c+1)*(size_U+size_V+size_W)].reshape((p, n))
            
            # Add contribution
            # M[i, j, k, i_p, k_p] += U_c[i, j] * V_c[j, k] * W_c[k_p, i_p]
            
            # Efficient computation:
            # For each i, j, k:
            # val = U_c[i, j] * V_c[j, k]
            # M[i, j, k, :, :] += val * W_c
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = U_c[i, j] * V_c[j, k]
                        if val != 0:
                            M[i, j, k, :, :] += val * W_c
        
        return (M - T_target).flatten()

    # Search for minimal rank
    best_solution = None
    best_rank = n * m * p # 30
    
    start_time = time.time()
    deadline = start_time + 110 # Leave 10s margin
    
    for r in range(25, best_rank + 1):
        if time.time() > deadline:
            break
            
        size_U = n * m
        size_V = m * p
        size_W = p * n
        total_size = r * (size_U + size_V + size_W)
        
        found = False
        # Try a few random restarts
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            # Random initialization
            np.random.seed(42 + r + attempt)
            params = np.random.randn(total_size) * 0.1
            
            # Optimize
            try:
                result = least_squares(residuals, params, method='lm', max_nfev=1000)
            except Exception:
                continue
                
            if result.cost < 1e-10:
                # Check if we can round to integers/half-integers
                # Reconstruct U, V, W
                U_mat = np.zeros((size_U, r))
                V_mat = np.zeros((size_V, r))
                W_mat = np.zeros((size_W, r))
                
                for c in range(r):
                    offset = c * (size_U + size_V + size_W)
                    U_flat = result.x[offset : offset+size_U]
                    V_flat = result.x[offset+size_U : offset+size_U+size_V]
                    W_flat = result.x[offset+size_U+size_V : offset+size_U+size_V+size_W]
                    
                    # Round to nearest integer or half-integer
                    # Multiply by 2, round to nearest integer, divide by 2
                    U_rounded = np.round(U_flat * 2) / 2
                    V_rounded = np.round(V_flat * 2) / 2
                    W_rounded = np.round(W_flat * 2) / 2
                    
                    # Check if rounded version is exact
                    # Reconstruct M with rounded values
                    M_rounded = np.zeros((n, m, p, n, p))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = U_rounded[i*m+j] * V_rounded[j*p+k]
                                if val != 0:
                                    # W index is k*n+i? No, W is (p, n) in my residual func?
                                    # In residual func: W_c is (p, n).
                                    # W_c[k_p, i_p].
                                    # So W_flat index for (k, i) is k*n + i?
                                    # W_flat is flattened from (p, n).
                                    # So index is k*n + i.
                                    M_rounded[i, j, k, :, :] += val * W_rounded.reshape((p, n))
                    
                    if np.allclose(M_rounded, T_target, atol=1e-9):
                        # Success
                        # Format output as lists
                        U_list = U_rounded.reshape((size_U, r)).tolist()
                        V_list = V_rounded.reshape((size_V, r)).tolist()
                        W_list = W_rounded.reshape((size_W, r)).tolist()
                        
                        # Ensure integers are ints
                        def clean_matrix(mat):
                            cleaned = []
                            for row in mat:
                                new_row = []
                                for val in row:
                                    if abs(val - round(val)) < 1e-9:
                                        new_row.append(int(round(val)))
                                    elif abs(val*2 - round(val*2)) < 1e-9:
                                        new_row.append(val) # keep as float if half-integer
                                    else:
                                        new_row.append(val)
                                cleaned.append(new_row)
                            return cleaned

                        return {"U": clean_matrix(U_list), "V": clean_matrix(V_list), "W": clean_matrix(W_list)}
                        
        # If not found, continue to next rank
    # If no better solution found, return schoolbook
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
