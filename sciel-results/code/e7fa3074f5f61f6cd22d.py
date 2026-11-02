import numpy as np
import time

def construct():
    n, m, p = 2, 3, 4
    
    # Target Rank
    # Schoolbook is 24. Winograd suggests 19.
    # We will try to find a solution for rank 19.
    target_rank = 19
    
    # Shapes
    rows_U = n * m  # 6
    rows_V = m * p  # 12
    rows_W = p * n  # 8
    
    # Pre-compute Target Tensor
    # T_target[a, b, c, d, e, f] = 1 if a==e, b==c, d==f else 0
    # a: 0..n-1, b: 0..m-1, c: 0..m-1, d: 0..p-1, e: 0..n-1, f: 0..p-1
    T_target = np.zeros((n, m, m, p, n, p), dtype=np.float64)
    for a in range(n):
        for b in range(m):
            for d in range(p):
                # c must be b, f must be d, e must be a
                T_target[a, b, b, d, a, d] = 1.0
                
    def compute_error(U, V, W):
        # U: (R, n, m), V: (R, m, p), W: (R, n, p)
        # T_decomp[a, b, c, d, e, f] = sum_r U[r, a, b] * V[r, c, d] * W[r, e, f]
        # We can use einsum
        # 'rab,rcd,ref->abcdef'
        T_decomp = np.einsum('rab,rcd,ref->abcdef', U, V, W)
        diff = T_decomp - T_target
        return np.sum(diff * diff)
        
    def try_local_search(start_state, max_iters=20000):
        # start_state: flat array
        R = target_rank
        U = start_state[:rows_U*R].reshape(R, n, m).copy()
        V = start_state[rows_U*R:rows_U*R+rows_V*R].reshape(R, m, p).copy()
        W = start_state[rows_U*R+rows_V*R:].reshape(R, n, p).copy()
        
        current_err = compute_error(U, V, W)
        
        # Simple local search with random perturbations
        # Since the tensor is small, we can try to be smart.
        # But random perturbations on the flat vector might be slow.
        # Let's try updating one variable at a time.
        
        total_vars = 26 * R
        
        for _ in range(max_iters):
            var_idx = np.random.randint(0, total_vars)
            
            if var_idx < rows_U * R:
                idx = var_idx
                mat = U
                r, rem = divmod(idx, R)
                a, b = divmod(rem, m)
            elif var_idx < rows_U * R + rows_V * R:
                idx = var_idx - rows_U * R
                mat = V
                r, rem = divmod(idx, R)
                c, d = divmod(rem, p)
            else:
                idx = var_idx - rows_U * R - rows_V * R
                mat = W
                r, rem = divmod(idx, R)
                e, f = divmod(rem, p)
                
            old_val = mat[r, a, b] if var_idx < rows_U * R else (mat[r, c, d] if var_idx < rows_U * R + rows_V * R else mat[r, e, f])
            
            # Random perturbation
            # Scale perturbation based on error?
            perturbation = np.random.normal(0, 0.1)
            new_val = old_val + perturbation
            
            # Apply
            if var_idx < rows_U * R:
                U[r, a, b] = new_val
            elif var_idx < rows_U * R + rows_V * R:
                V[r, c, d] = new_val
            else:
                W[r, e, f] = new_val
                
            new_err = compute_error(U, V, W)
            
            if new_err < current_err:
                current_err = new_err
                if current_err < 1e-9:
                    return U, V, W, current_err
            else:
                # Revert
                if var_idx < rows_U * R:
                    U[r, a, b] = old_val
                elif var_idx < rows_U * R + rows_V * R:
                    V[r, c, d] = old_val
                else:
                    W[r, e, f] = old_val
                    
        return U, V, W, current_err

    best_result = None
    best_err = float('inf')
    
    start_time = time.time()
    deadline = start_time + 140
    
    # Multiple restarts
    while time.time() < deadline:
        # Random initialization
        # Small random values
        init_state = np.random.normal(0, 1.0, size=26 * target_rank)
        
        U, V, W, err = try_local_search(init_state, max_iters=5000)
        
        if err < best_err:
            best_err = err
            best_result = (U, V, W)
            
        if best_err < 1e-9:
            break
            
    if best_result is not None and best_err < 1e-9:
        U, V, W = best_result
        # Convert back to flat matrices as required by interface
        # U: (n*m, R) -> (R, n, m) flattened row-major?
        # Interface: U (n*m, R). 
        # My U is (R, n, m).
        # U_flat[i*m+j, c] should be U[c, i, j].
        U_out = np.zeros((n*m, R))
        V_out = np.zeros((m*p, R))
        W_out = np.zeros((p*n, R))
        
        for c in range(R):
            for i in range(n):
                for j in range(m):
                    U_out[i*m+j, c] = U[c, i, j]
            for j in range(m):
                for k in range(p):
                    V_out[j*p+k, c] = V[c, j, k]
            for k in range(p):
                for i in range(n):
                    W_out[k*n+i, c] = W[c, i, k]
                    
        return {
            "U": U_out.tolist(),
            "V": V_out.tolist(),
            "W": W_out.tolist()
        }
        
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
