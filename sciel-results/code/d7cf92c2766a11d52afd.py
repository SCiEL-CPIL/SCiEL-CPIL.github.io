import numpy as np
import time
# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 5
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s kill
    
    # Build target tensor T
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0
                
    # Shapes
    rows_U = n * m
    rows_V = m * p
    rows_W = p * n
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((rows_U, rank))
        V = np.zeros((rows_V, rank))
        W = np.zeros((rows_W, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Allowed values for discrete search
    allowed_vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    def check_exact(U, V, W):
        # Verify entries are in allowed set (with tolerance)
        for M in [U, V, W]:
            for val in M:
                dist = np.min(np.abs(val - allowed_vals))
                if dist > 1e-6:
                    return False, None, None, None
        
        # Snap to closest allowed value
        U_r = np.array([[np.min(allowed_vals[np.argmin(np.abs(val - allowed_vals))]) for val in row] for row in U])
        V_r = np.array([[np.min(allowed_vals[np.argmin(np.abs(val - allowed_vals))]) for val in row] for row in V])
        W_r = np.array([[np.min(allowed_vals[np.argmin(np.abs(val - allowed_vals))]) for val in row] for row in W])
        
        # Check tensor decomposition
        R = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
        if np.max(np.abs(R - T)) < 1e-9:
            return True, U_r, V_r, W_r
        return False, None, None, None

    def discrete_local_search(r, max_iters=10000):
        # Initialize with random allowed values
        U = np.array([[np.random.choice(allowed_vals) for _ in range(r)] for _ in range(rows_U)])
        V = np.array([[np.random.choice(allowed_vals) for _ in range(r)] for _ in range(rows_V)])
        W = np.array([[np.random.choice(allowed_vals) for _ in range(r)] for _ in range(rows_W)])
        
        # Compute initial residual and loss
        R = np.einsum('ic,jc,kc->ijk', U, V, W) - T
        loss = np.sum(R**2)
        
        if loss < 1e-9:
            return U, V, W
        
        no_improve_count = 0
        max_no_improve = 500
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Pick a random variable to update
            mat_idx = np.random.randint(3)
            if mat_idx == 0:
                mat = U
                rows = rows_U
            elif mat_idx == 1:
                mat = V
                rows = rows_V
            else:
                mat = W
                rows = rows_W
                
            row_idx = np.random.randint(rows)
            col_idx = np.random.randint(r)
            
            old_val = mat[row_idx, col_idx]
            
            best_val = old_val
            best_loss = loss
            best_R = R
            
            # Try other allowed values
            for new_val in allowed_vals:
                if np.isclose(new_val, old_val):
                    continue
                
                delta = new_val - old_val
                
                # Compute change in R
                # R[I, J, K] = sum_c U[I, c] V[J, c] W[K, c]
                # If U changes, R[row_idx, J, K] += delta * V[J, col_idx] * W[K, col_idx]
                
                if mat_idx == 0:
                    # Update R for row_idx
                    # V_col = V[:, col_idx]
                    # W_col = W[:, col_idx]
                    # Delta_R = delta * outer(V_col, W_col)
                    # R[row_idx] += Delta_R
                    v_col = V[:, col_idx]
                    w_col = W[:, col_idx]
                    delta_R = delta * np.outer(v_col, w_col)
                    
                    R_new = R.copy()
                    R_new[row_idx] += delta_R
                elif mat_idx == 1:
                    # Update R for col_idx? No, V[J, c] affects all I for this J, K?
                    # R[I, J, K] = sum_c U[I, c] V[J, c] W[K, c]
                    # If V[J0, c0] changes, R[I, J0, K] += delta * U[I, c0] * W[K, c0]
                    u_col = U[:, col_idx]
                    w_col = W[:, col_idx]
                    delta_R = delta * np.outer(u_col, w_col)
                    
                    R_new = R.copy()
                    # R[I, row_idx, K] += delta_R[I, K]
                    R_new[:, row_idx, :] += delta_R
                else:
                    # W[K, c] changes. R[I, J, K0] += delta * U[I, c0] * V[J, c0]
                    u_col = U[:, col_idx]
                    v_col = V[:, col_idx]
                    delta_R = delta * np.outer(u_col, v_col)
                    
                    R_new = R.copy()
                    # R[I, J, row_idx] += delta_R[I, J]
                    R_new[:, :, row_idx] += delta_R
                
                new_loss = np.sum(R_new**2)
                
                if new_loss < best_loss:
                    best_loss = new_loss
                    best_val = new_val
                    best_R = R_new
            
            if best_val != old_val:
                # Apply best change
                mat[row_idx, col_idx] = best_val
                R = best_R
                loss = best_loss
                no_improve_count = 0
                
                if loss < 1e-9:
                    return U, V, W
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
        
        return None

    # Search for lower rank
    # Start from 44 down to 35
    for r in range(44, 34, -1):
        if time.time() > deadline - 10:
            break
            
        # Number of restarts based on time
        remaining = deadline - time.time()
        num_restarts = max(2, int(remaining / 5))
        
        found = False
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(12345 + restart * 1000 + r)
            
            U, V, W = discrete_local_search(r, max_iters=5000)
            
            if U is not None:
                is_exact, U_r, V_r, W_r = check_exact(U, V, W)
                if is_exact:
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
        if found:
            break
            
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
