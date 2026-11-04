import numpy as np
import random
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build target tensor T_target (dense)
    # T_target[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T_target = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                
    max_rank = n * m * p
    
    def schoolbook():
        rank = max_rank
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def compute_error(U, V, W):
        # S[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        # Using einsum for speed
        S = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((S - T_target) ** 2)

    def try_find_solution(R, max_iters=5000, initial_temp=1.0, cooling=0.99):
        # Random initialization
        # We use values in {-1, 0, 1} initially
        U = np.random.randint(-1, 2, (nm, R)).astype(np.float64)
        V = np.random.randint(-1, 2, (mp, R)).astype(np.float64)
        W = np.random.randint(-1, 2, (pn, R)).astype(np.float64)
        
        current_error = compute_error(U, V, W)
        
        if current_error < 1e-9:
             return U, V, W

        temp = initial_temp
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_error = current_error
        
        for it in range(max_iters):
            # Pick a random variable
            mat_choice = np.random.randint(0, 3)
            if mat_choice == 0:
                mat = U
            elif mat_choice == 1:
                mat = V
            else:
                mat = W
                
            rows, cols = mat.shape
            r_idx = np.random.randint(0, rows)
            c_idx = np.random.randint(0, cols)
            
            old_val = mat[r_idx, c_idx]
            # Propose a change
            # Deltas: -1, -0.5, 0.5, 1
            delta = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            new_val = old_val + delta
            
            # Update matrix temporarily
            mat[r_idx, c_idx] = new_val
            
            # Compute new error
            # Re-assign to variables to ensure einsum uses updated data
            if mat_choice == 0:
                U_curr, V_curr, W_curr = U, V, W
            elif mat_choice == 1:
                U_curr, V_curr, W_curr = U, V, W
            else:
                U_curr, V_curr, W_curr = U, V, W
                
            # Note: U, V, W are references to the arrays being modified
            new_error = compute_error(U, V, W)
            
            delta_error = new_error - current_error
            
            if delta_error < 0:
                # Accept
                current_error = new_error
                if current_error < best_error:
                    best_error = current_error
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    if best_error < 1e-9:
                        return best_U, best_V, best_W
            else:
                # Metropolis criterion
                if np.random.rand() < np.exp(-delta_error / temp):
                    current_error = new_error
                    if current_error < best_error:
                        best_error = current_error
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                        if best_error < 1e-9:
                            return best_U, best_V, best_W
                else:
                    # Revert
                    mat[r_idx, c_idx] = old_val
            
            temp *= cooling
            
        return None

    # Try ranks from max_rank - 1 down to max_rank - 8
    # We limit the search to save time
    for R in range(max_rank - 1, max(10, max_rank - 15), -1):
        # Try a few restarts for each rank
        found = False
        for restart in range(5):
            res = try_find_solution(R, max_iters=2000, initial_temp=2.0, cooling=0.995)
            if res:
                U, V, W = res
                # Snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify
                err = compute_error(U_snap, V_snap, W_snap)
                if err < 1e-9:
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                    
    return schoolbook()
