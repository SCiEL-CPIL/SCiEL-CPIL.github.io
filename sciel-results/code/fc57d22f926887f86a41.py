import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    n, m, p = 2, 2, 4
    
    # Precompute index arrays for vectorized residual computation
    # U index: i*m+j for i in range(n), j in range(m)
    # V index: j*p+k for j in range(m), k in range(p)
    # W index: k*n+i for k in range(p), i in range(n)
    
    def get_indices():
        u_idx = []
        v_idx = []
        w_idx = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx.append(i*m+j)
                    v_idx.append(j*p+k)
                    w_idx.append(k*n+i)
        return np.array(u_idx), np.array(v_idx), np.array(w_idx)
    
    u_idx, v_idx, w_idx = get_indices()
    
    def residual_sq_vec(U, V, W):
        """Vectorized residual computation."""
        rank = U.shape[1]
        # For each (i,j,k) triple, compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Shape: (n*m*p, rank)
        U_sel = U[u_idx, :]  # (16, rank)
        V_sel = V[v_idx, :]  # (16, rank)
        W_sel = W[w_idx, :]  # (16, rank)
        prod = U_sel * V_sel * W_sel  # (16, rank)
        vals = prod.sum(axis=1)  # (16,)
        return float(np.sum((vals - 1.0)**2))
    
    def verify(U, V, W, tol=1e-9):
        rank = U.shape[1]
        U_sel = U[u_idx, :]
        V_sel = V[v_idx, :]
        W_sel = W[w_idx, :]
        prod = U_sel * V_sel * W_sel
        vals = prod.sum(axis=1)
        return bool(np.all(np.abs(vals - 1.0) <= tol))
    
    def get_rank14_solution():
        """Return the known rank-14 decomposition."""
        rank = 14
        U = np.zeros((4, rank))
        V = np.zeros((8, rank))
        W = np.zeros((8, rank))
        
        def set_col(col, u_dict, v_dict, w_dict):
            for idx, val in u_dict.items():
                U[idx, col] = val
            for idx, val in v_dict.items():
                V[idx, col] = val
            for idx, val in w_dict.items():
                W[idx, col] = val
            
        # Strassen-based rank 14
        # Left block (k=0,1)
        set_col(0, {0:1, 3:1}, {0:1, 5:1}, {0:1, 3:1})
        set_col(1, {2:1, 3:1}, {0:1}, {1:1, 3:-1})
        set_col(2, {0:1}, {1:1, 5:-1}, {2:1, 3:1})
        set_col(3, {3:1}, {4:1, 0:-1}, {0:1, 1:1})
        set_col(4, {0:1, 1:1}, {5:1}, {0:-1, 2:1})
        set_col(5, {2:1, 0:-1}, {0:1, 1:1}, {3:1})
        set_col(6, {1:1, 3:-1}, {4:1, 5:1}, {0:1})
        
        # Right block (k=2,3)
        set_col(7, {0:1, 3:1}, {2:1, 7:1}, {4:1, 7:1})
        set_col(8, {2:1, 3:1}, {2:1}, {5:1, 7:-1})
        set_col(9, {0:1}, {3:1, 7:-1}, {6:1, 7:1})
        set_col(10, {3:1}, {6:1, 2:-1}, {4:1, 5:1})
        set_col(11, {0:1, 1:1}, {7:1}, {4:-1, 6:1})
        set_col(12, {2:1, 0:-1}, {2:1, 3:1}, {7:1})
        set_col(13, {1:1, 3:-1}, {6:1, 7:1}, {4:1})
        
        return U, V, W
    
    def local_search(U, V, W, max_steps=200):
        """Perform local search with integer/half-integer entries."""
        rank = U.shape[1]
        current_val = residual_sq_vec(U, V, W)
        
        if current_val < 1e-9:
            return U, V, W, current_val
        
        steps = 0
        no_improve = 0
        while steps < max_steps and time.time() < deadline:
            steps += 1
            # Pick a random matrix and entry
            mat_idx = np.random.randint(0, 3)
            if mat_idx == 0:
                row = np.random.randint(0, 4)
                col = np.random.randint(0, rank)
                old_val = U[row, col]
                # Try different values: flip sign, set to 0, set to +/-1, +/-0.5
                candidates = []
                if old_val != 0:
                    candidates.append(-old_val)
                candidates.extend([0, 1, -1, 0.5, -0.5])
                # Remove duplicates and old value
                candidates = [c for c in candidates if c != old_val]
                if not candidates:
                    candidates = [1.0]
                new_val = candidates[np.random.randint(0, len(candidates))]
                U[row, col] = new_val
                new_val_res = residual_sq_vec(U, V, W)
                if new_val_res < current_val - 1e-12:
                    current_val = new_val_res
                    no_improve = 0
                else:
                    U[row, col] = old_val
                    no_improve += 1
            elif mat_idx == 1:
                row = np.random.randint(0, 8)
                col = np.random.randint(0, rank)
                old_val = V[row, col]
                candidates = []
                if old_val != 0:
                    candidates.append(-old_val)
                candidates.extend([0, 1, -1, 0.5, -0.5])
                candidates = [c for c in candidates if c != old_val]
                if not candidates:
                    candidates = [1.0]
                new_val = candidates[np.random.randint(0, len(candidates))]
                V[row, col] = new_val
                new_val_res = residual_sq_vec(U, V, W)
                if new_val_res < current_val - 1e-12:
                    current_val = new_val_res
                    no_improve = 0
                else:
                    V[row, col] = old_val
                    no_improve += 1
            else:
                row = np.random.randint(0, 8)
                col = np.random.randint(0, rank)
                old_val = W[row, col]
                candidates = []
                if old_val != 0:
                    candidates.append(-old_val)
                candidates.extend([0, 1, -1, 0.5, -0.5])
                candidates = [c for c in candidates if c != old_val]
                if not candidates:
                    candidates = [1.0]
                new_val = candidates[np.random.randint(0, len(candidates))]
                W[row, col] = new_val
                new_val_res = residual_sq_vec(U, V, W)
                if new_val_res < current_val - 1e-12:
                    current_val = new_val_res
                    no_improve = 0
                else:
                    W[row, col] = old_val
                    no_improve += 1
            
            if no_improve > 50:
                break
        
        return U, V, W, current_val
    
    def try_rank13_from_rank14():
        """Start from rank-14 solution, try to merge two columns into one for rank 13."""
        rank13 = 13
        best_val = float('inf')
        best_result = None
        
        U14, V14, W14 = get_rank14_solution()
        
        # Strategy: try to express one column as a combination of others
        # or start from rank-14 with one column zeroed and adjust
        
        for attempt in range(300):
            if time.time() > deadline:
                break
            
            # Start with rank-14, pick a random column to "remove"
            # and try to distribute its contributions
            U = U14[:, :rank13].copy()
            V = V14[:, :rank13].copy()
            W = W14[:, :rank13].copy()
            
            # Add small random perturbations to help escape local minima
            if np.random.random() < 0.3:
                U[np.random.randint(0, 4), np.random.randint(0, rank13)] += np.random.choice([-1, 1])
                V[np.random.randint(0, 8), np.random.randint(0, rank13)] += np.random.choice([-1, 1])
                W[np.random.randint(0, 8), np.random.randint(0, rank13)] += np.random.choice([-1, 1])
            
            # Also try: start with rank-14, zero out one column, then search
            if np.random.random() < 0.5:
                U = U14.copy()
                V = V14.copy()
                W = W14.copy()
                # Zero out a random column
                col_to_zero = np.random.randint(0, 14)
                U[:, col_to_zero] = 0
                V[:, col_to_zero] = 0
                W[:, col_to_zero] = 0
                # Remove that column and add a new random one
                remaining = [c for c in range(14) if c != col_to_zero]
                U = np.column_stack([U[:, remaining], np.random.randint(-1, 2, (4, 1))])
                V = np.column_stack([V[:, remaining], np.random.randint(-1, 2, (8, 1))])
                W = np.column_stack([W[:, remaining], np.random.randint(-1, 2, (8, 1))])
            
            U, V, W, val = local_search(U, V, W, max_steps=150)
            
            if val < best_val:
                best_val = val
                best_result = (U.copy(), V.copy(), W.copy())
                
            if best_val < 1e-9:
                if verify(best_result[0], best_result[1], best_result[2]):
                    return best_result
        
        return best_result
    
    def try_rank13_random():
        """Random restarts for rank 13."""
        rank13 = 13
        best_val = float('inf')
        best_result = None
        
        for attempt in range(500):
            if time.time() > deadline:
                break
            
            U = np.random.randint(-1, 2, (4, rank13))
            V = np.random.randint(-1, 2, (8, rank13))
            W = np.random.randint(-1, 2, (8, rank13))
            
            U, V, W, val = local_search(U, V, W, max_steps=100)
            
            if val < best_val:
                best_val = val
                best_result = (U.copy(), V.copy(), W.copy())
                
            if best_val < 1e-9:
                if verify(best_result[0], best_result[1], best_result[2]):
                    return best_result
        
        return best_result
    
    def try_rank12():
        """Try to find a rank-12 solution."""
        rank12 = 12
        best_val = float('inf')
        best_result = None
        
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            U = np.random.randint(-1, 2, (4, rank12))
            V = np.random.randint(-1, 2, (8, rank12))
            W = np.random.randint(-1, 2, (8, rank12))
            
            U, V, W, val = local_search(U, V, W, max_steps=80)
            
            if val < best_val:
                best_val = val
                best_result = (U.copy(), V.copy(), W.copy())
                
            if best_val < 1e-9:
                if verify(best_result[0], best_result[1], best_result[2]):
                    return best_result
        
        return best_result
    
    # Try rank 12 first (theoretical minimum)
    result = try_rank12()
    if result is not None:
        U, V, W = result
        if verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank 13 from rank-14 base
    result = try_rank13_from_rank14()
    if result is not None:
        U, V, W = result
        if verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank 13 random
    result = try_rank13_random()
    if result is not None:
        U, V, W = result
        if verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: rank 14 (known valid solution)
    U, V, W = get_rank14_solution()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
