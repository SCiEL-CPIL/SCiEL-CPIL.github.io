"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_fb = np.zeros((n * m, rank_school), dtype=float)
    V_fb = np.zeros((m * p, rank_school), dtype=float)
    W_fb = np.zeros((p * n, rank_school), dtype=float)
    for idx, (i, j, k) in enumerate(product(range(n), range(m), range(p))):
        U_fb[i * m + j, idx] = 1
        V_fb[j * p + k, idx] = 1
        W_fb[k * n + i, idx] = 1
    
    def to_list(mat):
        # Convert to list of lists with clean integer/half-integer values
        result = []
        for row in mat:
            new_row = []
            for v in row:
                # Round to nearest half-integer
                r = round(v * 2) / 2
                if abs(r - round(r)) < 1e-9:
                    new_row.append(int(round(r)))
                else:
                    new_row.append(r)
            result.append(new_row)
        return result
    
    def verify(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        # Reshape to (n, m, r), (m, p, r), (p, n, r)
        U_t = U.reshape(n, m, -1)
        V_t = V.reshape(m, p, -1)
        W_t = W.reshape(p, n, -1)
        
        # Compute T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        
        # Check if T is all ones
        if not np.allclose(T, 1.0, atol=tol):
            return False
        return True

    def snap_to_half_integers(U, V, W):
        """Snap values to nearest half-integer and return matrices."""
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        return U_s, V_s, W_s

    def compute_error(U, V, W):
        """Compute the squared error of the tensor reconstruction."""
        n, m, p = DIMENSIONS
        U_t = U.reshape(n, m, -1)
        V_t = V.reshape(m, p, -1)
        W_t = W.reshape(p, n, -1)
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        return np.sum((T - 1.0)**2)

    def search_rank(r, time_budget=5.0):
        """Try to find exact decomposition of rank r using Simulated Annealing on half-integers."""
        if time.time() > deadline - 5:
            return None
        
        n, m, p = DIMENSIONS
        # Total number of elements in U, V, W
        num_u = n * m * r
        num_v = m * p * r
        num_w = p * n * r
        total_params = num_u + num_v + num_w
        
        # Random initialization with half-integers in [-1, 1]
        # Values: -1, -0.5, 0, 0.5, 1
        init_vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        start_time = time.time()
        max_iterations = 50000 # Cap iterations
        
        # Simulated Annealing
        current_U = np.random.choice(init_vals, size=(n*m, r))
        current_V = np.random.choice(init_vals, size=(m*p, r))
        current_W = np.random.choice(init_vals, size=(p*n, r))
        
        current_err = compute_error(current_U, current_V, current_W)
        
        if current_err < 1e-6:
            return snap_to_half_integers(current_U, current_V, current_W)
            
        best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
        best_err = current_err
        
        T_initial = 1.0
        T_final = 0.01
        cooling_rate = 0.999
        
        T = T_initial
        
        for it in range(max_iterations):
            if time.time() - start_time > time_budget:
                break
            
            # Pick a random parameter to change
            param_idx = np.random.randint(0, total_params)
            
            if param_idx < num_u:
                idx_u = param_idx
                row = idx_u // r
                col = idx_u % r
                old_val = current_U[row, col]
                # Perturb by +/- 0.5
                delta = 0.5 if np.random.rand() > 0.5 else -0.5
                new_val = old_val + delta
                # Keep in range [-1, 1] for now, or allow larger?
                # Let's allow [-2, 2]
                if new_val < -2.0 or new_val > 2.0:
                    continue
                current_U[row, col] = new_val
                new_err = compute_error(current_U, current_V, current_W)
                
            elif param_idx < num_u + num_v:
                idx_v = param_idx - num_u
                row = idx_v // r
                col = idx_v % r
                old_val = current_V[row, col]
                delta = 0.5 if np.random.rand() > 0.5 else -0.5
                new_val = old_val + delta
                if new_val < -2.0 or new_val > 2.0:
                    continue
                current_V[row, col] = new_val
                new_err = compute_error(current_U, current_V, current_W)
                
            else:
                idx_w = param_idx - num_u - num_v
                row = idx_w // r
                col = idx_w % r
                old_val = current_W[row, col]
                delta = 0.5 if np.random.rand() > 0.5 else -0.5
                new_val = old_val + delta
                if new_val < -2.0 or new_val > 2.0:
                    continue
                current_W[row, col] = new_val
                new_err = compute_error(current_U, current_V, current_W)
            
            # Acceptance criterion
            if new_err < current_err:
                current_err = new_err
                if new_err < best_err:
                    best_err = new_err
                    best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
                    if best_err < 1e-9:
                        return snap_to_half_integers(best_U, best_V, best_W)
            else:
                # Metropolis
                if np.random.rand() < np.exp(-(new_err - current_err) / T):
                    current_err = new_err
            
            T *= cooling_rate
            if T < T_final:
                T = T_final
        
        # Final check on best found
        if best_err < 1e-6:
            return snap_to_half_integers(best_U, best_V, best_W)
            
        return None

    # Try ranks from known lower bounds upward
    # Lower bound: max(n*m, m*p, p*n) = 28
    # We try a range. If we find one, we return it.
    
    # Heuristic: Try ranks that are multiples of small numbers or near known bounds
    # Let's try from 50 to 83
    # To save time, we can try a few random ranks first?
    # No, ascending is safer to ensure minimal rank if found.
    
    # However, searching 50 might take time.
    # Let's allocate time per rank.
    
    # If we don't find anything < 84, return schoolbook.
    
    # Optimization: Check if rank 83, 82, etc. are likely?
    # Usually, reductions come in chunks.
    
    # Let's try a few specific ranks first that are "strassen-like" reductions.
    # But for 3x4x7, it's not clear.
    
    # We will iterate from 40 to 83.
    # Time budget per rank: (deadline - now) / (83 - current_rank)
    
    current_time = time.time()
    for r in range(40, 84):
        if time.time() > deadline - 10:
            break
        
        # Dynamic time budget
        remaining_ranks = 84 - r
        time_per_rank = (deadline - time.time() - 10) / max(1, remaining_ranks)
        time_per_rank = min(time_per_rank, 5.0) # Cap at 5s per rank to avoid getting stuck
        
        result = search_rank(r, time_budget=time_per_rank)
        if result is not None:
            U, V, W = result
            # Double verify
            if verify(U, V, W):
                return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    
    # Fallback to schoolbook
    return {"U": to_list(U_fb), "V": to_list(V_fb), "W": to_list(W_fb)}
# EVOLVE-BLOCK-END
