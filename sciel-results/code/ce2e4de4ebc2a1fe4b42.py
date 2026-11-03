"""Matrix multiplication 3x3x7 rank-minimal decomposition search."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 7]

def _verify(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition is correct.
    U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # T[a, b, c] = sum_d U[a, d] V[b, d] W[c, d]
    T = np.einsum('ad,bd,cd->abc', U_arr, V_arr, W_arr)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > 1e-6:
                    return False
    
    valid = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid[a, b, c] = True
        
    invalid_mask = ~valid
    if np.any(np.abs(T[invalid_mask]) > 1e-6):
        return False
        
    return True

def _snap_to_half_integer(x):
    """Snap values to nearest half-integer."""
    return np.round(x * 2) / 2.0

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Precompute target tensor
    T_target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    valid_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                valid_indices.append((a, b, c))
    
    # Schoolbook baseline (rank 63)
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb), dtype=np.float64)
    V_sb = np.zeros((m*p, rank_sb), dtype=np.float64)
    W_sb = np.zeros((p*n, rank_sb), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    def calculate_error(U, V, W):
        """Calculate sum of squared errors between computed tensor and target."""
        T = np.einsum('ad,bd,cd->abc', U, V, W)
        err = T - T_target
        return np.sum(err * err)

    def simulated_annealing_search(target_rank, time_limit, init_U=None, init_V=None, init_W=None):
        """
        Perform Simulated Annealing on the half-integer lattice.
        """
        if init_U is not None:
            U = init_U.copy()
            V = init_V.copy()
            W = init_W.copy()
        else:
            # Random initialization in [-2, 2]
            U = np.round(np.random.uniform(-2.0, 2.0, (n*m, target_rank)) * 2) / 2.0
            V = np.round(np.random.uniform(-2.0, 2.0, (m*p, target_rank)) * 2) / 2.0
            W = np.round(np.random.uniform(-2.0, 2.0, (p*n, target_rank)) * 2) / 2.0
        
        current_err = calculate_error(U, V, W)
        if current_err < 1e-10:
            return U, V, W, True

        T = 1.0
        T_min = 0.01
        cooling_rate = 0.995
        no_improve_count = 0
        max_no_improve = 500
        
        start_sa = time.time()
        
        while time.time() - start_sa < time_limit and T > T_min:
            # Pick a random matrix
            mat_idx = random.randint(0, 2)
            if mat_idx == 0:
                mat = U
            elif mat_idx == 1:
                mat = V
            else:
                mat = W
            
            # Pick a random row and column
            row_idx = random.randint(0, mat.shape[0] - 1)
            col_idx = random.randint(0, mat.shape[1] - 1)
            
            # Pick a random perturbation
            delta = random.choice([-1.0, -0.5, 0.5, 1.0])
            
            old_val = mat[row_idx, col_idx]
            new_val = old_val + delta
            new_val_snapped = _snap_to_half_integer(np.array([new_val]))[0]
            
            if new_val_snapped == old_val:
                continue
                
            # Apply change
            mat[row_idx, col_idx] = new_val_snapped
            
            new_err = calculate_error(U, V, W)
            
            if new_err < current_err:
                current_err = new_err
                no_improve_count = 0
                if current_err < 1e-10:
                    return U, V, W, True
            else:
                # Accept with probability exp(-delta_err / T)
                delta_err = new_err - current_err
                if delta_err < 1e-12:
                    prob = 1.0
                else:
                    prob = np.exp(-delta_err / T)
                
                if random.random() < prob:
                    current_err = new_err
                else:
                    # Revert
                    mat[row_idx, col_idx] = old_val
                    no_improve_count += 1
                    if no_improve_count > max_no_improve:
                        break
            
            T *= cooling_rate
        
        return U, V, W, False

    # Search for lower ranks
    # Try ranks from 62 down to 21
    # We prioritize ranks close to 63 as they are more likely to exist and easier to find.
    
    for target_rank in range(62, 20, -1):
        if time.time() > deadline - 5.0:
            break
            
        # Allocate time based on rank
        # Ranks 62-60 get 15s, 59-50 get 10s, others 5s
        if target_rank >= 60:
            time_budget = 15.0
        elif target_rank >= 50:
            time_budget = 10.0
        else:
            time_budget = 5.0
            
        # Ensure we don't exceed deadline
        time_left = deadline - time.time()
        if time_budget > time_left - 5.0:
            time_budget = max(1.0, time_left - 5.0)
            
        # Special initialization for rank 62: Use first 62 columns of schoolbook
        if target_rank == 62:
            U_init = U_sb[:, :62]
            V_init = V_sb[:, :62]
            W_init = W_sb[:, :62]
        else:
            U_init = None
            V_init = None
            W_init = None
            
        found, U_found, V_found, W_found = simulated_annealing_search(
            target_rank, time_budget, U_init, V_init, W_init
        )
        
        if found:
            best_rank = target_rank
            best_U = U_found
            best_V = V_found
            best_W = W_found
            # Continue searching for even lower ranks
            # But if we are running low on time, we should stop
            if time.time() > deadline - 10.0:
                break

    # Convert to lists of floats
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
