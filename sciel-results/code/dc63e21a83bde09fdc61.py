"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def _verify_fast(U, V, W, n, m, p):
    """
    Fast verification using numpy einsum.
    U: (n*m, R), V: (m*p, R), W: (p*n, R)
    Returns max error.
    """
    R = U.shape[1]
    # Reshape to 3D tensors
    U_3d = U.reshape(n, m, R)
    V_3d = V.reshape(m, p, R)
    W_3d = W.reshape(p, n, R)
    
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Note: W index is k*n+i. In W_3d, index is [k, i, r].
    # So W_3d[k, i, r] corresponds to W[k*n+i, r].
    
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    # Target is all ones
    error = np.max(np.abs(T - 1.0))
    return error

def _is_half_integer(x):
    """Check if x is integer or half-integer."""
    return abs(x * 2 - round(x * 2)) < 1e-9

def _check_half_integer_matrices(U, V, W):
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if not _is_half_integer(val):
                    return False
    return True

def _schoolbook(n, m, p):
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
    return U, V, W

def _random_half_integer_matrix(rows, cols, rng):
    """Generate a matrix with entries from {-1, -0.5, 0, 0.5, 1}."""
    # Using discrete uniform distribution
    vals = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(rows, cols))
    return vals

def _local_search(n, m, p, rank, deadline, seed):
    """
    Perform randomized local search (Simulated Annealing) to find a rank-rank decomposition.
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Allowed values
    allowed_vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    # Initialize
    U = _random_half_integer_matrix(n * m, rank, np.random)
    V = _random_half_integer_matrix(m * p, rank, np.random)
    W = _random_half_integer_matrix(p * n, rank, np.random)
    
    current_error = _verify_fast(U, V, W, n, m, p)
    
    # Simulated Annealing parameters
    T_start = 10.0
    T_end = 0.01
    steps = 20000  # Adjust based on time
    # Estimate time per step
    # Each step involves 1-3 error calculations.
    # Let's just run until deadline or steps exhausted.
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_error = current_error
    
    # Pre-compute shapes for speed
    shapes = [(n * m, rank), (m * p, rank), (p * n, rank)]
    mats = [U, V, W]
    
    for step in range(steps):
        if time.time() > deadline:
            break
        
        # Cooling
        progress = step / steps
        T = T_start * (T_end / T_start) ** progress
        
        # Pick a random matrix, row, col
        mi = np.random.randint(3)
        mat = mats[mi]
        ri, ci = shapes[mi]
        row_idx = np.random.randint(ri)
        col_idx = np.random.randint(rank)
        
        old_val = mat[row_idx, col_idx]
        
        # Pick a new value from allowed_vals, different from old_val
        # To be efficient, just pick a random index different from current
        new_idx = np.random.randint(5)
        # Ensure it's different
        while allowed_vals[new_idx] == old_val:
            new_idx = np.random.randint(5)
        new_val = allowed_vals[new_idx]
        
        mat[row_idx, col_idx] = new_val
        
        new_error = _verify_fast(U, V, W, n, m, p)
        
        if new_error < current_error:
            current_error = new_error
            if current_error < 1e-9:
                # Found a solution
                if _check_half_integer_matrices(U.tolist(), V.tolist(), W.tolist()):
                    return U.tolist(), V.tolist(), W.tolist()
                # If not half integer (shouldn't happen with this init/step), snap and check
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                if _verify_fast(U_snap, V_snap, W_snap, n, m, p) < 1e-9:
                    return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
            if current_error < best_error:
                best_error = current_error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        else:
            # Accept worse move with probability exp(-delta/T)
            delta = new_error - current_error
            if delta < 0:
                prob = 1.0
            else:
                prob = np.exp(-delta / T)
            if np.random.random() < prob:
                current_error = new_error
            else:
                # Reject move
                mat[row_idx, col_idx] = old_val
        
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # We try ranks from 23 down to 15 (or lower if time permits)
    # Each rank gets a portion of the remaining time.
    
    ranks_to_try = range(23, 14, -1)
    
    # Estimate time per rank attempt. 
    # Let's allocate 5 seconds per rank attempt initially, then use remainder for the last one.
    time_per_rank = 5.0
    
    current_time = start_time
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Calculate remaining time
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
        
        # If this is the last rank or we have plenty of time, give it more
        # Simple strategy: give each rank 5s, unless it's the last one, then give the rest.
        # But we don't know which is last until we finish the loop.
        # Let's just use a fixed time slice per rank, e.g., 5s.
        rank_deadline = time.time() + time_per_rank
        if rank_deadline > deadline:
            rank_deadline = deadline
            
        result = _local_search(n, m, p, r, rank_deadline, seed=42 + r)
        
        if result is not None:
            U_r, V_r, W_r = result
            # Final verification
            if _verify_fast(np.array(U_r), np.array(V_r), np.array(W_r), n, m, p) < 1e-9:
                if _check_half_integer_matrices(U_r, V_r, W_r):
                    best_result = {"U": U_r, "V": V_r, "W": W_r}
                    best_rank = r
                    # If we found a lower rank, we can try to go even lower?
                    # The loop continues to lower r, so yes.
                    # But we might want to spend more time on the new best?
                    # For now, keep the schedule.
    
    return best_result
# EVOLVE-BLOCK-END
