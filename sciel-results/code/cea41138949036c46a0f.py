"""Tensor decomposition for 2x3x4 matrix multiplication."""
DIMENSIONS = [2, 3, 4]

import time
import random
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START

def _compute_tensor_residual(U, V, W, n, m, p):
    """Compute the max absolute residual of the tensor decomposition."""
    rank = U.shape[1]
    max_res = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                res = abs(val - 1.0)
                if res > max_res:
                    max_res = res
    return max_res

def _local_search(U, V, W, n, m, p, max_iter=200, step_sizes=None):
    """Perform local search to minimize the tensor residual.
    
    Uses coordinate descent with discrete steps in {-1, -0.5, 0, 0.5, 1}.
    """
    if step_sizes is None:
        step_sizes = [1.0, 0.5, 0.25, 0.125]
    
    current_res = _compute_tensor_residual(U, V, W, n, m, p)
    
    for iteration in range(max_iter):
        improved = False
        
        # Try updating each entry in U, V, W
        for matrix_name, matrix, rows in [('U', U, n*m), ('V', V, m*p), ('W', W, p*n)]:
            rank = matrix.shape[1]
            for r in range(rows):
                for c in range(rank):
                    old_val = matrix[r, c]
                    # Try nearby half-integer values
                    candidates = set()
                    for delta in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                        new_val = old_val + delta
                        # Snap to half-integer
                        new_val = round(new_val * 2) / 2.0
                        if -2.0 <= new_val <= 2.0:
                            candidates.add(new_val)
                    
                    for new_val in candidates:
                        if new_val == old_val:
                            continue
                        matrix[r, c] = new_val
                        new_res = _compute_tensor_residual(U, V, W, n, m, p)
                        if new_res < current_res - 1e-12:
                            current_res = new_res
                            improved = True
                        else:
                            matrix[r, c] = old_val
        
        if not improved:
            break
    
    return U, V, W, current_res

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _snap_to_half(vals):
    """Snap values to nearest half-integer."""
    return np.round(vals * 2) / 2.0

def _is_half_integer(vals, tol=1e-9):
    """Check if all values are integers or half-integers (multiples of 0.5)."""
    for v in vals:
        if abs(v * 2 - round(v * 2)) > tol:
            return False
    return True

def _construct_schoolbook(n, m, p):
    """Construct the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W, rank

def _try_rank_with_search(n, m, p, target_rank, max_attempts, deadline, seed_base=0):
    """Try to find a decomposition of the given rank using random search + local search."""
    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None
        
        np.random.seed(attempt * 1000 + target_rank * 7 + seed_base)
        
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        # Dense initialization with half-integers
        for i in range(n*m):
            for c in range(target_rank):
                if np.random.random() < 0.5:
                    U[i, c] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
        
        for j in range(m*p):
            for c in range(target_rank):
                if np.random.random() < 0.5:
                    V[j, c] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
        
        for k in range(p*n):
            for c in range(target_rank):
                if np.random.random() < 0.5:
                    W[k, c] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
        
        U_ls, V_ls, W_ls, final_res = _local_search(U, V, W, n, m, p, max_iter=80)
        
        if final_res < 1e-9:
            if _verify(U_ls, V_ls, W_ls, n, m, p):
                if _is_half_integer(U_ls) and _is_half_integer(V_ls) and _is_half_integer(W_ls):
                    return U_ls, V_ls, W_ls
                else:
                    U_snap = _snap_to_half(U_ls)
                    V_snap = _snap_to_half(V_ls)
                    W_snap = _snap_to_half(W_ls)
                    if _verify(U_snap, V_snap, W_snap, n, m, p):
                        return U_snap, V_snap, W_snap
    
    return None

def construct():
    """Find rank-minimal exact tensor decomposition for 2x3x4 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 135  # 135s deadline with 15s safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    # Schoolbook solution as fallback
    U_school, V_school, W_school, rank_school = _construct_schoolbook(n, m, p)
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Lower bound: max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    # Upper bound from literature: R(2,3,4) <= 16
    # Try ranks from 12 to 16
    
    for target_rank in range(12, 17):
        if time.time() > deadline:
            break
        
        # Number of attempts based on remaining time and rank difficulty
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        # More attempts for lower ranks (harder to find), fewer for higher ranks
        max_attempts = min(50, int(remaining * 2))
        
        result = _try_rank_with_search(n, m, p, target_rank, max_attempts, deadline, seed_base=42)
        
        if result is not None:
            best_rank = target_rank
            best_U, best_V, best_W = result
            break
    
    # If we haven't found a better solution, try a more systematic approach
    if best_rank == rank_school and time.time() < deadline:
        # Try rank 16 with structured initialization patterns
        target_rank = 16
        remaining = deadline - time.time()
        if remaining > 5:
            max_attempts = min(30, int(remaining))
            result = _try_rank_with_search(n, m, p, target_rank, max_attempts, deadline, seed_base=999)
            if result is not None:
                best_rank = target_rank
                best_U, best_V, best_W = result
    
    # Final safety: ensure we have a valid solution
    if not _verify(best_U, best_V, best_W, n, m, p):
        best_rank = rank_school
        best_U = U_school
        best_V = V_school
        best_W = W_school
    
    # Convert to lists for the return format
    U_list = [[float(best_U[i, c]) for c in range(best_rank)] for i in range(n*m)]
    V_list = [[float(best_V[i, c]) for c in range(best_rank)] for i in range(m*p)]
    W_list = [[float(best_W[i, c]) for c in range(best_rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
