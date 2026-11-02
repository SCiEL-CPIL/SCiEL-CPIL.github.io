"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 5]
n, m, p = 2, 5, 5

# EVOLVE-BLOCK-START

def _snapped_halfint(x, tol=1e-6):
    """Snap x to nearest half-integer if within tol, else return x as float."""
    h = round(2 * x) / 2.0
    if abs(x - h) < tol:
        return h
    return x

def _is_valid_decomposition(U, V, W, tol=1e-6):
    """Check if U, V, W form an exact decomposition of the matrix multiplication tensor."""
    rank = len(U[0])
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Check each (i,j,k) entry
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _construct_schoolbook():
    """Return the standard schoolbook decomposition of rank 50."""
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

def _try_column_addition(U, V, W, target_entries, used_entries, max_tries=200):
    """
    Try to add a new column (u_col, v_col, w_col) such that:
    - For each (i,j,k) in target_entries: u[i,j]*v[j,k]*w[k,i] = 1
    - For each (i,j,k) in used_entries: u[i,j]*v[j,k]*w[k,i] = 0
    Returns (u_col, v_col, w_col) on success, None on failure.
    """
    for _ in range(max_tries):
        # Random assignment: assign random nonzero values to some entries
        u_col = [0.0] * (n * m)
        v_col = [0.0] * (m * p)
        w_col = [0.0] * (p * n)
        
        # For each target (i,j,k), we need u[i,j]*v[j,k]*w[k,i] = 1
        # Strategy: pick random u[i,j] and v[j,k], then w[k,i] = 1/(u[i,j]*v[j,k])
        # But this might conflict with other targets or used entries.
        
        # Try: for each target, assign random nonzero to two of the three, compute the third.
        for (i, j, k) in target_entries:
            ui = i * m + j
            vk = j * p + k
            wi = k * n + i
            
            # If u and v are already set (from other targets), check if w can be determined
            if u_col[ui] != 0 and v_col[vk] != 0:
                if abs(u_col[ui] * v_col[vk]) < 1e-12:
                    return None
                w_val = 1.0 / (u_col[ui] * v_col[vk])
                if w_col[wi] != 0 and abs(w_col[wi] - w_val) > 1e-6:
                    return None
                w_col[wi] = w_val
            elif u_col[ui] != 0 and w_col[wi] != 0:
                if abs(u_col[ui] * w_col[wi]) < 1e-12:
                    return None
                v_val = 1.0 / (u_col[ui] * w_col[wi])
                if v_col[vk] != 0 and abs(v_col[vk] - v_val) > 1e-6:
                    return None
                v_col[vk] = v_val
            elif v_col[vk] != 0 and w_col[wi] != 0:
                if abs(v_col[vk] * w_col[wi]) < 1e-12:
                    return None
                u_val = 1.0 / (v_col[vk] * w_col[wi])
                if u_col[ui] != 0 and abs(u_col[ui] - u_val) > 1e-6:
                    return None
                u_col[ui] = u_val
            else:
                # All zero: assign random values
                u_rand = random.choice([1.0, -1.0, 0.5, -0.5, 2.0, -2.0])
                v_rand = random.choice([1.0, -1.0, 0.5, -0.5, 2.0, -2.0])
                u_col[ui] = u_rand
                v_col[vk] = v_rand
                w_col[wi] = 1.0 / (u_rand * v_rand)
        
        # Now check that all used entries have zero contribution
        valid = True
        for (i, j, k) in used_entries:
            prod = u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
            if abs(prod) > 1e-6:
                valid = False
                break
        if not valid:
            continue
        
        # Success
        return (u_col, v_col, w_col)
    
    return None

def _search_decomposition(max_rank=45, time_limit=120, seed=42):
    """
    Search for a low-rank decomposition.
    Returns dict with U, V, W or None.
    """
    random.seed(seed)
    np.random.seed(seed)
    start_time = time.time()
    
    # List of all (i,j,k) entries
    all_entries = list(product(range(n), range(m), range(p)))
    random.shuffle(all_entries)
    
    # Greedy: try to build up column by column
    # Each column matches some subset of entries
    for target_rank in range(max_rank, 49, -1):
        if time.time() - start_time > time_limit - 10:
            break
        
        # Try multiple random orders
        for trial in range(50):
            if time.time() - start_time > time_limit - 10:
                break
            
            entries = list(all_entries)
            random.shuffle(entries)
            
            U = [[0.0] * target_rank for _ in range(n * m)]
            V = [[0.0] * target_rank for _ in range(m * p)]
            W = [[0.0] * target_rank for _ in range(p * n)]
            
            unmatched = set(entries)
            used_entries = set()
            col_idx = 0
            success = True
            
            while unmatched and col_idx < target_rank:
                # Pick a random subset of unmatched entries to try to match in this column
                # Start with 1 entry, increase as we go
                max_targets = min(len(unmatched), 3)
                targets = list(unmatched)
                random.shuffle(targets)
                targets = targets[:random.randint(1, max_targets)]
                
                result = _try_column_addition(U, V, W, targets, used_entries, max_tries=100)
                if result is None:
                    success = False
                    break
                
                u_col, v_col, w_col = result
                for idx in range(n * m):
                    U[idx][col_idx] = u_col[idx]
                for idx in range(m * p):
                    V[idx][col_idx] = v_col[idx]
                for idx in range(p * n):
                    W[idx][col_idx] = w_col[idx]
                
                # Update used and unmatched
                for (i, j, k) in targets:
                    used_entries.add((i, j, k))
                    unmatched.discard((i, j, k))
                
                col_idx += 1
            
            if success and not unmatched:
                # Verify
                if _is_valid_decomposition(U, V, W):
                    # Snap to half-integers
                    for row in U:
                        for c in range(len(row)):
                            row[c] = _snapped_halfint(row[c])
                    for row in V:
                        for c in range(len(row)):
                            row[c] = _snapped_halfint(row[c])
                    for row in W:
                        for c in range(len(row)):
                            row[c] = _snapped_halfint(row[c])
                    # Re-verify after snapping
                    if _is_valid_decomposition(U, V, W):
                        return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x5x5 matrix multiplication.
    Returns dict with keys 'U', 'V', 'W'.
    """
    start_time = time.time()
    
    # Deadline guard: stop searching if we're getting close to the 120s limit
    deadline = start_time + 110  # 10s safety margin
    
    # First, get the schoolbook decomposition as a fallback
    schoolbook = _construct_schoolbook()
    
    # Try to find a better decomposition
    for seed in range(10):
        if time.time() > deadline:
            break
        result = _search_decomposition(max_rank=45, time_limit=min(110, deadline - time.time()), seed=seed)
        if result is not None:
            # Found a valid decomposition with rank < 50
            return result
    
    # Fallback to schoolbook
    return schoolbook
# EVOLVE-BLOCK-END
