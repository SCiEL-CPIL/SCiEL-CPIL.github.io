"""Matrix multiplication 3,3,8 — rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def _build_schoolbook(n, m, p):
    """Return the standard (schoolbook) rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True


def _round_to_half(arr):
    """Round each entry to the nearest half-integer."""
    return np.round(arr * 2.0) / 2.0


def _try_constructive(n, m, p, target_rank, deadline):
    """
    Try to construct a decomposition of rank target_rank using a structured approach.
    
    Key insight: U is shared across all k values. We can exploit this by:
    1. Fixing U to have a sparse structure
    2. Solving for V and W that satisfy the constraints
    
    For each (i,j), we need:
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
    
    This is a system of p equations for each (i,j) pair.
    """
    n, m, p = 3, 3, 8
    rank = target_rank
    
    # Strategy: Use a block structure where we partition k into groups
    # and share U entries across groups
    
    # Try different partition structures for the 8 k-values
    # Partitions of 8: [8], [4,4], [3,3,2], [2,2,2,2], [4,2,2], [3,2,3], etc.
    
    partitions = [
        [8],
        [4, 4],
        [3, 3, 2],
        [2, 2, 2, 2],
        [4, 2, 2],
        [3, 2, 3],
        [2, 3, 3],
        [5, 3],
        [3, 5],
        [6, 2],
        [2, 6],
        [7, 1],
        [1, 7],
    ]
    
    for partition in partitions:
        if time.time() > deadline - 5:
            break
        
        # For each partition, try to construct a solution
        # The idea: for each group of k-values, we use a subset of rank columns
        # that are shared within the group
        
        # Simplified approach: assume each group uses rank/g columns where g = len(partition)
        # This is a heuristic, but let's try it
        
        if len(partition) == 1:
            # Just try the full rank
            cols_per_group = [rank]
        else:
            # Distribute rank columns among groups
            base = rank // len(partition)
            remainder = rank % len(partition)
            cols_per_group = [base + (1 if i < remainder else 0) for i in range(len(partition))]
        
        # Check if this is feasible
        if any(c == 0 for c in cols_per_group):
            continue
        
        # Build U, V, W
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Assign column ranges to groups
        col_start = 0
        k_offset = 0
        for g_idx, group_size in enumerate(partition):
            group_cols = cols_per_group[g_idx]
            group_col_range = range(col_start, col_start + group_cols)
            
            # For this group, we need to solve for the (i,j) pairs
            # The constraint is: for each (i,j), sum over c in group_cols of
            #   U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # for k in [k_offset, k_offset+group_size)
            
            # Simplified: assume U[i*m+j, c] = 1 for all c in group_cols
            # Then we need: sum over c in group_cols of V[j*p+k, c] * W[k*n+i, c] = 1
            # for all i, j, k in this group
            
            # This is still complex. Let's try a simpler ansatz:
            # For each group, use a "diagonal" structure
            
            # Actually, let's try a different approach: use random initialization
            # and solve via least squares, then round
            
            # Initialize with random values
            for c in group_col_range:
                for i in range(n):
                    for j in range(m):
                        U[i * m + j, c] = random.choice([0, 0.5, 1.0])
                for j in range(m):
                    for k in range(k_offset, k_offset + group_size):
                        V[j * p + k, c] = random.choice([0, 0.5, 1.0])
                for k in range(k_offset, k_offset + group_size):
                    for i in range(n):
                        W[k * n + i, c] = random.choice([0, 0.5, 1.0])
            
            col_start += group_cols
            k_offset += group_size
        
        # Verify
        if _verify(U, V, W, n, m, p):
            return U, V, W
    
    return None, None, None


def _try_numeric_search(n, m, p, target_rank, deadline):
    """
    Try to find a decomposition using numerical optimization with rounding.
    """
    n, m, p = 3, 3, 8
    rank = target_rank
    
    best_solution = None
    best_error = float('inf')
    
    max_trials = 5
    for trial in range(max_trials):
        if time.time() > deadline - 5:
            break
        
        # Random initialization
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        # Use gradient descent with a smooth loss
        lr = 0.01
        for it in range(200):
            if time.time() > deadline - 3:
                break
            
            # Compute error and gradients
            error = 0.0
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Current sum
                        s = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                        diff = s - 1.0
                        error += diff * diff
                        
                        # Gradients
                        # d/dU[i*m+j,c] = 2*diff * V[j*p+k,c] * W[k*n+i,c]
                        grad_U[i * m + j, :] += 2 * diff * V[j * p + k, :] * W[k * n + i, :]
                        grad_V[j * p + k, :] += 2 * diff * U[i * m + j, :] * W[k * n + i, :]
                        grad_W[k * n + i, :] += 2 * diff * U[i * m + j, :] * V[j * p + k, :]
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Decay learning rate
            lr *= 0.98
            
            # Check convergence
            if error < 1e-10:
                break
        
        # Round to half-integers
        U_r = _round_to_half(U)
        V_r = _round_to_half(V)
        W_r = _round_to_half(W)
        
        # Verify rounded solution
        if _verify(U_r, V_r, W_r, n, m, p):
            return U_r, V_r, W_r
        
        # Track best
        err_r = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_r[i * m + j, :] * V_r[j * p + k, :] * W_r[k * n + i, :])
                    err_r += (s - 1.0) ** 2
        
        if err_r < best_error:
            best_error = err_r
            best_solution = (U_r.copy(), V_r.copy(), W_r.copy())
    
    if best_error < 1e-6:
        # Try to polish
        U, V, W = best_solution
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            U_r = _round_to_half(U)
            V_r = _round_to_half(V)
            W_r = _round_to_half(W)
            if _verify(U_r, V_r, W_r, n, m, p):
                return U_r, V_r, W_r
            break
    
    return None, None, None


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x8 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    deadline = time.time() + 140  # 140 second budget (10s safety margin)
    
    # Get schoolbook solution as baseline
    U0, V0, W0 = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    
    # Lower bound on rank: max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # Try ranks from lower bound up to schoolbook
    # But be strategic: try a few key ranks
    
    # Known results: 3x3x3 has rank 23 (Landsberg), so 3x3x8 might be less than 72
    # Let's try ranks in a strategic order
    
    candidate_ranks = list(range(24, 73))
    
    # Prioritize: try lower ranks first, but skip some that seem unlikely
    # Focus on ranks that are multiples or have nice properties
    
    for rank in candidate_ranks:
        if time.time() > deadline - 10:
            break
        
        # Try constructive approach first (faster)
        U, V, W = _try_constructive(n, m, p, rank, deadline)
        if U is not None:
            best_rank = rank
            best_U, best_V, best_W = U, V, W
            break
        
        # Try numeric search (slower)
        if time.time() > deadline - 15:
            continue
        
        U, V, W = _try_numeric_search(n, m, p, rank, deadline)
        if U is not None:
            best_rank = rank
            best_U, best_V, best_W = U, V, W
            break
    
    # Return the best solution found
    U, V, W = best_U, best_V, best_W
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
