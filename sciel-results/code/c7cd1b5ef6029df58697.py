"""Matrix multiplication 3x4x5 - rank-minimal decomposition search."""
DIMENSIONS = [3, 4, 5]

import time
import numpy as np
import random

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify if the given factor matrices decompose the matrix multiplication tensor.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    Convention: W[k*n + i, r] corresponds to output C[i, k]
    """
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    D = np.einsum('xr,yr,zr->xyz', U_np, V_np, W_np)
    
    expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                x = i * m + j
                y = j * p + k
                z = k * n + i
                expected[x, y, z] = 1.0
                
    return np.allclose(D, expected, atol=tol)

def _generate_schoolbook(n, m, p):
    """Generate the standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _compute_error(U, V, W, target):
    """Compute total squared error of the decomposition."""
    D = np.einsum('xr,yr,zr->xyz', U, V, W)
    return np.sum((D - target) ** 2)

def _incremental_error_delta_U(U, V, W, target, r, c, old_val, new_val, n, m, p):
    """
    Compute the change in squared error when U[r,c] changes from old_val to new_val.
    U has shape (n*m, R), so r indexes the x dimension, c indexes the rank.
    D[x,y,z] = sum_r U[x,r]*V[y,r]*W[z,r]
    Changing U[r,c] by delta affects D[x=r, y, z] for all y,z.
    """
    delta = new_val - old_val
    # The change in D at position (r, y, z) is delta * V[y,c] * W[z,c]
    # Let's compute the contribution
    vc = V[:, c]  # shape (m*p,)
    wc = W[:, c]  # shape (p*n,)
    
    # Old contribution to D[r, :, :] from rank c: U[r,c] * outer(V[:,c], W[:,c])
    # New contribution: (U[r,c] + delta) * outer(V[:,c], W[:,c])
    # Change in D[r,y,z] = delta * V[y,c] * W[z,c]
    
    # Current D at (r, :, :) before change
    # D[r,y,z] = sum_r' U[r,r'] * V[y,r'] * W[z,r']
    # We need old D values at row r
    # D_row = sum over r' of U[r,r'] * (V[:,r'] * W[:,r']^T) -- this is expensive
    
    # Alternative: compute the error contribution of just the changed part
    # Let dD[y,z] = delta * V[y,c] * W[z,c]
    # Old error at (r,y,z): (D[r,y,z] - target[r,y,z])^2
    # New error at (r,y,z): (D[r,y,z] + dD[y,z] - target[r,y,z])^2
    # Delta error = sum_{y,z} [(D[r,y,z] + dD[y,z] - t)^2 - (D[r,y,z] - t)^2]
    #            = sum_{y,z} [2*dD[y,z]*(D[r,y,z]-t) + dD[y,z]^2]
    
    # We need D[r, :, :] which requires computing the full slice
    # D[r,y,z] = sum_{r'} U[r,r'] * V[y,r'] * W[z,r']
    # This is: (U[r,:] @ V.T) gives (m*p,) then... no.
    # D[r,y,z] = sum_{r'} U[r,r'] * V[y,r'] * W[z,r']
    # = (V * U[r,:]) @ W.T  where V is (m*p, R), so V * U[r,:] is (m*p,)
    # Then result[y,z] = sum_{r'} (V[y,r'] * U[r,r']) * W[z,r']
    # = (V * U[r,:]) @ W.T  -- shape (m*p, R) @ (R, p*n) = (m*p, p*n)
    
    # Actually more directly:
    # D[r, :, :] = (V * U[r, :]) @ W.T
    # V * U[r,:] broadcasts: V is (m*p, R), U[r,:] is (R,), so result is (m*p, R)
    # Then @ W.T where W.T is (R, p*n) gives (m*p, p*n)
    
    D_row = (V * U[r, :]) @ W.T  # shape (m*p, p*n)
    t_row = target[r, :, :]
    
    dD = delta * np.outer(vc, wc)  # shape (m*p, p*n)
    
    old_diff = D_row - t_row
    new_diff = old_diff + dD
    
    delta_error = np.sum(new_diff ** 2) - np.sum(old_diff ** 2)
    return delta_error

def _incremental_error_delta_V(U, V, W, target, r, c, old_val, new_val, n, m, p):
    """
    Compute the change in squared error when V[r,c] changes from old_val to new_val.
    V has shape (m*p, R), so r indexes the y dimension, c indexes the rank.
    """
    delta = new_val - old_val
    uc = U[:, c]  # shape (n*m,)
    wc = W[:, c]  # shape (p*n,)
    
    # D[x, r, z] = sum_{r'} U[x,r'] * V[r,r'] * W[z,r']
    # Change in D[x, r, z] = delta * U[x,c] * W[z,c]
    # D[:, r, :] = (U * V[r, :]) @ W.T  -- U is (n*m, R), V[r,:] is (R,)
    # result[x, z] = sum_{r'} U[x,r'] * V[r,r'] * W[z,r']
    # = (U * V[r,:]) @ W.T  shape (n*m, R) @ (R, p*n) = (n*m, p*n)
    
    D_col = (U * V[r, :]) @ W.T  # shape (n*m, p*n)
    t_col = target[:, r, :]
    
    dD = delta * np.outer(uc, wc)  # shape (n*m, p*n)
    
    old_diff = D_col - t_col
    new_diff = old_diff + dD
    
    delta_error = np.sum(new_diff ** 2) - np.sum(old_diff ** 2)
    return delta_error

def _incremental_error_delta_W(U, V, W, target, r, c, old_val, new_val, n, m, p):
    """
    Compute the change in squared error when W[r,c] changes from old_val to new_val.
    W has shape (p*n, R), so r indexes the z dimension, c indexes the rank.
    """
    delta = new_val - old_val
    uc = U[:, c]  # shape (n*m,)
    vc = V[:, c]  # shape (m*p,)
    
    # D[x, y, r] = sum_{r'} U[x,r'] * V[y,r'] * W[r,r']
    # Change in D[x, y, r] = delta * U[x,c] * V[y,c]
    # D[:, :, r] = (U * W[r, :]) @ V.T  -- U is (n*m, R), W[r,:] is (R,)
    # result[x, y] = sum_{r'} U[x,r'] * W[r,r'] * V[y,r']
    # = (U * W[r,:]) @ V.T  shape (n*m, R) @ (R, m*p) = (n*m, m*p)
    
    D_slice = (U * W[r, :]) @ V.T  # shape (n*m, m*p)
    t_slice = target[:, :, r]
    
    dD = delta * np.outer(uc, vc)  # shape (n*m, m*p)
    
    old_diff = D_slice - t_slice
    new_diff = old_diff + dD
    
    delta_error = np.sum(new_diff ** 2) - np.sum(old_diff ** 2)
    return delta_error

def _local_search(n, m, p, rank, time_limit, init_type='random'):
    """
    Perform a local search for a valid decomposition of given rank.
    Uses incremental error updates for speed.
    """
    start_time = time.time()
    
    # Precompute target tensor
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                x = i * m + j
                y = j * p + k
                z = k * n + i
                target[x, y, z] = 1.0
                
    # Value set
    values = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    
    # Dimensions
    nx = n * m
    ny = m * p
    nz = p * n
    
    best_overall = float('inf')
    best_solution = None
    
    max_restarts = 50
    restarts = 0
    while restarts < max_restarts and time.time() - start_time < time_limit:
        restarts += 1
        
        if init_type == 'sparse_schoolbook':
            # Initialize from schoolbook with rank columns zeroed out
            U = np.zeros((nx, rank))
            V = np.zeros((ny, rank))
            W = np.zeros((nz, rank))
            # Place schoolbook pattern in first min(rank, n*m*p) columns
            cols_to_use = min(rank, n * m * p)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col >= cols_to_use:
                            break
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1
            # Add small random perturbations to remaining columns
            if cols_to_use < rank:
                for c in range(cols_to_use, rank):
                    U[:, c] = np.random.choice([-1.0, 0.0, 1.0], size=nx)
                    V[:, c] = np.random.choice([-1.0, 0.0, 1.0], size=ny)
                    W[:, c] = np.random.choice([-1.0, 0.0, 1.0], size=nz)
        else:
            # Random initialization with sparsity (Vectorized for speed)
            U = np.zeros((nx, rank))
            V = np.zeros((ny, rank))
            W = np.zeros((nz, rank))
            # Each entry has ~30% chance of being non-zero
            # Generate masks and values
            for mat, rows in [(U, nx), (V, ny), (W, nz)]:
                mask = np.random.rand(rows, rank) < 0.3
                values = np.random.choice([-1.0, -0.5, 0.5, 1.0], size=(rows, rank))
                mat[:] = np.where(mask, values, 0.0)
        
        # Calculate initial error
        error = _compute_error(U, V, W, target)
        
        if error < 1e-9:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        if error < best_overall:
            best_overall = error
            best_solution = (U.copy(), V.copy(), W.copy())
        
        # Local search loop with incremental updates
        no_improvement_count = 0
        max_no_improvement = 500
        step_count = 0
        max_steps_per_restart = 5000
        
        while (no_improvement_count < max_no_improvement and 
               step_count < max_steps_per_restart and 
               time.time() - start_time < time_limit):
            
            step_count += 1
            # Extra time guard
            if time.time() - start_time > time_limit - 5:
                break
            
            # Pick a random matrix and entry
            mat_idx = random.randint(0, 2)
            if mat_idx == 0:
                mat = U
                rows, cols = nx, rank
            elif mat_idx == 1:
                mat = V
                rows, cols = ny, rank
            else:
                mat = W
                rows, cols = nz, rank
                
            r, c = random.randint(0, rows - 1), random.randint(0, cols - 1)
            old_val = mat[r, c]
            
            # Try a random new value from the set
            new_val = np.random.choice(values)
            if new_val == old_val:
                continue
            
            # Compute incremental error delta
            if mat_idx == 0:
                delta_err = _incremental_error_delta_U(U, V, W, target, r, c, old_val, new_val, n, m, p)
            elif mat_idx == 1:
                delta_err = _incremental_error_delta_V(U, V, W, target, r, c, old_val, new_val, n, m, p)
            else:
                delta_err = _incremental_error_delta_W(U, V, W, target, r, c, old_val, new_val, n, m, p)
            
            new_error = error + delta_err
            
            if new_error < error - 1e-12:
                mat[r, c] = new_val
                error = new_error
                no_improvement_count = 0
                if error < 1e-9:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                if error < best_overall:
                    best_overall = error
                    best_solution = (U.copy(), V.copy(), W.copy())
            else:
                no_improvement_count += 1
                
                # If stuck, try a larger perturbation (simulated annealing style)
                if no_improvement_count > 500 and random.random() < 0.01:
                    # Accept worse moves occasionally
                    if delta_err < 100.0:
                        mat[r, c] = new_val
                        error = new_error
                        no_improvement_count = 0
    
    # Return best found if it's valid
    if best_solution is not None and best_overall < 1e-9:
        U_b, V_b, W_b = best_solution
        return {"U": U_b.tolist(), "V": V_b.tolist(), "W": W_b.tolist()}
    
    return None

def _search_lower_rank(n, m, p, max_rank, time_limit):
    """
    Attempt to find a decomposition with rank < max_rank using local search.
    """
    start_time = time.time()
    
    # Try ranks from max_rank-1 downwards
    # Allocate time based on rank: lower ranks get more time (harder but more valuable)
    # But we want to find ANY improvement first
    
    # Strategy: Try a sequence of ranks, allocating time proportionally
    # Start with ranks close to schoolbook (easier to find), then go lower
    
    # For 3x4x5, schoolbook is 60. Known bounds suggest rank is between ~45-55.
    # Let's try ranks: 59, 58, 57, ..., down to some limit
    
    # Time allocation: give more time to lower ranks
    # But first, quickly try a few high ranks to see if any work
    
    ranks_to_try = list(range(max_rank - 1, max(1, max_rank - 30), -1))
    
    # Allocate time: first 20% to the top 5 ranks (quick check), rest spread out
    n_quick = min(5, len(ranks_to_try))
    quick_time = time_limit * 0.15
    remaining_time = time_limit - quick_time
    remaining_ranks = ranks_to_try[n_quick:]
    
    if remaining_ranks:
        time_per_rank = remaining_time / len(remaining_ranks)
    else:
        time_per_rank = 0
    
    for idx, rank in enumerate(ranks_to_try):
        if time.time() - start_time >= time_limit:
            break
            
        if idx < n_quick:
            t_alloc = quick_time / n_quick
        else:
            t_alloc = time_per_rank
        
        if t_alloc < 0.5:
            continue
            
        # Alternate between initialization strategies
        init_type = 'sparse_schoolbook' if idx % 2 == 0 else 'random'
        
        res = _local_search(n, m, p, rank, t_alloc, init_type=init_type)
        if res is not None:
            return res
            
    return None

def construct():
    """
    Constructs the minimal rank tensor decomposition for 3x4 matrix multiplication.
    Uses a search strategy to attempt lower ranks, falling back to schoolbook.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safe margin before 120s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Try to find a better rank than schoolbook
    search_time_allowed = 95.0
    
    result = _search_lower_rank(n, m, p, schoolbook_rank, search_time_allowed)
    
    if result is not None:
        return result
        
    # Fallback to schoolbook if search fails or times out
    return _generate_schoolbook(n, m, p)

# EVOLVE-BLOCK-END
