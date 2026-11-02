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
    
    # D[x, y, z] = sum_r U[x, r] * V[y, r] * W[z, r]
    # x = i*m + j
    # y = j*p + k
    # z = k*n + i
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

def _local_search(n, m, p, rank, time_limit):
    """
    Perform a local search for a valid decomposition of given rank.
    Uses random restarts and local improvements (flipping entries).
    """
    start_time = time.time()
    best_error = float('inf')
    best_solution = None
    
    # Precompute target tensor indices
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                x = i * m + j
                y = j * p + k
                z = k * n + i
                target[x, y, z] = 1.0
                
    # Possible values for half-integers in a reasonable range
    # We start with a small range and expand if needed, but for now fixed
    values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    restarts = 0
    while time.time() - start_time < time_limit:
        restarts += 1
        # Random initialization
        U = np.random.choice(values, size=(n * m, rank))
        V = np.random.choice(values, size=(m * p, rank))
        W = np.random.choice(values, size=(p * n, rank))
        
        # Calculate initial error
        D = np.einsum('xr,yr,zr->xyz', U, V, W)
        error = np.sum((D - target) ** 2)
        
        if error < 1e-9:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
        # Local search loop
        no_improvement_count = 0
        max_no_improvement = 5000
        while no_improvement_count < max_no_improvement and time.time() - start_time < time_limit:
            # Pick a random matrix and entry to flip
            mat_idx = np.random.randint(3)
            if mat_idx == 0:
                mat = U
                rows, cols = n * m, rank
            elif mat_idx == 1:
                mat = V
                rows, cols = m * p, rank
            else:
                mat = W
                rows, cols = p * n, rank
                
            r, c = np.random.randint(rows), np.random.randint(cols)
            old_val = mat[r, c]
            
            # Try a random new value from the set
            new_val = np.random.choice(values)
            if new_val == old_val:
                continue
                
            mat[r, c] = new_val
            
            # Recalculate error
            # Optimization: Only recalculate affected parts?
            # For now, full recalculation is O(R * n * m * p * n) which is manageable for small R.
            D = np.einsum('xr,yr,zr->xyz', U, V, W)
            new_error = np.sum((D - target) ** 2)
            
            if new_error < error:
                error = new_error
                no_improvement_count = 0
                if error < 1e-9:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            else:
                mat[r, c] = old_val
                no_improvement_count += 1
                
    return None

def _search_lower_rank(n, m, p, max_rank, time_limit):
    """
    Attempt to find a decomposition with rank < max_rank using local search.
    """
    # Try ranks from max_rank-1 downwards
    # We prioritize ranks that are known to be possible or close to bounds
    # For 3x4x5, let's try a few specific ranks
    ranks_to_try = [max_rank - 1, max_rank - 2, max_rank - 4, max_rank - 8]
    # Also try some lower ranks if time permits
    for r in range(max_rank - 10, 0, -1):
        if r not in ranks_to_try:
            ranks_to_try.append(r)
            
    # Sort by rank ascending to find minimal? No, we want to find *any* lower rank first.
    # Actually, finding the *minimal* is the goal.
    # But finding *any* rank < 60 is an improvement.
    # Let's try ranks from 60 downwards, but spend more time on promising ones.
    
    # Heuristic: Start with ranks that are multiples of something?
    # Let's just iterate from max_rank-1 down to 1, but allocate time based on rank.
    # Lower rank is harder, so maybe allocate less time? Or more?
    # Let's allocate a fixed chunk of time per rank attempt.
    
    time_per_rank = time_limit / 10.0 # Try ~10 different ranks
    
    for rank in range(max_rank - 1, 0, -1):
        if time.time() - (time_limit - time_per_rank) > time_limit: # Rough check
            break
        res = _local_search(n, m, p, rank, time_per_rank)
        if res is not None:
            return res
            
    return None

def construct():
    """
    Constructs the minimal rank tensor decomposition for 3x4 matrix multiplication.
    Uses a search strategy to attempt lower ranks, falling back to schoolbook.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10s margin before 120s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Try to find a better rank than schoolbook
    # We limit the search time to ensure we don't timeout
    search_time_allowed = 100.0
    
    result = _search_lower_rank(n, m, p, schoolbook_rank, search_time_allowed)
    
    if result is not None:
        return result
        
    # Fallback to schoolbook if search fails or times out
    if time.time() > deadline:
         # Emergency return, though schoolbook generation is instant
         pass
         
    return _generate_schoolbook(n, m, p)

# EVOLVE-BLOCK-END
