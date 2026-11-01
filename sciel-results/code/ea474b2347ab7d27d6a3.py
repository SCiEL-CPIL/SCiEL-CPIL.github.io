"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return U, V, W as lists for the schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify exact decomposition with tolerance."""
    rank = len(U[0])
    # Use numpy for speed
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
    U_3d = U_np.reshape(n, m, rank)
    V_3d = V_np.reshape(m, p, rank)
    W_3d = W_np.reshape(p, n, rank)
    
    # Compute tensor T[i, j, k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # This is a contraction over r.
    # We can do this via einsum: 'ijr,jkr,kir->ijk'
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    # Check if T is all 1s
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True

def _is_half_integer(x):
    return abs(2 * x - round(2 * x)) < 1e-9

def _verify_entries(U, V, W):
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if not _is_half_integer(val):
                    return False
    return True

def _try_rank(n, m, p, rank, deadline, rng_seed):
    """
    Attempt to find a rank-r decomposition using vectorized ALS.
    Returns (U, V, W) if successful, else None.
    """
    if time.time() > deadline:
        return None
        
    rng = np.random.default_rng(rng_seed)
    
    # Initialize with random values
    # Scaling is important. Let's start with small random values.
    U = rng.normal(0, 1, (n * m, rank))
    V = rng.normal(0, 1, (m * p, rank))
    W = rng.normal(0, 1, (p * n, rank))
    
    # Reshape for efficient operations
    # U: (n, m, r), V: (m, p, r), W: (p, n, r)
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # ALS Loop
    # We have 3 blocks of variables.
    # 1. Fix V, W. Solve for U.
    #    For each (i, j), U[i,j,:] is a vector x.
    #    Equation: sum_r x_r V[j,k,r] W[k,i,r] = 1 for all k.
    #    Let M_VW[j, i, k, r] = V[j,k,r] * W[k,i,r].
    #    Then for fixed i, j: sum_r x_r M_VW[j, i, k, r] = 1 for all k.
    #    This is a linear system: A_ij x = 1, where A_ij[k, r] = M_VW[j, i, k, r].
    #    A_ij is p x rank.
    
    # Pre-compute M_VW? It depends on V and W.
    # M_VW shape: (m, n, p, rank)
    # V_3d: (m, p, rank) -> V_3d[:, :, None, :] (m, p, 1, rank)
    # W_3d: (p, n, rank) -> W_3d[None, :, :, :] (1, n, p, rank)
    # Wait, indices:
    # V[j, k, r]
    # W[k, i, r]
    # We want M[j, i, k, r] = V[j, k, r] * W[k, i, r]
    # V_3d is (m, p, rank). We need to broadcast k with k.
    # Let's use einsum to construct the linear system matrices efficiently?
    # Or just loop over i, j, k? 
    # n=2, m=5, p=6. rank ~ 30-60.
    # Loops: i(2)*j(5)*k(6) = 60 systems of size p x rank.
    # Solving 60 systems of 6x30 is fast.
    
    max_iters = 200
    for it in range(max_iters):
        if time.time() > deadline - 2:
            break
            
        # 1. Update U
        # For each i, j:
        # A[k, r] = V[j, k, r] * W[k, i, r]
        # b = 1
        for i in range(n):
            for j in range(m):
                # Construct A (p, rank)
                # V_3d[j] is (p, rank)
                # W_3d[:, i, :] is (p, rank)
                A = V_3d[j] * W_3d[:, i, :]
                try:
                    # Solve A x = 1
                    x = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                    U_3d[i, j, :] = x
                except:
                    pass
        
        # 2. Update V
        # For each j, k:
        # A[i, r] = U[i, j, r] * W[k, i, r]
        # b = 1
        for j in range(m):
            for k in range(p):
                # U_3d[:, j, :] is (n, rank)
                # W_3d[k, :, :] is (n, rank)
                A = U_3d[:, j, :] * W_3d[k, :, :]
                try:
                    x = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                    V_3d[j, k, :] = x
                except:
                    pass
        
        # 3. Update W
        # For each k, i:
        # A[j, r] = U[i, j, r] * V[j, k, r]
        # b = 1
        for k in range(p):
            for i in range(n):
                # U_3d[i, :, :] is (m, rank)
                # V_3d[:, k, :] is (m, rank)
                A = U_3d[i, :, :] * V_3d[:, k, :]
                try:
                    x = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                    W_3d[k, i, :] = x
                except:
                    pass
        
        # Compute Error
        # T = einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        err = np.sqrt(np.mean((T - 1.0)**2))
        
        if err < best_err:
            best_err = err
            best_U = U_3d.copy()
            best_V = V_3d.copy()
            best_W = W_3d.copy()
            
        if err < 1e-9:
            break

    if best_err < 1e-6 and best_U is not None:
        # Convert back to lists
        U_list = best_U.reshape(n * m, rank).tolist()
        V_list = best_V.reshape(m * p, rank).tolist()
        W_list = best_W.reshape(p * n, rank).tolist()
        
        # Round to nearest half-integer
        def _round_halfint(x):
            return round(2 * x) / 2.0
            
        U_list = [[_round_halfint(x) for x in row] for row in U_list]
        V_list = [[_round_halfint(x) for x in row] for row in V_list]
        W_list = [[_round_halfint(x) for x in row] for row in W_list]
        
        # Verify
        if _verify_decomposition(U_list, V_list, W_list, n, m, p) and _verify_entries(U_list, V_list, W_list):
            return U_list, V_list, W_list
            
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    
    # Lower bound for rank
    lower_bound = max(n * m, m * p, p * n)
    
    # Try ranks from lower_bound up to schoolbook rank
    # We iterate in increasing order of rank to find the *minimal* rank solution.
    # If we find a valid solution at rank r, we return it immediately.
    # This ensures we don't waste time on higher ranks if a lower one is found.
    # However, ALS might get stuck in local minima for lower ranks.
    # To mitigate this, we use multiple random seeds for each rank.
    
    # If we don't find anything better, return schoolbook.
    # Note: Schoolbook rank is n*m*p.
    
    # Heuristic: Lower ranks are harder. 
    # Let's try a few random seeds for each rank.
    
    # Time budget per rank?
    # Total time ~ 140s.
    # Ranks: 30 to 60. 31 ranks.
    # ~4s per rank.
    
    for r in range(lower_bound, n * m * p + 1):
        if time.time() > deadline - 5:
            break
            
        # Try multiple random initializations for this rank
        # More trials for lower ranks? Or equal?
        # Let's use 5 trials per rank.
        trials = 5
        time_per_trial = (deadline - 5 - time.time()) / trials
        if time_per_trial < 1.0:
             # Not enough time for this rank, break?
             # Or just do 1 trial?
             trials = 1
             
        for trial in range(trials):
            if time.time() > deadline - 2:
                break
            seed = r * 100 + trial
            result = _try_rank(n, m, p, r, deadline, seed)
            if result is not None:
                U, V, W = result
                return {"U": U, "V": V, "W": W}
                
    # Fallback to schoolbook if no better found
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
