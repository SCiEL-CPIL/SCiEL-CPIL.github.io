"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank by attempting to find linear dependencies
    # For each candidate rank, check if we can find a valid decomposition
    for target_rank in range(rank - 1, max(0, rank - 20), -1):
        if time.time() > deadline - 5:
            break
        
        # Try to find a decomposition of target_rank
        # We'll use a simple approach: try to merge pairs of schoolbook terms
        # by checking if their sum can be represented with one term
        
        success = False
        for merge_attempt in range(10):  # Limit attempts per rank
            if time.time() > deadline - 5:
                break
            
            # Start with schoolbook and try to eliminate one column
            U_try = U.copy()
            V_try = V.copy()
            W_try = W.copy()
            
            # Pick a column to eliminate (try different strategies)
            col_to_remove = merge_attempt % rank
            
            # Check if the column can be expressed as a linear combination of others
            # This is complex, so we'll use a simpler heuristic
            
            # For now, just check if removing this column still satisfies the tensor
            # (This won't work in general, but it's a starting point)
            
            # Actually, let's try a different approach: use random initialization
            # and check if we can satisfy the tensor equations
            
            np.random.seed(42 + merge_attempt)
            U_rand = np.random.randn(n*m, target_rank)
            V_rand = np.random.randn(m*p, target_rank)
            W_rand = np.random.randn(p*n, target_rank)
            
            # Check if this random initialization satisfies the tensor (very unlikely)
            # Instead, let's try to solve the system directly
            
            # Build the system: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # This is a multilinear system, hard to solve directly
            
            # Alternative: use the fact that we can fix two matrices and solve for the third
            # Fix U and V, then for each (i,k), we need:
            # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all j
            # This gives us a system for W[k*n+i,:]
            
            # For a given (i,k), the equation is:
            # sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # Let M_{j,c} = U[i*m+j,c]*V[j*p+k,c]
            # Then sum_c (sum_j M_{j,c}) * W[k*n+i,c] = 1
            # This is a linear equation in W[k*n+i,:]
            
            # But this only gives one equation per (i,k), and we have n*p = 32 equations
            # with target_rank unknowns each, so 32*target_rank unknowns total
            # This is underdetermined for target_rank > 1
            
            # This approach is getting complex. Let's try a simpler heuristic.
            
            # For now, just keep the schoolbook if we can't find a better one
            pass
        
        if not success:
            continue
    
    # Convert to lists for return
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
