"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    # Try to find a lower-rank decomposition
    # Start with schoolbook as fallback
    rank_schoolbook = n * m * p
    
    # Build the target tensor
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # We need: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # And for invalid index combos, the sum should be 0 (but they don't appear in the tensor)
    
    # Actually, the tensor is only defined on valid (i,j,k) triples.
    # The decomposition must satisfy: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let's try to use a rank-reduction approach
    # First, let's try rank 48 and see if we can find a solution
    
    # Use a simple approach: try random initializations and use iterative refinement
    # Or use known structural decompositions
    
    # For 2x3 x 3x9, let's try to exploit the structure
    # Split into three 2x3 x 3x3 blocks
    
    # Try rank 45 first (5 per block of 3x3 for each of 3 blocks, times 3 blocks)
    # Actually, let's just try to find any rank < 54 solution
    
    # Simple approach: use the schoolbook but try to merge some terms
    # If we can find two terms that can be combined...
    
    # Let me try a direct numerical approach with small rank
    best_result = None
    
    for target_rank in [48, 45, 42, 40, 36, 30]:
        if time.time() > deadline - 10:
            break
        # Try to find a decomposition of this rank
        # Use a simple iterative approach
        success = False
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            # Random initialization with structure
            np.random.seed(attempt * 1000 + target_rank)
            
            # Initialize U, V, W randomly
            U_init = np.random.randn(n * m, target_rank) * 0.1
            V_init = np.random.randn(m * p, target_rank) * 0.1
            W_init = np.random.randn(p * n, target_rank) * 0.1
            
            # Use alternating least squares to fit the tensor
            # For each (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            # This is a 3-way tensor fitting problem
            
            # We can fix one matrix and solve for the others
            # But this is non-convex. Let's use a simpler approach.
            
            # Actually, let's use a structured approach:
            # The tensor has a specific structure. Let's try to exploit it.
            
            # For now, let's just return the schoolbook if we can't find better
            pass
        
        if success:
            break
    
    # Fallback: schoolbook decomposition
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
# EVOLVE-BLOCK-END
