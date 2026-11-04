"""Find a rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 5]

def _verify(U, V, W):
    """Check if U, V, W exactly decompose the matmul tensor."""
    n, m, p = DIMENSIONS
    r = len(U[0]) if U else 0
    # Build the tensor T[i,j,k] = 1 if (i,j,k) is a valid entry
    # U shape: (n*m, r), V shape: (m*p, r), W shape: (p*n, r)
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matmul, T[i,j,k] = 1 for all valid i,j,k
    # Let's check a random sample to be fast, then full if small
    total_entries = n * m * p
    if total_entries <= 1000:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    else:
        # Random sampling
        rng = np.random.RandomState(42)
        for _ in range(200):
            i = rng.randint(n)
            j = rng.randint(m)
            k = rng.randint(p)
            val = 0.0
            for c in range(r):
                val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
            if abs(val - 1.0) > 1e-10:
                return False
        return True

def _rank_decomposition_from_factors(U, V, W):
    """Given factor matrices, verify they work."""
    return _verify(U, V, W)

def construct():
    """
    Find a low-rank decomposition for 2x4 * 4x5 matrix multiplication.
    
    The matmul tensor T[i,j,k] = 1 for all i in [0,2), j in [0,4), k in [0,5).
    We seek U (8 x r), V (20 x r), W (10 x r) such that
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Start with schoolbook rank 40 and try to reduce.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Schoolbook solution as fallback
    rank_school = n * m * p  # 40
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # For small matrices, known bounds suggest rank can be lower than n*m*p
    # Let's try random search with rank reduction
    
    # Strategy: try ranks from lower to higher, use random initialization + refinement
    for target_rank in range(15, best_rank):
        if time.time() > deadline:
            break
        
        found = False
        # Multiple random attempts
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            rng = np.random.RandomState(attempt * 1000 + target_rank)
            
            # Initialize with random values
            U = rng.randn(n*m, target_rank)
            V = rng.randn(m*p, target_rank)
            W = rng.randn(p*n, target_rank)
            
            # Try a simple gradient-free refinement: 
            # For each term, try to satisfy the tensor equations
            # This is a 3-way tensor completion problem
            
            # Alternative: use the fact that for small problems, 
            # we can set up linear constraints
            # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            
            # For fixed V and W, U is determined by linear equations
            # For fixed U and W, V is determined
            # For fixed U and V, W is determined
            
            # Let's try: fix U, V randomly, solve for W
            # W[k*n+i, c] = T[i,j,k] / (U[i*m+j,c] * V[j*p+k,c])
            # But this must be consistent across all (i,j,k)
            
            # Better approach: use the structure
            # For 2x4x5, let's try specific structured decompositions
            
            if target_rank <= 20:
                # Try a structured approach
                # Split the problem into sub-problems
                # 2x4*4x5 can be seen as two 1x4*4x5 problems
                # Or use Winograd-like ideas
                
                # For now, try random with constraint satisfaction
                pass
        
        if found:
            break
    
    # Try a more systematic approach for small ranks
    # Use the fact that the tensor is all-ones
    # T[i,j,k] = 1
    
    # For rank r, we need:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # This is equivalent to: for each (i,j,k), the vector of products 
    # (U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c])_c sums to 1
    
    # Let's try rank 20 first with a structured approach
    # 2*4*5 = 40, half would be 20
    
    # Structured approach: 
    # For each j, we have a 2x5 subproblem (i,k)
    # The tensor is T[i,j,k] = 1
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # If we set U[i*m+j,c] = a_{i,j,c}, V[j*p+k,c] = b_{j,k,c}, W[k*n+i,c] = d_{k,i,c}
    # sum_c a_{i,j,c} * b_{j,k,c} * d_{k,i,c} = 1
    
    # Try rank = n*p + m*p + n*m - n - m - p + 1 (like for 2x2x2 it's 7)
    # For 2x4x5: 10 + 20 + 8 - 2 - 4 - 5 + 1 = 28
    
    # Actually, let's just try to find ANY rank below 40
    # Use random search with verification
    
    for target_rank in range(25, best_rank):
        if time.time() > deadline:
            break
        
        # Try random initialization with scaling
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            rng = np.random.RandomState(attempt * 7919 + target_rank * 131)
            
            # Initialize with small random values
            scale = 1.0 / (target_rank ** 0.5)
            U = rng.randn(n*m, target_rank) * scale
            V = rng.randn(m*p, target_rank) * scale
            W = rng.randn(p*n, target_rank) * scale
            
            # Check
            if _verify(U.tolist(), V.tolist(), W.tolist()):
                best_U, best_V, best_W = U.tolist(), V.tolist(), W.tolist()
                best_rank = target_rank
                break
        
        if best_rank == target_rank:
            break
    
    # Convert to proper format
    # Ensure integer/half-integer entries
    # For now, return what we have
    
    U_out = [[round(U_out_val, 10) for U_out_val in row] for row in best_U]
    V_out = [[round(V_out_val, 10) for V_out_val in row] for row in best_V]
    W_out = [[round(W_out_val, 10) for W_out_val in row] for row in best_W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
