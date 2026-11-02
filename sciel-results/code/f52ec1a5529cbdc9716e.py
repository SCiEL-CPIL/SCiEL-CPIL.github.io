"""Rewritten program for matrix multiplication 4x5 by 5x5 (tensor decomposition)."""
import time
import random

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x5 matrix by a 5x5 matrix.
    
    The schoolbook rank is 4*5*5 = 100. We attempt to find a lower rank decomposition
    using random search with rank reduction strategies.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns: {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    """
    n, m, p = DIMENSIONS
    nm = n * m  # 20
    mp = m * p  # 25
    pn = p * n  # 20
    
    # Deadline guard: 150s wall-clock limit, use 120s for safety
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # Start with schoolbook decomposition
    schoolbook_rank = nm * mp  # Wait, that's wrong. Schoolbook is n*m*p = 100
    schoolbook_rank = n * m * p  # 100
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Build schoolbook decomposition
    U_school = [[0] * schoolbook_rank for _ in range(nm)]
    V_school = [[0] * schoolbook_rank for _ in range(mp)]
    W_school = [[0] * schoolbook_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find lower rank decompositions
    # Strategy: try various rank values and use random initialization + verification
    
    # For matrix multiplication, we can try to find decompositions with rank < 100
    # Known lower bounds for 4x5x5 are not well established, but let's search
    
    # Try ranks from 99 down to some minimum
    # Use a heuristic: for small matrices, rank reductions are possible but limited
    
    # Let's try a structured approach:
    # The tensor T[i,j,k] = delta_{i,k} * A[i,j] * B[j,k] for matmul
    # Actually for the multiplication tensor: T_{i,j,k} = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) entry of C
    # The standard tensor: T(i,j,k) = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k) where i in [0,n), j in [0,m), k in [0,p)
    
    # A rank-r decomposition: T = sum_{r=1}^{R} U[:,r] * V[:,r] * W[:,r]
    # where U[i,j] * V[j,k] * W[k,i] should equal T[i,j,k] = 1 for all valid (i,j,k)
    
    # Actually the tensor for matrix multiplication is:
    # T_{i,j,k} = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
    # In the standard formulation, T_{i,j,k} = 1 for all i,j,k (it's the identity tensor in some sense)
    # More precisely: T_{i,j,k} = 1 if i in [0,n), j in [0,m), k in [0,p)
    
    # For a rank-r decomposition with factors U (nm x r), V (mp x r), W (pn x r):
    # sum_{r'} U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r'] = T[i,j,k]
    
    # T[i,j,k] = 1 for all valid (i,j,k)
    
    # Let's try to find lower rank decompositions using random search
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact."""
        r = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r_idx in range(r):
                        s += U[i * m + j][r_idx] * V[j * p + k][r_idx] * W[k * n + i][r_idx]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def try_random_decomposition(rank, num_trials=50):
        """Try to find a rank-r decomposition using random initialization."""
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Random integer/half-integer entries
            U = [[random.choice([-2, -1, -0.5, 0, 0.5, 1, 2]) for _ in range(rank)] for _ in range(nm)]
            V = [[random.choice([-2, -1, -0.5, 0, 0.5, 1, 2]) for _ in range(rank)] for _ in range(mp)]
            W = [[random.choice([-2, -1, -0.5, 0, 0.5, 1, 2]) for _ in range(rank)] for _ in range(pn)]
            
            if verify(U, V, W):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try structured decompositions first
    # One idea: use the fact that n=4, m=5, p=5
    # Try rank values that are factors or near factors
    
    # Actually, let's think about this differently.
    # The matrix multiplication tensor for n x m * m x p has rank at least n*m*p / something
    # For square n x n * n x n, the rank is at least n^2 (from the identity matrix argument)
    # For 4x5 * 5x5, lower bounds are less clear
    
    # Let's try a greedy rank reduction approach:
    # Start with schoolbook, try to merge terms
    
    # Alternative: use known results or try specific structured decompositions
    
    # For now, let's try random search for ranks slightly below 100
    # This is unlikely to succeed for large reductions, but may find small improvements
    
    # Try ranks 99 down to 90
    for rank in range(schoolbook_rank - 1, max(schoolbook_rank - 20, 0), -1):
        if time.time() > deadline:
            break
        
        result = try_random_decomposition(rank, num_trials=10)
        if result is not None:
            best_rank = rank
            best_result = result
            break
    
    # If no improvement found, return schoolbook
    return best_result
# EVOLVE-BLOCK-END
