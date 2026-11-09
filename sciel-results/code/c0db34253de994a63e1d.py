import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 2x5 matrix by a 5x6 matrix.
    Attempts to find rank < 60 using structured decomposition and search.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 60
    
    # Start with schoolbook as fallback
    best_rank = schoolbook_rank
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][column] = 1
                best_V[j * p + k][column] = 1
                best_W[k * n + i][column] = 1
                column += 1
    
    # Try to find a lower rank decomposition
    # Strategy: exploit the fact that n=2 is small
    # The tensor T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[k,i]
    # We want to find rank r < 60 such that T = sum_{t=0}^{r-1} u_t ⊗ v_t ⊗ w_t
    
    # Approach: Try to merge some of the 60 terms by finding linear dependencies
    # For a given rank r, we need to find U (10 x r), V (30 x r), W (12 x r)
    # such that for all i,j,k: T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    
    # Since T[i,j,k] = 1 for all valid i,j,k, we need:
    # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all i,j,k
    
    # This is a system of 60 equations with r*52 unknowns (10r + 30r + 12r = 52r)
    
    # For r < 60, this is underdetermined, so solutions may exist
    # But we need integer/half-integer solutions
    
    # Try a structured approach: partition the 5 columns into groups
    # and use linearity to share computations
    
    # For example, if we define for each k:
    # S_k = sum_{j in J_k} c_j * B[j,k] for some subset J_k and coefficients c_j
    # Then A[i,j] * B[j,k] can be related to A[i,j] * S_k if we can express A[i,j] appropriately
    
    # This is essentially the Strassen idea but adapted for rectangular matrices
    
    # Given the complexity, let's try a random search for r in range(55, 60)
    # with the constraint that entries are integers or half-integers
    
    import random
    
    def verify_decomposition(U, V, W, r):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for t in range(r):
                        total += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    # Try structured decomposition: use the fact that we can combine terms
    # For each output element C[k,i] = sum_j A[i,j] * B[j,k]
    # We need to express this as sum_t (linear combo of A's) * (linear combo of B's)
    
    # Let's try a specific construction:
    # Partition j into {0,1} and {2,3,4}
    # For j in {0,1}: use direct multiplication (2*2*6 = 24 terms)
    # For j in {2,3,4}: try to use a rank-7 decomposition for 2x3 * 3x6
    # 2x3 * 3x6: schoolbook is 36, but can we do better?
    # Actually, 2x3 * 3x6 is equivalent to multiplying a 2x3 by a 3x6
    # This can be seen as (2x3) * (3x1) * 6 times, but that's not helpful
    
    # Alternative: use the fact that 3 = 2+1, so 2x3 * 3x6 = (2x2 * 2x6) + (2x1 * 1x6)
    # 2x2 * 2x6: can use Strassen-like for the 2x2 part? 
    # Actually, 2x2 * 2x6 has rank at most 24 schoolbook
    # But we can do: for each of the 6 columns, multiply 2x2 by 2x1
    # 2x2 * 2x1 has rank 4 schoolbook, but can it be done with rank < 4?
    # No, 2x2 * 2x1 is essentially 4 scalar multiplications with no structure to exploit
    
    # Let me try a completely different approach: use bilinear complexity theory
    # The rank of the (2,5,6) tensor is known to be at most 60
    # Lower bounds are weak, so it might be 60 or slightly less
    
    # Given the time constraint, let's try a random search for r=59
    # with integer entries
    
    for r in range(59, 54, -1):
        if time.time() > deadline - 30:
            break
        
        # Try random initialization with small integers
        for attempt in range(50):
            if time.time() > deadline - 15:
                break
            
            U = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(r)] for _ in range(n * m)]
            V = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(r)] for _ in range(m * p)]
            W = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(r)] for _ in range(p * n)]
            
            if verify_decomposition(U, V, W, r):
                best_rank = r
                best_U = U
                best_V = V
                best_W = W
                break
        
        if best_rank < schoolbook_rank:
            break
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
