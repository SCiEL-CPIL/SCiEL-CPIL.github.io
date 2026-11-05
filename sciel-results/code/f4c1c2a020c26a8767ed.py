"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank
    # W index: k*n + i, where k is output col (0..p-1), i is output row (0..n-1)
    
    # The tensor T_{i,j,k} = 1 if we multiply A_{i,j} * B_{j,k}
    # We want sum_c U_{i*m+j, c} * V_{j*p+k, c} * W_{k*n+i, c} = 1 for all i,j,k
    
    # Schoolbook is rank 42. Can we do better?
    # For 3x3 * 3x7, the lower bound is max(9, 21, 21) = 21.
    # Actually, the border rank lower bound for matrix multiplication M_{n,m,p} is n*m*p / max(n,m,p) ... no.
    # The trivial lower bound is max(n*m, m*p, p*n) = max(9, 21, 21) = 21.
    
    # Let's try to find a rank < 42 decomposition.
    # Since p=7 is large relative to n,m=3, there might be structure.
    
    # Strategy: Try to find a decomposition of rank r < 42.
    # We can use the fact that the 7 columns of the output are somewhat independent.
    
    # Let's try rank 21 first (the theoretical lower bound).
    # If that fails, try 22, 23, etc.
    
    # Actually, let's use a randomized search with verification.
    # We'll try to find integer/half-integer coefficients.
    
    # A better approach: use the known result that M_{3,3,7} might have rank 21 or slightly higher.
    # Let's try to construct a rank 21 decomposition.
    
    # For rank 21, we need 21 terms.
    # U has 9 rows, V has 21 rows, W has 21 rows.
    
    # Let's try a structured approach:
    # Group the 7 columns into sets. 7 = 3 + 3 + 1 or 2+2+2+1 etc.
    
    # Actually, let's just do a simple search for small ranks.
    # Start with rank 21 and go up.
    
    def verify(U, V, W, rank):
        """Check if U, V, W form a valid decomposition."""
        # T_{i,j,k} should be 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    def verify_fast(U, V, W, rank):
        """Faster verification using numpy-like logic (but with fractions for exactness)."""
        import numpy as np
        U_arr = np.array(U, dtype=object)
        V_arr = np.array(V, dtype=object)
        W_arr = np.array(W, dtype=object)
        
        # Reshape: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        # We need to check that for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        
        # This is a tensor contraction.
        # Let's do it directly.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    # Try to find a decomposition with rank < 42.
    # Start with a randomized approach for small ranks.
    
    # Actually, let's use a smarter approach.
    # The matrix multiplication tensor for M_{3,3,7} can be decomposed using
    # the fact that 7 = 2*3 + 1.
    
    # Let's try rank 21.
    # If we can find a rank 21 decomposition, that would be optimal.
    
    # For now, let's just return the schoolbook decomposition if we can't find better.
    # But let's try a bit of search.
    
    # Try rank 21 with random initialization and local search.
    # This is complex, so let's try a simpler structured approach.
    
    # Known: M_{3,3,3} has rank 23 (Laderman).
    # M_{3,3,7} is larger.
    
    # Let's just do a quick search for rank 21-30.
    # If we find one, great. Otherwise, return schoolbook.
    
    for target_rank in range(21, 42):
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with small integers
            U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p*n)]
            
            if verify_fast(U, V, W, target_rank):
                return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
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
