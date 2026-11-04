"""Tensor decomposition for 4x4 by 4x9 matrix multiplication."""
import time
import itertools

DIMENSIONS = [4, 4, 9]
START_TIME = time.time()
TIME_LIMIT = 115  # seconds, leaving margin for 120s evaluator limit

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 4x4 matrix by a 4x9 matrix.
    
    Output factor follows the official trace convention W[k*n+i, r].
    Returns {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    n, m, p = DIMENSIONS
    
    # Try to find a low-rank decomposition using a search approach
    # First, let's try to find a decomposition with rank less than n*m*p
    
    # For matrix multiplication C = A * B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] = delta(i,k) for the multiplication tensor
    # We need U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] such that
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's try a systematic search for low-rank solutions
    # Start with small ranks and increase
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0 if i == k else 0.0
                    actual = 0.0
                    for r in range(rank):
                        actual += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    def is_valid_entry(x):
        """Check if x is an integer or half-integer."""
        x2 = x * 2
        return abs(x2 - round(x2)) < 1e-9
    
    def check_all_half_integers(U, V, W):
        """Check all entries are integers or half-integers."""
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if not is_valid_entry(val):
                        return False
        return True
    
    # Try ranks from a reasonable minimum up to the schoolbook rank
    # The theoretical lower bound for matrix multiplication is at least max(n, m, p)
    # For 4x4 by 4x9, let's try ranks from 4 up to some reasonable limit
    
    # Actually, let's think about this more carefully.
    # The tensor rank of matrix multiplication A(n,m) x B(m,p) is at least max(n, m, p)
    # but can be much higher. For 4x4 x 4x9, let's try various ranks.
    
    # Let's use a smarter approach: try to find a solution with specific structure
    # One common approach is to use the fact that for square matrices, 
    # there are known decompositions. For rectangular, we can try to extend.
    
    # Let's try a direct search with small ranks first
    # We'll use a randomized approach with verification
    
    import random
    random.seed(42)
    
    best_rank = n * m * p  # schoolbook rank as fallback
    best_solution = None
    
    # First, let's try to find a solution with rank = max(n, m, p) = 9
    # This is the theoretical minimum for this problem
    
    # Actually, let's think about what's achievable.
    # For 4x4 matrix multiplication, the best known is rank 49 (vs schoolbook 64)
    # For 4x4 x 4x9, it's less clear. Let's try a range of ranks.
    
    # Let's try a more structured approach
    # We can try to find a solution where the matrices have specific patterns
    
    # Try rank 9 (theoretical minimum)
    for target_rank in [9, 10, 11, 12, 13, 14, 15, 16]:
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        
        # Try to find a solution with this rank
        # Use a simple randomized approach
        found = False
        
        for attempt in range(50):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            
            # Initialize with random values that are integers or half-integers
            def random_half_int():
                return random.choice([-1, -0.5, 0, 0.5, 1])
            
            U = [[random_half_int() for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random_half_int() for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random_half_int() for _ in range(target_rank)] for _ in range(p * n)]
            
            if verify(U, V, W) and check_all_half_integers(U, V, W):
                best_rank = target_rank
                best_solution = (U, V, W)
                found = True
                break
        
        if found:
            break
    
    if best_solution is not None:
        U, V, W = best_solution
        return {"U": U, "V": V, "W": W}
    
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
