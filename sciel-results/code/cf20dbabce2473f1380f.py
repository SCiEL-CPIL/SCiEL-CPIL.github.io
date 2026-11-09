"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Find a low-rank tensor decomposition for 4x4 * 4x7 matrix multiplication.
    Returns dict with U, V, W matrices.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Try to find a decomposition with rank < n*m*p = 112
    # We'll use a simple approach: try known low-rank decompositions or use
    # the schoolbook as fallback
    
    # First, let's try to find a better decomposition using random search
    # with integer/half-integer constraints
    
    best_rank = n * m * p
    best_decomp = None
    
    # Try various ranks starting from a reasonable lower bound
    # Theoretical lower bound is max(n*m, m*p, p*n) = max(16, 28, 28) = 28
    # But practical lower bounds are higher
    
    # Use a simple randomized search with integer entries
    import random
    
    def validate_decomposition(U, V, W, n, m, p):
        """Check if U, V, W form a valid decomposition."""
        rank = len(U[0]) if U else 0
        if len(U) != n * m:
            return False
        if len(V) != m * p:
            return False
        if len(W) != p * n:
            return False
        
        # Check that all entries are integers or half-integers
        for matrix in [U, V, W]:
            for row in matrix:
                for val in row:
                    if val is None:
                        return False
                    # Check if integer or half-integer
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        
        # Verify the tensor equation
        # For each (i, j, k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
        # Wait, that's not right. The tensor decomposition should satisfy:
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k
        # where T is the multiplication tensor which is 1 for all (i,j,k) in the range
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    def generate_random_decomposition(rank, n, m, p, max_val=2):
        """Generate a random decomposition with integer/half-integer entries."""
        U = [[random.choice([0, 0.5, 1, 1.5, 2, -0.5, -1, -1.5, -2]) for _ in range(rank)] 
             for _ in range(n * m)]
        V = [[random.choice([0, 0.5, 1, 1.5, 2, -0.5, -1, -1.5, -2]) for _ in range(rank)] 
             for _ in range(m * p)]
        W = [[random.choice([0, 0.5, 1, 1.5, 2, -0.5, -1, -1.5, -2]) for _ in range(rank)] 
             for _ in range(p * n)]
        return U, V, W
    
    # Try random search for lower ranks
    for rank in range(max(n*m, m*p, p*n), best_rank):
        if time.time() > deadline:
            break
        
        # Try a few random attempts for this rank
        attempts = 10
        for _ in range(attempts):
            if time.time() > deadline:
                break
            
            U, V, W = generate_random_decomposition(rank, n, m, p)
            
            # Quick check: verify a subset of tensor entries
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        total = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                        if abs(total - 1) > 1e-9:
                            valid = False
                            break
            
            if valid:
                # Full validation
                if validate_decomposition(U, V, W, n, m, p):
                    best_rank = rank
                    best_decomp = (U, V, W)
                    break  # Move to next rank
    
    # If no better decomposition found, use schoolbook
    if best_decomp is None:
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
    else:
        U, V, W = best_decomp
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
