"""Matrix multiplication 2,5,6 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 60

    # Try to find a decomposition with rank less than schoolbook
    # Using a simple randomized search over {-1, 0, 1} entries
    # For the 2x5x6 case, the minimal rank is believed to be 54 (or close)
    # but finding it requires sophisticated algorithms.
    # We'll try a few random restarts to see if we can find anything better.
    
    import random
    import time
    
    deadline = time.time() + 120.0
    
    # Schoolbook as fallback
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Verify schoolbook is correct
    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        # For each (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Try randomized search for lower rank
    # Start with rank = schoolbook_rank - 1 and try to reduce
    # Given time constraints, just try a few random configurations
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Randomized search: try to find a decomposition with rank < 60
    # This is a hard problem, so we'll do limited attempts
    
    for target_rank in [59, 58, 57, 56, 55, 54]:
        if time.time() > deadline - 10:
            break
        
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            random.seed(attempt * 1000 + target_rank)
            
            # Initialize with random sparse matrices
            U = [[random.choice([-1, 0, 0, 1]) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 0, 1]) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice([-1, 0, 0, 1]) for _ in range(target_rank)] for _ in range(p * n)]
            
            if verify(U, V, W, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
        
        if best_rank < schoolbook_rank:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
