import numpy as np
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s to stay safely under 150s
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Schoolbook decomposition of rank n*m*p
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    # Try to find a lower rank decomposition using Strassen-like ideas
    # For 2x3x6, we can try to exploit the structure
    
    # Known result: rank of 2x2x2 is 7 (Strassen)
    # For 2x3x3, the minimum rank is known to be 11
    # For 2x3x6, we can potentially do better than 36
    
    # Let's try a block decomposition approach
    # Split B (3x6) into two 3x3 blocks: B = [B1, B2]
    # Then C = A * B = [A*B1, A*B2]
    # If we have a rank-r decomposition for 2x3x3, we can use it twice
    # Total rank = 2r
    
    # Let's try to construct a rank-11 decomposition for 2x3x3
    # based on known results from the literature
    
    # The 2x3x3 matrix multiplication tensor has rank 11
    # Here's a known decomposition (from Landsberg and others):
    
    # Actually, let me try a different approach:
    # Use the fact that for 2x3x6, we might be able to use a more efficient decomposition
    
    # Let's try to use a random search with verification
    # to see if we can find a lower rank decomposition
    
    import random
    import itertools
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition is correct."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # The tensor T[i,j,k] should equal sum_r U[i,r] * V[j,r] * W[k,r]
        # where i is in [0,n), j is in [0,m), k is in [0,p)
        # But wait, the indices are:
        # U has shape (n*m, rank), indexed by (i,j) -> i*m+j
        # V has shape (m*p, rank), indexed by (j,k) -> j*p+k
        # W has shape (p*n, rank), indexed by (k,i) -> k*n+i
        
        # The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k
        # Actually no, the tensor is defined such that:
        # (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
        # The tensor T_{i,j,k} represents the coefficient of A_{i,j} B_{j,k} in the product
        # So T_{i,j,k} = 1 for all i,j,k
        
        # In the decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for r in range(len(U[0])):
                        actual += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    def try_random_decomposition(rank, n, m, p, max_attempts=100):
        """Try to find a rank-r decomposition using random initialization."""
        for attempt in range(max_attempts):
            # Random initialization with values in {-1, 0, 1}
            U = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n*m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m*p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p*n)]
            
            if verify_decomposition(U, V, W, n, m, p):
                return U, V, W
        
        return None
    
    # Try to find a decomposition with rank less than schoolbook
    # Start from rank 30 and go down
    best_rank = schoolbook_rank
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # For 2x3x6, known lower bounds suggest rank >= 24 or so
    # Let's try a few specific ranks
    
    # First, let's try to see if rank 24 is possible
    # (This is just a heuristic; the actual minimum might be different)
    
    # Given the time constraint, let's be conservative
    # and just return the schoolbook decomposition unless we find something better quickly
    
    # Try a few random attempts with lower ranks
    for target_rank in [30, 28, 26, 24]:
        if time.time() > deadline - 5:
            break
        
        result = try_random_decomposition(target_rank, n, m, p, max_attempts=50)
        if result is not None:
            best_rank = target_rank
            best_U, best_V, best_W = result
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
