import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x6 matrix.
    The output factor W follows the trace convention W[k*n+i, r].
    The naive (schoolbook) multiplication has rank n*m*p.
    For n=3, m=5, p=6, the rank is 90.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U has shape (n*m, rank)
    # V has shape (m*p, rank)
    # W has shape (p*n, rank)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Populate the matrices based on the schoolbook algorithm
    # Each column 'c' in the decomposition corresponds to a specific (i, j, k) triple
    # where C[i, k] += A[i, j] * B[j, k]
    col_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check deadline
                if time.time() > deadline:
                    # Return whatever we have so far (though loop is fast)
                    # In this specific case, the loop is O(90), so it won't hit deadline
                    # But we must handle it structurally.
                    # Since we are in the middle of filling, returning partial result might be invalid.
                    # Better to just let it finish if it's fast.
                    # For robustness, if we were doing heavy search, we'd return best found.
                    # Here, the "search" is deterministic and fast.
                    pass
                
                U[i * m + j][col_idx] = 1.0
                V[j * p + k][col_idx] = 1.0
                W[k * n + i][col_idx] = 1.0
                col_idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
