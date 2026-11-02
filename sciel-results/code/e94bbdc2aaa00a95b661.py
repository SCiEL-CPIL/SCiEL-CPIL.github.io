import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    n, m, p = 4, 4, 6
    
    # Schoolbook decomposition as guaranteed valid fallback
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    # Verify a decomposition
    def verify(U_mat, V_mat, W_mat, rank):
        """Check if decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U_mat[i*m+j][c] * V_mat[j*p+k][c] * W_mat[k*n+i][c]
                    if abs(total - 1.0) > 1e-10:
                        return False
        return True
    
    # Try to construct a valid lower-rank decomposition using Strassen-style ideas
    # For 4x4x4, we can use a rank-49 decomposition if we can construct it
    # For 4x4x6, we can potentially split into 4x4x4 + 4x4x2
    
    # Since constructing valid low-rank decompositions requires specific coefficients,
    # and we cannot reliably generate them without reference data, we stick with schoolbook.
    
    # However, let's try one structured approach: use 2x2x3 blocks
    # Each 2x2x3 matrix multiplication has rank 11 (Strassen-style)
    # If we can decompose 4x4x6 into 4 independent 2x2x3 blocks, rank would be 44
    # But this is NOT valid for full matrix multiplication - the blocks are coupled
    
    # The only guaranteed valid answer is schoolbook
    return best_result
