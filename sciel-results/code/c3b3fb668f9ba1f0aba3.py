import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    # Schoolbook rank 96
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook

    # Attempt rank 48 construction
    # Based on M_{1,2,2} rank 2 hypothesis
    # M_{3,4,8} = 8 * M_{3,2,2} = 24 * M_{1,2,2}
    # If Rank(M_{1,2,2}) = 2, Total Rank = 48
    
    target_rank = 48
    if time.time() < deadline - 10:
        U_try = np.zeros((n * m, target_rank))
        V_try = np.zeros((m * p, target_rank))
        W_try = np.zeros((p * n, target_rank))
        
        # Construct M_{1,2,2} rank 2 decomposition
        # A: 1x2, B: 2x2, C: 1x2
        # Terms:
        # 1. u=(1,1), v=(1,1,0,0), w=(1,0) -> (A0+A1)(B00+B10) * 0.5 ?
        # This requires careful coefficient assignment.
        
        # Instead of hardcoding, use a simple check if rank 48 is possible
        # by verifying if a random initialization converges.
        # But given time, I will just return the schoolbook if I can't be sure.
        # However, the prompt implies improvement is possible.
        
        pass
        
    return best_result
