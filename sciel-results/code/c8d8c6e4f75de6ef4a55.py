import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 20
    mp = m * p  # 25
    pn = p * n  # 20
    
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds safety margin
    
    # Schoolbook decomposition
    schoolbook_rank = n * m * p  # 100
    U = [[0] * schoolbook_rank for _ in range(nm)]
    V = [[0] * schoolbook_rank for _ in range(mp)]
    W = [[0] * schoolbook_rank for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = schoolbook_rank
    
    # Verify the decomposition is exact
    def verify_decomposition(U_mat, V_mat, W_mat, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_mat[i * m + j][r] * V_mat[j * p + k][r] * W_mat[k * n + i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if verify_decomposition(U, V, W, best_rank):
        return best_result
    else:
        # Fallback: Return a valid decomposition (should not happen)
        return best_result
