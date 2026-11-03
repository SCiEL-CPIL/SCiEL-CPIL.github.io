"""Matrix multiplication 3x5x6 tensor decomposition."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = _start_time + 110  # 10s safety margin before 120s limit

def construct():
    """
    Schoolbook decomposition for 3x5x6 matrix multiplication, rank 90.
    With time guard to ensure we return before deadline.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > _DEADLINE:
                    # Return partial result (shouldn't happen for rank 90)
                    return {"U": U, "V": V, "W": W}
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
