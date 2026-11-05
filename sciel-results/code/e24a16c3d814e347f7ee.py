"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct a low-rank decomposition of the 3x3x7 matrix multiplication tensor.
    The strategy is to partition the 7 output columns into three groups:
    - Group 1: columns 0, 1, 2 (a 3x3x3 sub-problem)
    - Group 2: columns 3, 4, 5 (a 3x3x3 sub-problem)
    - Group 3: column 6 (a 3x3x1 sub-problem)
    
    For each group, we use the naive (schoolbook) decomposition.
    The total rank is 9 + 9 + 9 = 27.
    """
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # n=3, m=3, p=7
    
    # Total rank for naive block decomposition
    rank = n * m * p  # 3*3*7 = 63, but we'll use block decomposition
    
    # Actually, let's use the block decomposition:
    # Split the 7 columns into: [0,1,2], [3,4,5], [6]
    # Block 1: 3x3x3 -> rank 9
    # Block 2: 3x3x3 -> rank 9
    # Block 3: 3x3x1 -> rank 9
    # Total rank = 27
    
    total_rank = 27
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: k in [0, 1, 2]
    for i in range(n):
        for j in range(m):
            for k in range(0, 3):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    # Block 2: k in [3, 4, 5]
    for i in range(n):
        for j in range(m):
            for k in range(3, 6):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    # Block 3: k in [6, 6]
    for i in range(n):
        for j in range(m):
            for k in range(6, 7):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

