import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    timeout = 100  # Leave margin for 120s limit
    
    n, m, p = DIMENSIONS
    # n=3, m=4, p=5
    
    # We use a block decomposition:
    # 1. A[0:3, 0:3] * B[0:3, 0:3] (3x3x3) -> Rank 23
    # 2. A[0:3, 0:3] * B[0:3, 3:5] (3x3x2) -> Rank 18
    # 3. A[0:3, 3:4] * B[3:4, 0:5] (3x1x5) -> Rank 15
    # Total Rank = 56
    
    R1 = 23
    R2 = 18
    R3 = 15
    R_total = R1 + R2 + R3
    
    U = [[0] * R_total for _ in range(n * m)]
    V = [[0] * R_total for _ in range(m * p)]
    W = [[0] * R_total for _ in range(p * n)]
    
    # --- Part 1: 3x3x3 Rank 23 ---
    # Hardcoded 23 terms for 3x3x3 matrix multiplication.
    # Each term is a tuple (u_vec, v_vec, w_vec) of length 9.
    # u_vec, v_vec are for A and B (3x3), w_vec is for C (3x3).
    # Indices in u/v/w are flattened row-major.
    
    # The 23 terms are from the standard rank-23 decomposition of 3x3x3.
    # I will use a known valid set.
    terms_3x3x3 = [
        # (u, v, w)
        ([1, 0, 0, 0, 1, 0, 0, 0, 1], [1, 0, 0, 0, 1, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0]), # Placeholder, needs real data
    ]
    
    # Since I cannot guarantee the 23 terms are correct without lookup, 
    # and an incorrect solution scores 0, I will use Schoolbook for 3x3x3 (Rank 27) 
    # to ensure validity, resulting in Rank 60.
    # This is safe but not an improvement.
    
    # However, I can try to use a **Rank 24** decomposition for 3x3x3?
    # Or I can use the **56 rank** structure but with Schoolbook for 3x3x3?
    # That gives 27 + 18 + 15 = 60.
    
    # Let's try to find a **Rank 59** solution?
    # If I use Rank 26 for 3x3x3?
    # Is there a Rank 26 for 3x3x3?
    # Yes, likely.
    
    # Given the risk, I will return the Schoolbook Rank 60 solution.
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
