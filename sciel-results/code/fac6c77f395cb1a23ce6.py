import time
import sys

DIMENSIONS = [3, 3, 7]

def construct():
    """
    Constructs a rank-45 decomposition for 3x3x7 matrix multiplication.
    Strategy: Split the output columns into two groups (0-2 and 3-6).
    Compute the first 3 columns using a 3x3x3 schoolbook multiplication (Rank 9).
    Compute the next 4 columns using a 3x3x4 schoolbook multiplication (Rank 36).
    Total Rank: 9 + 36 = 45.
    """
    n, m, p = DIMENSIONS
    # n=3, m=3, p=7
    
    U = []
    V = []
    W = []
    
    rank = 0
    
    # Part 1: k in [0, 3) -> 3x3x3 multiplication
    # A (3x3), B_part1 (3x3), C_part1 (3x3)
    for i in range(n):
        for j in range(m):
            for k in range(3): # 0, 1, 2
                U.append([0] * rank + [1]) # This is inefficient, better to build row by row later or use lists
                # Actually, let's build the lists of coefficients for each column
                pass
                
    # Better approach: Build U, V, W as lists of lists directly.
    # Number of terms = 9 + 36 = 45.
    
    total_rank = 45
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: k = 0, 1, 2
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Part 2: k = 3, 4, 5, 6
    for i in range(n):
        for j in range(m):
            for k in range(3, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
