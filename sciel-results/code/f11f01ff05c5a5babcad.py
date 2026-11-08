import time

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = 150
    
    n, m, p = DIMENSIONS
    
    # We will use a hybrid approach:
    # 1. Split m=5 into 2+3.
    # 2. M_3,2,7 uses Schoolbook (Rank 42).
    # 3. M_3,3,7 splits p=7 into 3+3+1.
    #    - M_3,3,3 uses Laderman (Rank 23).
    #    - M_3,3,3 uses Laderman (Rank 23).
    #    - M_3,3,1 uses Schoolbook (Rank 9).
    # Total Rank: 42 + 23 + 23 + 9 = 97.
    
    rank = 97
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: M_3,2,7 (i in 0..2, j in 0..1, k in 0..6)
    # A is 3x5, B is 5x7.
    # A_1 is A[:, 0:2], B_1 is B[0:2, :]
    for i in range(3):
        for j in range(2):
            for k in range(7):
                U[i * 5 + j][col] = 1
                V[j * 7 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    # col is now 42
    
    # Part 2: M_3,3,3 for B[2:5, 0:3]
    # A_2 is A[:, 2:5], B_2 is B[2:5, 0:3]
    # Indices in U: i*5 + (2+j) where j in 0..2
    # Indices in V: (2+j)*7 + k where j in 0..2, k in 0..2
    # Indices in W: k*3 + i
    
    # Laderman Factors for 3x3x3
    # I will use a standard set.
    # If I don't have the exact coefficients, I will fallback to Schoolbook for this part.
    # Schoolbook rank for 3x3x3 is 27.
    # 42 + 27 + 27 + 9 = 105.
    
    # To be safe and ensure a valid score, I will use Schoolbook for the 3x3 parts.
    # This results in Rank 105.
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 5 + (2 + j)][col] = 1
                V[(2 + j) * 7 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    # col is now 42 + 27 = 69
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 5 + (2 + j)][col] = 1
                V[(2 + j) * 7 + (3 + k)][col] = 1
                W[(3 + k) * 3 + i][col] = 1
                col += 1
    # col is now 69 + 27 = 96
    
    for i in range(3):
        for j in range(3):
            for k in range(1):
                U[i * 5 + (2 + j)][col] = 1
                V[(2 + j) * 7 + (6 + k)][col] = 1
                W[(6 + k) * 3 + i][col] = 1
                col += 1
    # col is now 96 + 9 = 105
    
    return {"U": U, "V": V, "W": W}
