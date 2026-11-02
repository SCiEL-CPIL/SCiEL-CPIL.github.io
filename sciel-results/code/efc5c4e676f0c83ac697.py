DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=3, p=5
    
    # Part 1: A (3x3) * B1 (3x3) using Landsberg Rank 23
    # Part 2: A (3x3) * B2 (3x2) using Schoolbook Rank 18
    
    rank1 = 23
    rank2 = 18
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: Landsberg 3x3x3 ---
    # Indices:
    # U: i*3+j
    # V: j*3+k (k=0,1,2)
    # W: k*3+i
    
    # The 23 terms of the Landsberg decomposition for 3x3x3
    # (u_idx, v_idx, w_idx, coeff)
    
    # I will use a known valid set of 23 terms.
    # If I am uncertain, I will fall back to schoolbook (27) to ensure validity, 
    # but 23 is the target.
    
    # For safety, I will implement the 23 terms.
    
    # Terms:
    # 1. (0, 0, 0, 1)
    # 2. (0, 1, 1, 1)
    # 3. (0, 2, 2, 1)
    # 4. (3, 3, 0, 1)
    # 5. (3, 4, 1, 1)
    # 6. (3, 5, 2, 1)
    # 7. (6, 6, 0, 1)
    # 8. (6, 7, 1, 1)
    # 9. (6, 8, 2, 1)
    # ...
    
    # I will use the schoolbook for 3x3x3 to guarantee validity if I cannot verify 23.
    # However, 27+18=45 is the baseline.
    
    # I will implement the 23-rank decomposition.
    
    # The 23 terms are:
    # (0,0,0), (0,3,3), (0,6,6),
    # (1,1,1), (1,4,4), (1,7,7),
    # (2,2,2), (2,5,5), (2,8,8),
    # (3,0,3), (3,3,0), (3,6,6),
    # (4,1,4), (4,4,1), (4,7,7),
    # (5,2,5), (5,5,2), (5,8,8),
    # (6,0,6), (6,3,3), (6,6,0),
    # (7,1,7), (7,4,4), (7,7,1),
    # (8,2,8), (8,5,5), (8,8,2)
    # This is 27 terms.
    
    # I will use the Schoolbook decomposition for 3x3x3 (Rank 27) 
    # and Schoolbook for 3x3x2 (Rank 18).
    # Total 45.
    
    # This is not an improvement.
    
    # I will return the Schoolbook decomposition.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
