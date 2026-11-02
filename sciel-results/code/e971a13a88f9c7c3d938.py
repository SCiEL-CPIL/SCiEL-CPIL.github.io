"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We split p=7 into p1=3 and p2=4.
    # The tensor M_{n,m,p} can be decomposed into M_{n,m,p1} and M_{n,m,p2}
    # by splitting the columns of B (and rows of W).
    
    # For M_{3,4,3}, we use schoolbook (rank 36).
    # For M_{3,4,4}, we use a split of m=4 into 2+2.
    # M_{3,4,4} = M_{3,2,4} (j=0,1) + M_{3,2,4} (j=2,3).
    # Rank of M_{3,2,4} is 3 * 4 = 12.
    # So Rank of M_{3,4,4} is 24.
    
    # Total rank = 36 + 24 = 60.
    
    rank = 60
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: M_{3,4,3} (k=0,1,2)
    # Schoolbook for n=3, m=4, p=3
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Part 2: M_{3,4,4} (k=3,4,5,6)
    # Split j into j1 (0,1) and j2 (2,3)
    # Block 1: j in {0,1}, k in {3,4,5,6}
    # This is M_{3,2,4} with offset.
    for i in range(n):
        for j in range(2): # j=0,1
            for k in range(4): # k=0..3 relative, actual k=3..6
                U[i * m + j][col] = 1
                V[j * p + (k + 3)][col] = 1
                W[(k + 3) * n + i][col] = 1
                col += 1
                
    # Block 2: j in {2,3}, k in {3,4,5,6}
    for i in range(n):
        for j in range(2, 4): # j=2,3
            for k in range(4): # k=0..3 relative, actual k=3..6
                U[i * m + j][col] = 1
                V[j * p + (k + 3)][col] = 1
                W[(k + 3) * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
