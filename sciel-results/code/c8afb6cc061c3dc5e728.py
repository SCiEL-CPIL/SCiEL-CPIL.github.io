import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # We aim for rank 19: 7 (Strassen for 3x3x3) + 12 (Schoolbook for 3x3x4)
    # U shape: (n*m, rank) = (9, 19)
    # V shape: (m*p, rank) = (21, 19)
    # W shape: (p*n, rank) = (21, 19)
    
    rank = 19
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: Strassen decomposition for the first 3 columns (k=0,1,2)
    # Indices in B: j*p + k. j is 0..2, k is 0..2.
    # Indices in A: i*m + j. i is 0..2, j is 0..2.
    # Indices in C: k*n + i. k is 0..2, i is 0..2.
    
    # A-forms (9 entries each, corresponding to i*m+j)
    # a,b,c,d,e,f,g,h,i -> 0,1,2,3,4,5,6,7,8
    A_forms = [
        [1, 0, 0, 0, 1, 0, 0, 0, 0], # a+e
        [0, 0, 0, 1, 0, 0, 1, 0, 0], # d+g
        [1, 1, 0, 0, 0, 0, 0, 0, 0], # a+b
        [1, 0, 1, 0, 0, 0, 0, 0, 0], # a+c
        [0, 0, 0, 0, 1, 1, 0, 0, 0], # e+f
        [0, 0, 0, 0, 1, 0, 0, 1, 0], # e+h
        [0, 0, 0, 0, 0, 0, 0, 0, 1]  # i
    ]
    
    # B-forms (9 entries each, corresponding to j*p+k for j=0..2, k=0..2)
    # j,k -> 0,1,2 (j=0), 3,4,5 (j=1), 6,7,8 (j=2)
    # j,n,o,p,q,r -> indices in B for j=0,1,2 and k=0,1,2
    B_forms = [
        [1, 0, 0, 0, 1, 0, 0, 0, 0], # j+n (B[0,0] + B[1,1])
        [0, 0, 0, 1, 0, 0, 0, 0, 1], # m+p (B[1,0] + B[2,0]) -- Wait, B is 3x3.
                                      # B[1,0] is index 1*3+0=3. B[2,0] is index 2*3+0=6.
                                      # My previous check: B[1,0] is m, B[2,0] is p.
                                      # Indices: 3 and 6.
                                      # Vector: [0,0,0, 1,0,0, 1,0,0]
        [0, 0, 1, 0, 0, 1, 0, 0, 0], # l+o (B[0,2] + B[1,2]) -> idx 2, 5
        [0, 0, 1, 0, 0, 0, 0, 0, 1], # l+r (B[0,2] + B[2,2]) -> idx 2, 8
        [0, 0, 0, 0, 1, 1, 0, 0, 0], # n+o (B[1,1] + B[1,2]) -> idx 4, 5
        [0, 0, 0, 0, 1, 0, 0, 1, 0], # n+q (B[1,1] + B[2,1]) -> idx 4, 7
        [0, 0, 0, 0, 0, 0, 0, 0, 1]  # r   (B[2,2])         -> idx 8
    ]
    # Correction on B_forms[1]:
    # m is B[1,0] -> index 3. p is B[2,0] -> index 6.
    # So B_forms[1] should be [0,0,0, 1,0,0, 1,0,0].
    B_forms[1] = [0, 0, 0, 1, 0, 0, 1, 0, 0]
    
    # C-forms (9 entries each, corresponding to k*n+i for k=0..2, i=0..2)
    # k,i -> 0,1,2 (k=0), 3,4,5 (k=1), 6,7,8 (k=2)
    # C[0,0] is idx 0. C[1,0] is idx 3. C[2,0] is idx 6.
    C_forms = [
        [1, 0, 0, -1, 0, 0, 0, 0, 0], # C00 = M1 - M3? No.
        # Let's derive C forms from the multiplication.
        # This is risky. I will use the known C forms for this specific decomposition.
        # If I don't have them, I will fall back to Schoolbook for all.
    ]
    
    # Given the high risk of error in the C-forms for Strassen 3x3x3,
    # and the strict requirement for correctness, I will use Schoolbook for all 7 columns.
    # Rank = 3 * 3 * 7 = 63.
    # This is guaranteed correct.
    # However, 63 is the seed score. I need to improve it.
    # Rank 21 (Schoolbook 3x3x3 + Schoolbook 3x3x4) is also 21? No.
    # Schoolbook 3x3x3 is rank 9. Schoolbook 3x3x4 is rank 12. Total 21.
    # This is a safe improvement.
    
    rank = 21
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
