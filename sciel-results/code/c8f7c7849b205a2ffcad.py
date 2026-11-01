import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 19 construction:
    # Split B (3x5) into B1 (3x2) and B2 (3x3).
    # For each part, split A's columns (j) into j in {0,1} and j=2.
    # Use Strassen (rank 7) for the 2x2x2 sub-problem.
    # Use Schoolbook (rank 2 or 3) for the remaining part.
    
    # Strassen decomposition for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We will build U, V, W matrices.
    # U shape: (n*m, rank) = (6, 19)
    # V shape: (m*p, rank) = (15, 19)
    # W shape: (p*n, rank) = (10, 19)
    
    rank = 19
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Part 1: k in {0, 1}
    # Sub-part 1a: j in {0, 1}, k in {0, 1} -> Strassen
    # A indices: i in {0,1}, j in {0,1} -> U[i*m+j]
    # B indices: j in {0,1}, k in {0,1} -> V[j*p+k]
    # C indices: k in {0,1}, i in {0,1} -> W[k*n+i]
    
    # Strassen coefficients
    # U, V, W for the 7 terms
    # Term 1: (a+d)(e+h)
    U[0][col] += 1; U[3][col] += 1 # a, d (j=0,1)
    V[0][col] += 1; V[1][col] += 1 # e, f? No, V[j*p+k]. j=0,1. k=0,1.
    # V[0*p+0]=V[0] (e), V[0*p+1]=V[1] (f), V[1*p+0]=V[5] (g), V[1*p+1]=V[6] (h).
    # Wait, V index is j*p+k.
    # j=0, k=0 -> 0. j=0, k=1 -> 1.
    # j=1, k=0 -> 5. j=1, k=1 -> 6.
    # So e=V[0], f=V[1], g=V[5], h=V[6].
    V[0][col] += 1; V[6][col] += 1 # e, h
    W[0][col] += 1; W[1][col] += 1; W[2][col] += 1; W[3][col] += 1 # C11, C12, C21, C22
    # W indices: k*n+i.
    # k=0, i=0 -> 0 (C11). k=0, i=1 -> 1 (C12).
    # k=1, i=0 -> 2 (C21). k=1, i=1 -> 3 (C22).
    # M1 contributes to all 4.
    col += 1
    
    # Term 2: (c+d)e
    U[3][col] += 1; U[4][col] += 1 # c, d (j=1)
    V[0][col] += 1 # e
    W[2][col] += 1; W[3][col] += 1 # C21, C22
    col += 1
    
    # Term 3: a(f-h)
    U[0][col] += 1 # a
    V[1][col] += 1; V[6][col] -= 1 # f, -h
    W[1][col] += 1; W[3][col] += 1 # C12, C22
    col += 1
    
    # Term 4: d(g-e)
    U[4][col] += 1 # d
    V[5][col] += 1; V[0][col] -= 1 # g, -e
    W[1][col] += 1; W[2][col] += 1 # C12, C21
    col += 1
    
    # Term 5: (a+b)h
    U[0][col] += 1; U[1][col] += 1 # a, b
    V[6][col] += 1 # h
    W[0][col] += 1; W[1][col] += 1 # C11, C12
    col += 1
    
    # Term 6: (c-a)(e+f)
    U[3][col] += 1; U[0][col] -= 1 # c, -a
    V[0][col] += 1; V[1][col] += 1 # e, f
    W[2][col] += 1; W[3][col] += 1 # C21, C22
    col += 1
    
    # Term 7: (b-d)(g+h)
    U[1][col] += 1; U[4][col] -= 1 # b, -d
    V[5][col] += 1; V[6][col] += 1 # g, h
    W[0][col] += 1; W[3][col] += 1 # C11, C22
    col += 1
    
    # Sub-part 1b: j=2, k in {0, 1} -> Schoolbook
    # A index: i in {0,1}, j=2 -> U[i*m+2] = U[2], U[5].
    # B index: j=2, k in {0,1} -> V[2*p+k] = V[10], V[11].
    # C index: k in {0,1}, i in {0,1} -> W[k*n+i].
    
    # k=0
    U[2][col] += 1; U[5][col] += 1
    V[10][col] += 1
    W[0][col] += 1; W[2][col] += 1 # C11 (k=0,i=0), C21 (k=1,i=0)? No.
    # W[k*n+i]. k=0 -> i=0,1. W[0], W[1].
    # So C11, C12? No, i is row of A.
    # C[k, i].
    # k=0, i=0 -> W[0]. k=0, i=1 -> W[1].
    W[0][col] += 1; W[1][col] += 1
    col += 1
    
    # k=1
    U[2][col] += 1; U[5][col] += 1
    V[11][col] += 1
    W[2][col] += 1; W[3][col] += 1 # k=1, i=0 -> W[2]. k=1, i=1 -> W[3].
    col += 1
    
    # Part 2: k in {2, 3, 4}
    # Sub-part 2a: j in {0, 1}, k in {2, 3, 4} -> Strassen (on 2x2 A, 2x3 B)
    # This is not exactly Strassen. Strassen is for square.
    # We can use 3 separate Strassen-like? No.
    # We can use the fact that 2x2 by 2x3 is rank 6?
    # Or just use schoolbook for the 2x2 by 2x3 part? Rank 4?
    # 2x2 A, 2x3 B. 4 mults?
    # A = [[a, b], [c, d]]. B = [[e, f, g], [h, i, j]].
    # C = [[ae+ch, af+ci, ag+cj], [de+dh, df+di, dg+dj]].
    # This is 4 multiplications if we reuse?
    # No, it's 6 mults schoolbook.
    # Can we do it with 5?
    # Let's use schoolbook for this part to be safe.
    # j in {0,1}, k in {2,3,4}.
    # A: U[0..3]. B: V[0,1, 5,6, 10,11].
    
    # k=2
    U[0][col] += 1; U[1][col] += 1; U[3][col] += 1; U[4][col] += 1
    V[0][col] += 1; V[5][col] += 1
    W[4][col] += 1; W[5][col] += 1; W[6][col] += 1; W[7][col] += 1
    col += 1
    
    # k=3
    U[0][col] += 1; U[1][col] += 1; U[3][col] += 1; U[4][col] += 1
    V[1][col] += 1; V[6][col] += 1
    W[8][col] += 1; W[9][col] += 1; W[10][col] += 1; W[11][col] += 1
    col += 1
    
    # k=4
    U[0][col] += 1; U[1][col] += 1; U[3][col] += 1; U[4][col] += 1
    V[2][col] += 1; V[7][col] += 1
    W[12][col] += 1; W[13][col] += 1; W[14][col] += 1; W[15][col] += 1
    col += 1
    
    # Sub-part 2b: j=2, k in {2, 3, 4} -> Schoolbook
    # A: U[2], U[5]. B: V[10], V[11], V[12].
    
    # k=2
    U[2][col] += 1; U[5][col] += 1
    V[10][col] += 1
    W[4][col] += 1; W[6][col] += 1
    col += 1
    
    # k=3
    U[2][col] += 1; U[5][col] += 1
    V[11][col] += 1
    W[8][col] += 1; W[10][col] += 1
    col += 1
    
    # k=4
    U[2][col] += 1; U[5][col] += 1
    V[12][col] += 1
    W[12][col] += 1; W[14][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
