import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=15
    # We want to decompose the tensor for multiplying a 2x2 matrix by a 2x15 matrix.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We use a sliding window approach with overlapping 3-column windows.
    # Each window covers 3 consecutive columns and uses a rank-10 decomposition for M_{2,2,3}.
    # With step 2, we have windows starting at 0, 2, 4, ..., 12 (7 windows).
    # Total rank: 7 * 10 = 70. This is worse than 60.
    
    # Instead, let's use a different approach: split into 2-column blocks and 1-column block.
    # 15 = 7*2 + 1
    # For each 2x2 * 2x2 block, use Strassen (rank 7).
    # For the 2x2 * 2x1 block, use rank 2.
    # Total: 7*7 + 2 = 51.
    
    # Let me carefully implement Strassen for 2x2 * 2x2.
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = A*B = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    
    # Correct Strassen formulas (verified):
    # M1 = (a + d)(e + h)
    # M2 = (c + d)e
    # M3 = a(f - h)
    # M4 = d(g - e)
    # M5 = (a + c)h
    # M6 = (d - b)(e + f)
    # M7 = (b - c)(g + h)
    # C00 = M1 + M3 - M4 - M6
    # C01 = M4 + M5
    # C10 = M2 + M6
    # C11 = M1 - M3 + M5 - M7
    
    # Let me verify with a=1,b=0,c=0,d=1,e=1,f=2,g=3,h=4:
    # M1 = (1+1)(1+4) = 10
    # M2 = (0+1)*1 = 1
    # M3 = 1*(2-4) = -2
    # M4 = 1*(3-1) = 2
    # M5 = (1+0)*4 = 4
    # M6 = (1-0)(1+2) = 3
    # M7 = (0-0)(3+4) = 0
    # C00 = 10 + (-2) - 2 - 3 = 3. Should be 1*1 + 0*3 = 1. WRONG.
    
    # The formulas I've been using are incorrect. Let me find the correct ones.
    # From Wikipedia (Strassen algorithm):
    # M1 = (a + d)(e + h)
    # M2 = (c + d)(e + f)
    # M3 = (a + b)(g + h)
    # M4 = (a + c)(e + g)
    # M5 = (b + d)(f + h)
    # M6 = (a - c)(e + g)
    # M7 = (b - d)(f + h)
    # C00 = M4 + M5 - M7 - M2
    # C01 = M1 + M7 - M5 - M3
    # C10 = M2 + M3 - M1 - M4
    # C11 = M1 + M4 - M6 - M3
    
    # Verify: a=1,b=0,c=0,d=1,e=1,f=2,g=3,h=4
    # M1 = (1+1)(1+4) = 10
    # M2 = (0+1)(1+2) = 3
    # M3 = (1+0)(3+4) = 7
    # M4 = (1+0)(1+3) = 4
    # M5 = (0+1)(2+4) = 6
    # M6 = (1-0)(1+3) = 4
    # M7 = (0-1)(2+4) = -6
    # C00 = 4 + 6 - (-6) - 3 = 4 + 6 + 6 - 3 = 13. Should be 1. WRONG.
    
    # I'm clearly not getting the Strassen formulas right. Let me just use the schoolbook
    # decomposition with rank 60, which is known to be correct.
    
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
