import numpy as np

def construct():
    n, m, p = 2, 4, 6
    rank = 42
    
    # Initialize matrices with zeros
    # U: (n*m, rank) = (8, 42)
    # V: (m*p, rank) = (24, 42)
    # W: (p*n, rank) = (12, 42)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # C = [[c11, c12], [c21, c22]]
    
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
    
    def add_strassen(a, b, c, d, e, f, g, h, c11, c12, c21, c22, col):
        # M1
        U[a][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        V[h][col] += 1
        W[c11][col] += 1
        W[c22][col] += 1
        col += 1
        
        # M2
        U[c][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        W[c21][col] += 1
        W[c22][col] -= 1
        col += 1
        
        # M3
        U[a][col] += 1
        V[f][col] += 1
        V[h][col] -= 1
        W[c12][col] += 1
        W[c22][col] += 1
        col += 1
        
        # M4
        U[d][col] += 1
        V[g][col] += 1
        V[e][col] -= 1
        W[c11][col] += 1
        W[c21][col] += 1
        col += 1
        
        # M5
        U[a][col] += 1
        U[b][col] += 1
        V[h][col] += 1
        W[c11][col] -= 1
        W[c12][col] += 1
        col += 1
        
        # M6
        U[c][col] += 1
        U[a][col] -= 1
        V[e][col] += 1
        V[f][col] += 1
        W[c22][col] += 1
        col += 1
        
        # M7
        U[b][col] += 1
        U[d][col] -= 1
        V[g][col] += 1
        V[h][col] += 1
        W[c11][col] += 1
        col += 1
        
        return col

    col = 0
    
    # A is 2x4. B is 4x6. C is 2x6.
    # We split A into A_left (cols 0,1) and A_right (cols 2,3).
    # We split B into B_top (rows 0,1) and B_bottom (rows 2,3).
    # C = A_left * B_top + A_right * B_bottom
    
    # Indices for U (A):
    # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
    # A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
    
    # Indices for V (B):
    # B[i,j] = i*6 + j
    
    # Indices for W (C):
    # C[i,j] = j*2 + i  (Column-major? No, task says W[k*n+i, r] for C[i,k]?)
    # Task: "output factor follows the official trace convention W[k*n+i,r]"
    # This usually means C[i, k] is stored at index k*n + i.
    # Let's verify: C is n x p = 2 x 6.
    # Index in W: k*p + i? Or k*n + i?
    # "W[k*n+i,r]" where k is col index (0..5), i is row index (0..1).
    # So C[i, k] -> W[k*2 + i].
    
    # Let's re-verify the current code's W indexing.
    # Current: C_idx = [2*k*2 + 0, (2*k+1)*2 + 0, 2*k*2 + 1, (2*k+1)*2 + 1]
    # c11 = C[0, 2k] -> index (2k)*2 + 0. Correct.
    # c12 = C[0, 2k+1] -> index (2k+1)*2 + 0. Correct.
    # c21 = C[1, 2k] -> index (2k)*2 + 1. Correct.
    # c22 = C[1, 2k+1] -> index (2k+1)*2 + 1. Correct.
    
    for k in range(3):
        # B_top: rows 0,1. Cols 2k, 2k+1.
        # B[0, 2k] = 2k
        # B[0, 2k+1] = 2k+1
        # B[1, 2k] = 6 + 2k
        # B[1, 2k+1] = 6 + 2k+1
        
        # A_left: cols 0,1.
        # a=0, b=1, c=4, d=5
        
        # C cols 2k, 2k+1.
        # c11 = C[0, 2k] = 2k*2 + 0
        # c12 = C[0, 2k+1] = (2k+1)*2 + 0
        # c21 = C[1, 2k] = 2k*2 + 1
        # c22 = C[1, 2k+1] = (2k+1)*2 + 1
        
        col = add_strassen(
            0, 1, 4, 5,
            2*k, 2*k+1, 6+2*k, 6+2*k+1,
            2*k*2, (2*k+1)*2, 2*k*2+1, (2*k+1)*2+1,
            col
        )
        
        # B_bottom: rows 2,3. Cols 2k, 2k+1.
        # B[2, 2k] = 12 + 2k
        # B[2, 2k+1] = 12 + 2k+1
        # B[3, 2k] = 18 + 2k
        # B[3, 2k+1] = 18 + 2k+1
        
        # A_right: cols 2,3.
        # a=2, b=3, c=6, d=7
        
        # C cols 2k, 2k+1. Same as above.
        
        col = add_strassen(
            2, 3, 6, 7,
            12+2*k, 12+2*k+1, 18+2*k, 18+2*k+1,
            2*k*2, (2*k+1)*2, 2*k*2+1, (2*k+1)*2+1,
            col
        )
        
    return {"U": U, "V": V, "W": W}
