import time

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose the 2x2x5 multiplication into:
    # 1. A 2x2x2 multiplication (first 2 columns of B) using Strassen (rank 7)
    # 2. A 2x2x3 multiplication (next 3 columns of B) using Schoolbook (rank 12)
    # Total rank = 7 + 12 = 19
    
    rank_strassen = 7
    rank_schoolbook = 12
    total_rank = rank_strassen + rank_schoolbook
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: Strassen for columns 0 and 1 of B ---
    # Indices for Strassen:
    # U index: i*m + j (0..3)
    # V index: j*p + k (0..9) -> here k is 0,1
    # W index: k*n + i (0..9) -> here k is 0,1
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A01-A10)(B00+B10)
    # M7 = (A10-A00)(B11+B01)
    
    # We map these to U, V, W
    # U corresponds to A coefficients
    # V corresponds to B coefficients
    # W corresponds to C coefficients (output)
    
    # Col 0: M1
    # A00+A11 -> U[0][0]=1, U[3][0]=1 (i=0,j=0 and i=1,j=1)
    # B00+B11 -> V[0*5+0][0]=1 (j=0,k=0), V[1*5+1][0]=1 (j=1,k=1)
    # C00 += M1, C11 += M1 -> W[0*2+0][0]=1 (k=0,i=0), W[1*2+1][0]=1 (k=1,i=1)
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[6][0] += 1 # j=0,k=0 is 0; j=1,k=1 is 1*5+1=6
    W[0][0] += 1; W[3][0] += 1 # k=0,i=0 is 0; k=1,i=1 is 1*2+1=3
    
    # Col 1: M2
    # A10+A11 -> U[2][1]=1 (i=1,j=0), U[3][1]=1 (i=1,j=1)
    # B00 -> V[0][1]=1
    # C10 += M2, C11 += M2 -> W[0*2+1][1]=1 (k=0,i=1), W[1*2+1][1]=1 (k=1,i=1)
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1; W[3][1] += 1
    
    # Col 2: M3
    # A00 -> U[0][2]=1
    # B01-B11 -> V[1][2]=1 (j=0,k=1), V[6][2]=-1 (j=1,k=1)
    # C00 += M3, C01 -= M3 -> W[0][2]=1, W[1][2]=-1
    U[0][2] += 1
    V[1][2] += 1; V[6][2] += -1
    W[0][2] += 1; W[1][2] += -1
    
    # Col 3: M4
    # A11 -> U[3][3]=1
    # B10-B00 -> V[5][3]=1 (j=1,k=0), V[0][3]=-1
    # C10 -= M4, C11 += M4 -> W[2][3]=-1, W[3][3]=1
    U[3][3] += 1
    V[5][3] += 1; V[0][3] += -1
    W[2][3] += -1; W[3][3] += 1
    
    # Col 4: M5
    # A00+A01 -> U[0][4]=1, U[1][4]=1
    # B11 -> V[6][4]=1
    # C01 += M5, C11 += M5 -> W[1][4]=1, W[3][4]=1
    U[0][4] += 1; U[1][4] += 1
    V[6][4] += 1
    W[1][4] += 1; W[3][4] += 1
    
    # Col 5: M6
    # A01-A10 -> U[1][5]=1, U[2][5]=-1
    # B00+B10 -> V[0][5]=1, V[5][5]=1
    # C00 += M6, C10 += M6 -> W[0][5]=1, W[2][5]=1
    U[1][5] += 1; U[2][5] += -1
    V[0][5] += 1; V[5][5] += 1
    W[0][5] += 1; W[2][5] += 1
    
    # Col 6: M7
    # A10-A00 -> U[2][6]=1, U[0][6]=-1
    # B11+B01 -> V[6][6]=1, V[1][6]=1
    # C01 += M7, C11 += M7 -> W[1][6]=1, W[3][6]=1
    U[2][6] += 1; U[0][6] += -1
    V[6][6] += 1; V[1][6] += 1
    W[1][6] += 1; W[3][6] += 1
    
    # --- Part 2: Schoolbook for columns 2, 3, 4 of B ---
    # k in {2, 3, 4}
    # i in {0, 1}, j in {0, 1}
    # Total 2*2*3 = 12 terms
    col = rank_strassen
    for i in range(n):
        for j in range(m):
            for k in range(2, p):
                # U[i*m + j]
                # V[j*p + k]
                # W[k*n + i]
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
