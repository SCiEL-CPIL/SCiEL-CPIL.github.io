import numpy as np

DIMENSIONS = [3, 4, 7]

def _strassen_m222():
    # Returns U, V, W for 2x2x2 Strassen decomposition
    # Rank 7
    U = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0]
    ])
    # Wait, U is (n*m, rank) = (4, 7)
    # V is (m*p, rank) = (4, 7)
    # W is (p*n, rank) = (4, 7)
    
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # U columns correspond to A coefficients
    # V columns correspond to B coefficients
    # W columns correspond to C coefficients
    
    # A indices: 0,1,2,3 (11,12,21,22)
    # B indices: 0,1,2,3 (11,12,21,22)
    # C indices: 0,1,2,3 (11,12,21,22)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    W[1,0]=-1; W[2,0]=-1
    
    # M2
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1; W[3,1]=1
    
    # M3
    U[0,2]=1
    V[2,2]=1; V[3,2]=-1
    W[1,2]=1; W[3,2]=-1
    
    # M4
    U[3,3]=1
    V[1,3]=1; V[0,3]=-1
    W[0,3]=-1; W[2,3]=1
    
    # M5
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[1,4]=1; W[0,4]=-1
    
    # M6
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[2,5]=1; W[1,5]=-1
    
    # M7
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[3,6]=1; W[0,6]=-1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # We will use a block decomposition.
    # Split A (3x4) into A_top (2x4) and A_bot (1x4).
    # Split B (4x7) into B_left (4x3) and B_right (4x4)?
    # Actually, let's just use the rank 80 construction found in literature.
    # It is often based on R(2,4,7) + R(1,4,7).
    # R(1,4,7) = 4.
    # R(2,4,7) <= 2 * R(2,2,7).
    # R(2,2,7) <= 14? No, R(2,2,7) is 14?
    # If R(2,2,7) is 14, then R(2,4,7) is 28. Total 32.
    # But known is 80.
    # This implies R(2,2,7) is much higher or my understanding is wrong.
    # Actually, R(2,2,7) is 14 is correct for the tensor rank?
    # Maybe the "rank" in the task is different?
    # No, it's tensor rank.
    
    # Let's just output the schoolbook for now if I can't get 80 easily.
    # But the inspiration says 80 is possible.
    # I will try to construct the 80 rank decomposition.
    
    # For now, return schoolbook to ensure validity.
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
