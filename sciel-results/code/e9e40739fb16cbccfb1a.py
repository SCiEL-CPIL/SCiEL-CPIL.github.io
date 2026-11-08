import numpy as np

def construct():
    n, m, p = 2, 4, 5
    # We split m=4 into m1=2 and m2=2.
    # A is 2x4, B is 4x5.
    # A = [A1, A2] where A1, A2 are 2x2.
    # B = [B1; B2] where B1, B2 are 2x5.
    # C = A1 B1 + A2 B2.
    
    # We need a decomposition for 2x2 x 2x5 with rank 19.
    # Split B (2x5) into B_top (2x2) and B_bot (2x3).
    # A B_top uses Strassen (rank 7).
    # A B_bot uses Schoolbook (rank 12).
    # Total rank 19.
    
    # Total rank for 2x4 x 5 is 19 + 19 = 38.
    
    rank = 38
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a decomposition block
    # u_idx: indices in U (2x2 block of A)
    # v_idx: indices in V (2x5 block of B)
    # w_idx: indices in W (2x5 block of C)
    
    col = 0
    
    # Block 1: A1 (2x2) * B1 (2x5)
    # A1 indices in U: 0, 1, 2, 3 (row-major 2x2)
    # B1 indices in V: 0..9 (2x5)
    # C1 indices in W: 0..9 (2x5)
    
    # Strassen for 2x2 x 2x2 (part of B1)
    # B1 is 2x2, B2 is 2x3.
    # B1 columns 0,1. B2 columns 2,3,4.
    
    # Strassen factors for A1 * B1_top (2x2)
    # U indices for A1: 0,1,2,3
    # V indices for B1_top: 0,1,4,5 (cols 0,1 of B1)
    # W indices for C1_top: 0,1,4,5 (cols 0,1 of C1)
    
    # Strassen 7 multiplications
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # A1 = [[a00, a01], [a10, a11]]
    # U indices: 0=a00, 1=a01, 2=a10, 3=a11
    
    # B1_top = [[b00, b01], [b10, b11]]
    # V indices: 0=b00, 1=b01, 4=b10, 5=b11
    
    # C1_top = [[c00, c01], [c10, c11]]
    # W indices: 0=c00, 1=c01, 4=c10, 5=c11
    
    # M1
    U[0][col] += 1; U[3][col] += 1
    V[0][col] += 1; V[5][col] += 1
    W[0][col] += 1; W[5][col] += 1
    col += 1
    
    # M2
    U[2][col] += 1; U[3][col] += 1
    V[0][col] += 1
    W[4][col] += 1; W[5][col] += 1
    col += 1
    
    # M3
    U[0][col] += 1
    V[1][col] += 1; V[5][col] -= 1
    W[1][col] += 1
    col += 1
    
    # M4
    U[3][col] += 1
    V[4][col] += 1; V[0][col] -= 1
    W[4][col] += 1; W[5][col] -= 1
    col += 1
    
    # M5
    U[0][col] += 1; U[1][col] += 1
    V[5][col] += 1
    W[0][col] -= 1; W[5][col] += 1
    col += 1
    
    # M6
    U[2][col] += 1; U[0][col] -= 1
    V[0][col] += 1; V[1][col] += 1
    W[5][col] += 1
    col += 1
    
    # M7
    U[1][col] += 1; U[3][col] -= 1
    V[4][col] += 1; V[5][col] += 1
    W[0][col] += 1; W[5][col] -= 1
    col += 1
    
    # Schoolbook for A1 * B1_bot (2x3)
    # B1_bot columns 2,3,4. V indices: 2,3,6,7,8,9
    # C1_bot columns 2,3,4. W indices: 2,3,6,7,8,9
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i*2+j][col] += 1
                V[j*5 + (2+k)][col] += 1
                W[(2+k)*2 + i][col] += 1
                col += 1
                
    # Block 2: A2 (2x2) * B2 (2x5)
    # A2 indices in U: 4, 5, 6, 7
    # B2 indices in V: 10..19
    # C2 indices in W: 10..19
    
    # Strassen for A2 * B2_top (2x2)
    # B2_top columns 0,1. V indices: 10,11,14,15
    # C2_top columns 0,1. W indices: 10,11,14,15
    
    # M1
    U[4][col] += 1; U[7][col] += 1
    V[10][col] += 1; V[15][col] += 1
    W[10][col] += 1; W[15][col] += 1
    col += 1
    
    # M2
    U[6][col] += 1; U[7][col] += 1
    V[10][col] += 1
    W[14][col] += 1; W[15][col] += 1
    col += 1
    
    # M3
    U[4][col] += 1
    V[11][col] += 1; V[15][col] -= 1
    W[11][col] += 1
    col += 1
    
    # M4
    U[7][col] += 1
    V[14][col] += 1; V[10][col] -= 1
    W[14][col] += 1; W[15][col] -= 1
    col += 1
    
    # M5
    U[4][col] += 1; U[5][col] += 1
    V[15][col] += 1
    W[10][col] -= 1; W[15][col] += 1
    col += 1
    
    # M6
    U[6][col] += 1; U[4][col] -= 1
    V[10][col] += 1; V[11][col] += 1
    W[15][col] += 1
    col += 1
    
    # M7
    U[5][col] += 1; U[7][col] -= 1
    V[14][col] += 1; V[15][col] += 1
    W[10][col] += 1; W[15][col] -= 1
    col += 1
    
    # Schoolbook for A2 * B2_bot (2x3)
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[4+i*2+j][col] += 1
                V[10+j*5 + (2+k)][col] += 1
                W[10+(2+k)*2 + i][col] += 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
