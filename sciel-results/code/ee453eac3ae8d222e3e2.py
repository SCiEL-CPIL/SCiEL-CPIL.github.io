"""Matrix multiplication 4x5x6 with rank-reduced decomposition."""
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

def _strassen_2x2():
    """Return (U, V, W) for 2x2x2 Strassen, shapes (4,7), (4,7), (4,7)."""
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # Strassen: M1=(A11+A22)(B11+B22), M2=(A21+A22)B11, M3=A11(B12-B22),
    #           M4=A22(B21-B11), M5=(A11+A12)B22, M6=(A21-A11)(B11+B12),
    #           M7=(A12-A22)(B21+B22)
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # Row indices: U: i*2+j, V: j*2+k, W: k*2+i
    # A11=U[0], A12=U[1], A21=U[2], A22=U[3]
    # B11=V[0], B12=V[1], B21=V[2], B22=V[3]
    # C11=W[0], C12=W[1], C21=W[2], C22=W[3]
    
    # M1 = (A11+A22)(B11+B22) -> C11+=M1, C22+=M1
    U[0][0]=1; U[3][0]=1  # A11, A22
    V[0][0]=1; V[3][0]=1  # B11, B22
    W[0][0]=1; W[3][0]=1  # C11, C22
    
    # M2 = (A21+A22)B11 -> C21+=M2
    U[2][1]=1; U[3][1]=1  # A21, A22
    V[0][1]=1            # B11
    W[2][1]=1            # C21
    
    # M3 = A11(B12-B22) -> C12+=M3
    U[0][2]=1            # A11
    V[1][2]=1; V[3][2]=-1  # B12, -B22
    W[1][2]=1            # C12
    
    # M4 = A22(B21-B11) -> C22+=M4
    U[3][3]=1            # A22
    V[2][3]=1; V[0][3]=-1  # B21, -B11
    W[3][3]=1            # C22
    
    # M5 = (A11+A12)B22 -> C12+=M5
    U[0][4]=1; U[1][4]=1  # A11, A12
    V[3][4]=1            # B22
    W[1][4]=1            # C12
    
    # M6 = (A21-A11)(B11+B12) -> C21+=M6
    U[2][5]=1; U[0][5]=-1  # A21, -A11
    V[0][5]=1; V[1][5]=1  # B11, B12
    W[2][5]=1            # C21
    
    # M7 = (A12-A22)(B21+B22) -> C11+=M7
    U[1][6]=1; U[3][6]=-1  # A12, -A22
    V[2][6]=1; V[3][6]=1  # B21, B22
    W[0][6]=1            # C11
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    rank = 0
    U_list = []
    V_list = []
    W_list = []
    
    # Block 1: 4x4x4 using Kronecker Strassen (rank 49)
    U_s, V_s, W_s = _strassen_2x2()
    U_s = np.array(U_s, dtype=int)
    V_s = np.array(V_s, dtype=int)
    W_s = np.array(W_s, dtype=int)
    
    # Kronecker: 4x4x4 from two 2x2x2 blocks
    U_44 = np.kron(U_s, U_s)  # shape (16, 49)
    V_44 = np.kron(V_s, V_s)  # shape (16, 49)
    W_44 = np.kron(W_s, W_s)  # shape (16, 49)
    
    rank_44 = 49
    # Map 4x4 block indices to full 4x5x6 indices
    # U row: i*5+j where i,j in 0..3 (from 4x4 sub-tensor)
    # V row: j*6+k where j,k in 0..3
    # W row: k*4+i where k,i in 0..3
    
    # Build mapping arrays
    # For U: local index (i,j) -> global i*5+j
    u_rows_44 = []
    v_rows_44 = []
    w_rows_44 = []
    for i in range(4):
        for j in range(4):
            u_rows_44.append(i*5 + j)
    for j in range(4):
        for k in range(4):
            v_rows_44.append(j*6 + k)
    for k in range(4):
        for i in range(4):
            w_rows_44.append(k*4 + i)
    
    # Create full matrices with zeros
    total_rank = rank_44
    # We'll build column by column
    U_full = [[0] * total_rank for _ in range(n*m)]
    V_full = [[0] * total_rank for _ in range(m*p)]
    W_full = [[0] * total_rank for _ in range(p*n)]
    
    col = 0
    for c in range(rank_44):
        for local_r, gr in zip(range(16), u_rows_44):
            if U_44[local_r, c] != 0:
                U_full[gr][col] = int(U_44[local_r, c])
        for local_r, gr in zip(range(16), v_rows_44):
            if V_44[local_r, c] != 0:
                V_full[gr][col] = int(V_44[local_r, c])
        for local_r, gr in zip(range(16), w_rows_44):
            if W_44[local_r, c] != 0:
                W_full[gr][col] = int(W_44[local_r, c])
        col += 1
    
    # Block 2: 4x1x4 (i in 0..3, j=4, k in 0..3) - schoolbook rank 16
    rank_414 = 16
    col = rank_44
    for i in range(4):
        for k in range(4):
            U_full[i*5 + 4][col] = 1
            V_full[4*6 + k][col] = 1
            W_full[k*4 + i][col] = 1
            col += 1
    total_rank += rank_414
    
    # Block 3: 4x4x2 (i in 0..3, j in 0..3, k in 4..5) - schoolbook rank 32
    rank_442 = 32
    col = rank_44 + rank_414
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                U_full[i*5 + j][col] = 1
                V_full[j*6 + k][col] = 1
                W_full[k*4 + i][col] = 1
                col += 1
    total_rank += rank_442
    
    # Block 4: 4x1x2 (i in 0..3, j=4, k in 4..5) - schoolbook rank 8
    rank_412 = 8
    col = rank_44 + rank_414 + rank_442
    for i in range(4):
        for k in range(4, 6):
            U_full[i*5 + 4][col] = 1
            V_full[4*6 + k][col] = 1
            W_full[k*4 + i][col] = 1
            col += 1
    total_rank += rank_412
    
    return {"U": U_full, "V": V_full, "W": W_full}
