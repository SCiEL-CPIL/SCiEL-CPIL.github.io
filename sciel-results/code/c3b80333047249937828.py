"""Matrix multiplication 2x4x5 with structured decomposition."""
DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # We'll build the decomposition as a direct sum of blocks:
    # Split m=4 into m1=2, m2=2
    # Split p=5 into p1=2, p2=3
    # Blocks: (n x m1 x p1), (n x m2 x p1), (n x m1 x p2), (n x m2 x p2)
    # = (2x2x2), (2x2x2), (2x2x3), (2x2x3)
    
    # Strassen decomposition for 2x2x2 (rank 7)
    # For A(2x2) * B(2x2) = C(2x2):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    def strassen_2x2x2():
        """Return (U, V, W) for 2x2x2 Strassen, rank 7.
        U: 4x7, V: 4x7, W: 4x7
        A indexed as [i*2+j], B as [j*2+k], C as [k*2+i]
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # A indices: 00=0, 01=1, 10=2, 11=3
        # B indices: 00=0, 01=1, 10=2, 11=3
        # C indices: 00=0, 01=1, 10=2, 11=3 (C[k*2+i])
        
        # M1 = (A11+A22)(B11+B22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1  # C11 += M1, C22 += M1
        
        # M2 = (A21+A22)B11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1  # C21 += M2
        
        # M3 = A11(B21-B22)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[2][2] = 1  # C12 += M3, C21 += M3
        
        # M4 = A22(B11-B21)
        U[3][3] = 1
        V[0][3] = 1; V[2][3] = -1
        W[0][3] = 1; W[2][3] = 1  # C11 += M4, C21 += M4
        
        # M5 = (A11+A12)B22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1  # C11 -= M5, C12 += M5
        
        # M6 = (A21-A11)(B11+B12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1  # C22 += M6
        
        # M7 = (A12-A22)(B21+B22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = -1  # C11 += M7, C22 -= M7
        
        return U, V, W
    
    def mm_2x2x3():
        """Return (U, V, W) for 2x2x3, rank 11.
        U: 4x11, V: 6x11, W: 6x11
        A indexed as [i*2+j] (4 entries), B as [j*3+k] (6 entries), C as [k*2+i] (6 entries)
        """
        # Use Strassen for k=0,1 (2x2x2 part) + 4 outer products for k=2
        U, V, W = strassen_2x2x2()
        rank_strassen = 7
        rank_total = 11
        
        U_ext = [[0]*rank_total for _ in range(4)]
        V_ext = [[0]*rank_total for _ in range(6)]
        W_ext = [[0]*rank_total for _ in range(6)]
        
        # Copy Strassen part (for k=0,1)
        for i in range(4):
            for c in range(rank_strassen):
                U_ext[i][c] = U[i][c]
        for j in range(4):  # j is inner index 0,1; but B has 6 entries
            for c in range(rank_strassen):
                V_ext[j][c] = V[j][c]
        for k in range(4):  # k is output index 0,1; but W has 6 entries
            for c in range(rank_strassen):
                W_ext[k][c] = W[k][c]
        
        # For k=2: C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2]
        # B[0,2] is index 0*3+2=2, B[1,2] is index 1*3+2=5
        # C[0,2] is index 2*2+0=4, C[1,2] is index 2*2+1=5
        
        # Term 1: A[0,0]*B[0,2] -> C[0,2]
        U_ext[0][7] = 1
        V_ext[2][7] = 1
        W_ext[4][7] = 1
        
        # Term 2: A[0,1]*B[1,2] -> C[0,2]
        U_ext[1][8] = 1
        V_ext[5][8] = 1
        W_ext[4][8] = 1
        
        # Term 3: A[1,0]*B[0,2] -> C[1,2]
        U_ext[2][9] = 1
        V_ext[2][9] = 1
        W_ext[5][9] = 1
        
        # Term 4: A[1,1]*B[1,2] -> C[1,2]
        U_ext[3][10] = 1
        V_ext[5][10] = 1
        W_ext[5][10] = 1
        
        return U_ext, V_ext, W_ext
    
    # Now assemble the full 2x4x5 decomposition
    # A is 2x4, indexed as [i*4+j], i=0,1; j=0,1,2,3
    # B is 4x5, indexed as [j*5+k], j=0,1,2,3; k=0,1,2,3,4
    # C is 5x2, indexed as [k*2+i], k=0,1,2,3,4; i=0,1
    
    # Block (i in {0,1}, j in {0,1}, k in {0,1}): 2x2x2 Strassen
    # A indices: i*4+j for i=0,1; j=0,1 -> 0,1,4,5
    # B indices: j*5+k for j=0,1; k=0,1 -> 0,1,5,6
    # C indices: k*2+i for k=0,1; i=0,1 -> 0,1,2,3
    
    # Block (i in {0,1}, j in {2,3}, k in {0,1}): 2x2x2 Strassen
    # A indices: i*4+j for i=0,1; j=2,3 -> 2,3,6,7
    # B indices: j*5+k for j=2,3; k=0,1 -> 10,11,15,16
    # C indices: k*2+i for k=0,1; i=0,1 -> 0,1,2,3
    
    # Block (i in {0,1}, j in {0,1}, k in {2,3,4}): 2x2x3
    # A indices: i*4+j for i=0,1; j=0,1 -> 0,1,4,5
    # B indices: j*5+k for j=0,1; k=2,3,4 -> 2,3,4,7,8,9
    # C indices: k*2+i for k=2,3,4; i=0,1 -> 4,5,6,7,8,9
    
    # Block (i in {0,1}, j in {2,3}, k in {2,3,4}): 2x2x3
    # A indices: i*4+j for i=0,1; j=2,3 -> 2,3,6,7
    # B indices: j*5+k for j=2,3; k=2,3,4 -> 12,13,14,17,18,19
    # C indices: k*2+i for k=2,3,4; i=0,1 -> 4,5,6,7,8,9
    
    U1, V1, W1 = strassen_2x2x2()  # rank 7
    U2, V2, W2 = strassen_2x2x2()  # rank 7
    U3, V3, W3 = mm_2x2x3()        # rank 11
    U4, V4, W4 = mm_2x2x3()        # rank 11
    
    total_rank = 7 + 7 + 11 + 11  # = 36
    
    U_full = [[0] * total_rank for _ in range(8)]   # 2*4 = 8
    V_full = [[0] * total_rank for _ in range(20)]  # 4*5 = 20
    W_full = [[0] * total_rank for _ in range(10)]  # 5*2 = 10
    
    # Block 1: j in {0,1}, k in {0,1}, Strassen
    offset = 0
    # A mapping: local [i*2+j_local] -> global [i*4+j_global], j_global = j_local
    a_map1 = [0, 1, 4, 5]  # local 0,1,2,3 -> global
    # B mapping: local [j_local*2+k_local] -> global [j_global*5+k_global], j_global=2+j_local, k_global=k_local
    b_map1 = [0, 1, 5, 6]  # local 0,1,2,3 -> global
    # W mapping: local [k_local*2+i] -> global [k_global*2+i], k_global=k_local
    w_map1 = [0, 1, 2, 3]
    
    for li in range(4):
        for c in range(7):
            U_full[a_map1[li]][offset + c] = U1[li][c]
    for lj in range(4):
        for c in range(7):
            V_full[b_map1[lj]][offset + c] = V1[lj][c]
    for lw in range(4):
        for c in range(7):
            W_full[w_map1[lw]][offset + c] = W1[lw][c]
    
    # Block 2: j in {2,3}, k in {0,1}, Strassen
    offset = 7
    a_map2 = [2, 3, 6, 7]
    b_map2 = [10, 11, 15, 16]
    w_map2 = [0, 1, 2, 3]
    
    for li in range(4):
        for c in range(7):
            U_full[a_map2[li]][offset + c] = U2[li][c]
    for lj in range(4):
        for c in range(7):
            V_full[b_map2[lj]][offset + c] = V2[lj][c]
    for lw in range(4):
        for c in range(7):
            W_full[w_map2[lw]][offset + c] = W2[lw][c]
    
    # Block 3: j in {0,1}, k in {2,3,4}, 2x2x3
    offset = 14
    a_map3 = [0, 1, 4, 5]
    # B local [j_local*3+k_local] -> global [j_global*5+k_global], j_global=j_local, k_global=2+k_local
    b_map3 = [2, 3, 4, 7, 8, 9]
    # W local [k_local*2+i] -> global [k_global*2+i], k_global=2+k_local
    w_map3 = [4, 5, 6, 7, 8, 9]
    
    for li in range(4):
        for c in range(11):
            U_full[a_map3[li]][offset + c] = U3[li][c]
    for lj in range(6):
        for c in range(11):
            V_full[b_map3[lj]][offset + c] = V3[lj][c]
    for lw in range(6):
        for c in range(11):
            W_full[w_map3[lw]][offset + c] = W3[lw][c]
    
    # Block 4: j in {2,3}, k in {2,3,4}, 2x2x3
    offset = 25
    a_map4 = [2, 3, 6, 7]
    b_map4 = [12, 13, 14, 17, 18, 19]
    w_map4 = [4, 5, 6, 7, 8, 9]
    
    for li in range(4):
        for c in range(11):
            U_full[a_map4[li]][offset + c] = U4[li][c]
    for lj in range(6):
        for c in range(11):
            V_full[b_map4[lj]][offset + c] = V4[lj][c]
    for lw in range(6):
        for c in range(11):
            W_full[w_map4[lw]][offset + c] = W4[lw][c]
    
    return {"U": U_full, "V": V_full, "W": W_full}
