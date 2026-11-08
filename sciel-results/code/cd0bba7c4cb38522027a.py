"""Matrix multiplication 2,4,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 5
    
    # We'll build the decomposition by splitting the 4 columns of B into two 2-column blocks.
    # Block 1: A * B_1 where B_1 is 4x2 (columns 0,1 of B) -> C_1 is 2x2
    # Block 2: A * B_2 where B_2 is 4x2 (columns 2,3 of B) -> C_2 is 2x2
    # Block 3: A * B_3 where B_3 is 4x1 (column 4 of B) -> C_3 is 2x1
    
    # For each 2x4x2 sub-problem, we can use Strassen on two 2x2x2 blocks.
    # But actually, 2x4x2 = (2x2) * (2x4). We can split the 2x4 matrix B into two 2x2 blocks.
    # C = A * [B_a | B_b] = [A*B_a | A*B_b]
    # Each A*B_a is 2x2x2, which can be done with rank 7 (Strassen).
    # So 2x4x2 has rank at most 14 (two Strassen blocks).
    
    # For 2x4x1: C = A * B where B is 4x1. This is just standard matrix multiplication.
    # We can view this as two 2x2x1 multiplications? No, B is 4x1, A is 2x4.
    # C_i = sum_j A[i,j] * B[j]
    # This is rank 8 (standard) or we can do better.
    # Actually, 2x4x1: we can split A into A_1 (2x2) and A_2 (2x2), and B into B_1 (2x1) and B_2 (2x1).
    # C = A_1*B_1 + A_2*B_2 where A_1*B_1 is 2x2x1 and A_2*B_2 is 2x2x1.
    # Each 2x2x1 has rank 4 (standard). So total rank 8.
    # But wait, can we do better? R(2,2,1) = 4 (it's just bilinear forms, need all 4 products).
    # So 2x4x1 has rank 8.
    
    # Total rank: 14 + 14 + 8 = 36.
    
    # Let's implement Strassen for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # The tensor for Strassen 2x2x2 has rank 7.
    # U matrix (4x7): rows indexed by (i,j) of A, columns by the 7 products
    # V matrix (4x7): rows indexed by (j,k) of B
    # W matrix (4x7): rows indexed by (k,i) of C (trace convention: W[k*n+i, r])
    
    # Let's index A rows as i*m+j = i*2+j for i in {0,1}, j in {0,1}
    # So A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B indices: j*p+k where j in {0,1} (since m=2 for sub-problem), k in {0,1}
    # Wait, in the sub-problem, B is 2x2, so j in {0,1}, k in {0,1}
    # B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C indices: k*n+i where k in {0,1}, i in {0,1}
    # C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    def strassen_2x2x2():
        """Return (U, V, W) for 2x2x2 Strassen decomposition, each of shape (4,7)."""
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A11+A22)(B11+B22)
        U[0][0] = 1  # A11
        U[3][0] = 1  # A22
        V[0][0] = 1  # B11
        V[3][0] = 1  # B22
        W[0][0] = 1  # C11 (k=0,i=0)
        W[3][0] = 1  # C22 (k=1,i=1)
        
        # M2 = (A21+A22)B11
        U[2][1] = 1  # A21
        U[3][1] = 1  # A22
        V[0][1] = 1  # B11
        W[2][1] = 1  # C21 (k=1,i=0)
        
        # M3 = A11(B21-B22)
        U[0][2] = 1  # A11
        V[2][2] = 1  # B21
        V[3][2] = -1 # B22
        W[1][2] = 1  # C12 (k=0,i=1)
        W[3][2] = 1  # C22 (k=1,i=1)
        
        # M4 = A22(B12-B11)
        U[3][3] = 1  # A22
        V[1][3] = 1  # B12
        V[0][3] = -1 # B11
        W[1][3] = 1  # C12 (k=0,i=1)
        W[2][3] = 1  # C21 (k=1,i=0)
        
        # M5 = (A11+A12)B22
        U[0][4] = 1  # A11
        U[1][4] = 1  # A12
        V[3][4] = 1  # B22
        W[0][4] = -1 # C11 (k=0,i=0)
        W[1][4] = 1  # C12 (k=0,i=1)
        
        # M6 = (A21-A11)(B11+B12)
        U[2][5] = 1  # A21
        U[0][5] = -1 # A11
        V[0][5] = 1  # B11
        V[1][5] = 1  # B12
        W[3][5] = 1  # C22 (k=1,i=1)
        
        # M7 = (A12-A22)(B21+B22)
        U[1][6] = 1  # A12
        U[3][6] = -1 # A22
        V[2][6] = 1  # B21
        V[3][6] = 1  # B22
        W[0][6] = 1  # C11 (k=0,i=0)
        W[1][6] = 1  # C12 (k=0,i=1)
        
        return U, V, W
    
    # For 2x2x1 (standard): C = A * B where A is 2x2, B is 2x1
    # C_i = sum_j A[i,j] B[j]
    # This is rank 4: each product A[i,j]*B[j] is a rank-1 tensor.
    def standard_2x2x1():
        """Return (U, V, W) for 2x2x1 standard decomposition, each of shape (4,4)."""
        U = [[0]*4 for _ in range(4)]
        V = [[0]*4 for _ in range(2)]
        W = [[0]*4 for _ in range(2)]
        
        # Wait, for 2x2x1: n=2, m=2, p=1
        # U shape: (n*m, rank) = (4, 4)
        # V shape: (m*p, rank) = (2, 4)
        # W shape: (p*n, rank) = (2, 4)
        # A indices: i*m+j, i in {0,1}, j in {0,1} -> 0,1,2,3
        # B indices: j*p+k, j in {0,1}, k in {0} -> 0,1
        # C indices: k*n+i, k in {0}, i in {0,1} -> 0,1
        
        col = 0
        for i in range(2):
            for j in range(2):
                U[i*2+j][col] = 1
                V[j*1+0][col] = 1  # j*1 + 0 = j
                W[0*2+i][col] = 1  # 0*2 + i = i
                col += 1
        
        return U, V, W
    
    # Now build the full 2x4x5 decomposition.
    # A is 2x4, B is 4x5, C is 2x5.
    # Split B into B_1 (4x2), B_2 (4x2), B_3 (4x1).
    # B_1 columns 0,1; B_2 columns 2,3; B_3 column 4.
    
    # For B_1 (4x2): split into B_1a (4x2) where we treat it as two 2x2 blocks vertically.
    # Actually, A is 2x4, B_1 is 4x2.
    # C_1 = A * B_1 is 2x2.
    # Split A into A_top (2x2, rows 0, cols 0,1) and A_bottom (2x2, rows 0, cols 2,3).
    # Split B_1 into B_1_top (2x2, rows 0,1) and B_1_bottom (2x2, rows 2,3).
    # C_1 = A_top * B_1_top + A_bottom * B_1_bottom
    # Each term is 2x2x2, use Strassen (rank 7 each).
    
    # Similarly for B_2 (4x2).
    # For B_3 (4x1): C_3 = A * B_3 is 2x1.
    # Split A into A_top (2x2) and A_bottom (2x2).
    # Split B_3 into B_3_top (2x1) and B_3_bottom (2x1).
    # C_3 = A_top * B_3_top + A_bottom * B_3_bottom
    # Each term is 2x2x1, use standard (rank 4 each).
    
    # Total rank: 7+7+7+7+4+4 = 36.
    
    sU, sV, sW = strassen_2x2x2()
    stdU, stdV, stdW = standard_2x2x1()
    
    # Full matrices
    n_full, m_full, p_full = 2, 4, 5
    rank_total = 36
    U_full = [[0]*rank_total for _ in range(n_full*m_full)]  # 8 x 36
    V_full = [[0]*rank_total for _ in range(m_full*p_full)]  # 20 x 36
    W_full = [[0]*rank_total for _ in range(p_full*n_full)]  # 10 x 36
    
    col_offset = 0
    
    # Block 1: A_top * B_1_top (Strassen)
    # A_top: rows 0, cols 0,1 -> A indices 0,1 (i=0, j=0,1)
    # A_bottom: rows 0, cols 2,3 -> A indices 2,3 (i=0, j=2,3) wait no.
    # A is 2x4. i in {0,1}, j in {0,1,2,3}.
    # A_top = A[:, 0:2] (cols 0,1), A_bottom = A[:, 2:4] (cols 2,3).
    # But for the multiplication C_1 = A * B_1, we need:
    # C_1[i,k] = sum_{j=0}^{3} A[i,j] * B_1[j,k]
    # = sum_{j=0}^{1} A[i,j] * B_1[j,k] + sum_{j=2}^{3} A[i,j] * B_1[j,k]
    # = (A_top * B_1_top)[i,k] + (A_bottom * B_1_bottom)[i,k]
    # where B_1_top = B_1[0:2, :] and B_1_bottom = B_1[2:4, :]
    
    # For A_top * B_1_top:
    # A_top is 2x2, B_1_top is 2x2.
    # A_top indices: i*4+j for i in {0}, j in {0,1} -> 0, 1
    # Wait, in the full problem, A[i,j] has index i*m_full+j = i*4+j.
    # So A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3, A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7.
    
    # For the sub-problem A_top (2x2), the local indices are:
    # local (i,j) -> global (i, j) -> global index i*4+j
    # B_1_top (2x2): local (j,k) where j is row index in B_1_top (0,1), k is col index in C_1 (0,1)
    # global B index: (j_global)*p_full + k_global where j_global = j_local (since B_1_top is rows 0,1 of B)
    # So B_1[j_local, k] has global index j_local*5 + k.
    # C_1 (2x2): local (i,k) -> global (i,k) -> global C index k*n_full + i = k*2 + i.
    
    # Let's map the Strassen U,V,W to global indices.
    # sU[local_a_idx][c] where local_a_idx = i*2+j for i in {0,1}, j in {0,1}
    # global_a_idx = i*4 + j
    # sV[local_b_idx][c] where local_b_idx = j*2+k for j in {0,1}, k in {0,1}
    # global_b_idx = j*5 + k
    # sW[local_c_idx][c] where local_c_idx = k*2+i for k in {0,1}, i in {0,1}
    # global_c_idx = k*2 + i
    
    # Block 1a: A_top * B_1_top
    for c in range(7):
        for la in range(4):
            # la = i*2+j, i = la//2, j = la%2
            i = la // 2
            j = la % 2
            ga = i * 4 + j
            U_full[ga][col_offset + c] = sU[la][c]
        for lb in range(4):
            # lb = j*2+k, j = lb//2, k = lb%2
            j = lb // 2
            k = lb % 2
            gb = j * 5 + k
            V_full[gb][col_offset + c] = sV[lb][c]
        for lc in range(4):
            # lc = k*2+i, k = lc//2, i = lc%2
            k = lc // 2
            i = lc % 2
            gc = k * 2 + i
            W_full[gc][col_offset + c] = sW[lc][c]
    col_offset += 7
    
    # Block 1b: A_bottom * B_1_bottom
    # A_bottom: A[:, 2:4], local (i,j) -> global (i, j+2) -> global index i*4 + (j+2)
    # B_1_bottom: B_1[2:4, :], local (j,k) where j is row in B_1_bottom (0,1) -> global row j+2
    # global B index: (j+2)*5 + k
    # C_1: same as before
    for c in range(7):
        for la in range(4):
            i = la // 2
            j = la % 2
            ga = i * 4 + (j + 2)
            U_full[ga][col_offset + c] = sU[la][c]
        for lb in range(4):
            j = lb // 2
            k = lb % 2
            gb = (j + 2) * 5 + k
            V_full[gb][col_offset + c] = sV[lb][c]
        for lc in range(4):
            k = lc // 2
            i = lc % 2
            gc = k * 2 + i
            W_full[gc][col_offset + c] = sW[lc][c]
    col_offset += 7
    
    # Block 2a: A_top * B_2_top
    # B_2 = B[2:4, :] wait no, B_2 is columns 2,3 of B.
    # B is 4x5. B_2 = B[:, 2:4] (4x2).
    # B_2_top = B_2[0:2, :] = B[0:2, 2:4] (2x2)
    # B_2_bottom = B_2[2:4, :] = B[2:4, 2:4] (2x2)
    # A_top * B_2_top: A_top is A[:, 0:2], B_2_top is B[0:2, 2:4]
    # local B index (j,k) where j in {0,1} (row in B_2_top), k in {0,1} (col in C_2)
    # global B index: j*5 + (k+2)
    # C_2: local (i,k) -> global (i, k+2) -> global C index (k+2)*2 + i
    for c in range(7):
        for la in range(4):
            i = la // 2
            j = la % 2
            ga = i * 4 + j
            U_full[ga][col_offset + c] = sU[la][c]
        for lb in range(4):
            j = lb // 2
            k = lb % 2
            gb = j * 5 + (k + 2)
            V_full[gb][col_offset + c] = sV[lb][c]
        for lc in range(4):
            k = lc // 2
            i = lc % 2
            gc = (k + 2) * 2 + i
            W_full[gc][col_offset + c] = sW[lc][c]
    col_offset += 7
    
    # Block 2b: A_bottom * B_2_bottom
    # A_bottom: A[:, 2:4]
    # B_2_bottom: B[2:4, 2:4]
    for c in range(7):
        for la in range(4):
            i = la // 2
            j = la % 2
            ga = i * 4 + (j + 2)
            U_full[ga][col_offset + c] = sU[la][c]
        for lb in range(4):
            j = lb // 2
            k = lb % 2
            gb = (j + 2) * 5 + (k + 2)
            V_full[gb][col_offset + c] = sV[lb][c]
        for lc in range(4):
            k = lc // 2
            i = lc % 2
            gc = (k + 2) * 2 + i
            W_full[gc][col_offset + c] = sW[lc][c]
    col_offset += 7
    
    # Block 3: A * B_3 where B_3 = B[:, 4] (4x1)
    # C_3 = A * B_3 (2x1)
    # Split into A_top * B_3_top + A_bottom * B_3_bottom
    # B_3_top = B[0:2, 4] (2x1), B_3_bottom = B[2:4, 4] (2x1)
    
    # Block 3a: A_top * B_3_top (2x2x1, standard, rank 4)
    # A_top: A[:, 0:2], local (i,j) -> global (i, j)
    # B_3_top: B[0:2, 4], local (j) where j in {0,1} -> global row j
    # global B index: j*5 + 4
    # C_3: local (i) where i in {0,1} -> global (i, 4) -> global C index 4*2 + i = 8 + i
    for c in range(4):
        for la in range(4):
            i = la // 2
            j = la % 2
            ga = i * 4 + j
            U_full[ga][col_offset + c] = stdU[la][c]
        for lb in range(2):
            j = lb
            gb = j * 5 + 4
            V_full[gb][col_offset + c] = stdV[lb][c]
        for lc in range(2):
            i = lc
            gc = 4 * 2 + i
            W_full[gc][col_offset + c] = stdW[lc][c]
    col_offset += 4
    
    # Block 3b: A_bottom * B_3_bottom (2x2x1, standard, rank 4)
    # A_bottom: A[:, 2:4], local (i,j) -> global (i, j+2)
    # B_3_bottom: B[2:4, 4], local (j) where j in {0,1} -> global row j+2
    # global B index: (j+2)*5 + 4
    for c in range(4):
        for la in range(4):
            i = la // 2
            j = la % 2
            ga = i * 4 + (j + 2)
            U_full[ga][col_offset + c] = stdU[la][c]
        for lb in range(2):
            j = lb
            gb = (j + 2) * 5 + 4
            V_full[gb][col_offset + c] = stdV[lb][c]
        for lc in range(2):
            i = lc
            gc = 4 * 2 + i
            W_full[gc][col_offset + c] = stdW[lc][c]
    col_offset += 4
    
    assert col_offset == rank_total
    
    return {"U": U_full, "V": V_full, "W": W_full}
# EVOLVE-BLOCK-END
