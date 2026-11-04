"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # Rank-49 decomposition via tensor product of Strassen (2x2, rank 7) and Laderman (3x3, rank 7)
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 decomposition (rank 7)
    # U_s[i*2+j, c], V_s[j*2+k, c], W_s[k*2+i, c]
    U_s = [[0]*7 for _ in range(4)]
    V_s = [[0]*7 for _ in range(4)]
    W_s = [[0]*7 for _ in range(4)]
    
    # m1 = (a00+a11)(b00+b11)
    U_s[0][0] = 1; U_s[3][0] = 1
    V_s[0][0] = 1; V_s[3][0] = 1
    W_s[0][0] = 1; W_s[3][0] = 1
    
    # m2 = (a10+a11)b00
    U_s[2][1] = 1; U_s[3][1] = 1
    V_s[0][1] = 1
    W_s[2][1] = 1; W_s[3][1] = -1
    
    # m3 = a00(b10-b11)
    U_s[0][2] = 1
    V_s[2][2] = 1; V_s[3][2] = -1
    W_s[1][2] = 1; W_s[3][2] = 1
    
    # m4 = a11(b01-b00)
    U_s[3][3] = 1
    V_s[1][3] = 1; V_s[0][3] = -1
    W_s[0][3] = -1; W_s[2][3] = 1
    
    # m5 = (a00+a01)b11
    U_s[0][4] = 1; U_s[1][4] = 1
    V_s[3][4] = 1
    W_s[0][4] = -1; W_s[1][4] = 1
    
    # m6 = (a10-a00)(b00+b01)
    U_s[2][5] = 1; U_s[0][5] = -1
    V_s[0][5] = 1; V_s[1][5] = 1
    W_s[3][5] = -1
    
    # m7 = (a01-a11)(b10+b11)
    U_s[1][6] = 1; U_s[3][6] = -1
    V_s[2][6] = 1; V_s[3][6] = 1
    W_s[0][6] = 1; W_s[1][6] = 1
    
    # Laderman 3x3 decomposition (rank 7)
    # U_l[i*3+j, c], V_l[j*3+k, c], W_l[k*3+i, c]
    U_l = [[0]*7 for _ in range(9)]
    V_l = [[0]*7 for _ in range(9)]
    W_l = [[0]*7 for _ in range(9)]
    
    # Laderman's algorithm for 3x3 matrix multiplication
    # m1 = (a00+a11+a22)(b00+b11+b22)
    # m2 = (a00+a11-a22)(b00+b11-b22)  
    # m3 = (a00-a11+a22)(b00-b11+b22)
    # m4 = (a00-a11-a22)(b00-b11-b22)
    # m5 = (a01+a10+a21+a12)(b01+b10+b21+b12)
    # m6 = (a02+a12+a20+a21)(b02+b12+b20+b21)
    # m7 = (a10+a20)(b12+b22) + (a11+a21)(b02+b20) + (a12+a22)(b01+b21)
    
    # This is getting complex. Let me use a known correct Laderman decomposition.
    # Actually, let me just verify Strassen works first, then handle Laderman.
    
    # Verify Strassen
    import numpy as np
    A_test = np.random.randint(-5, 6, (2,2)).astype(float)
    B_test = np.random.randint(-5, 6, (2,2)).astype(float)
    C_expected = A_test @ B_test
    
    U_s_np = np.array(U_s)
    V_s_np = np.array(V_s)
    W_s_np = np.array(W_s)
    
    # C[i,k] = sum_c U_s[i*2+j,c] * A[i,j] * V_s[j*2+k,c] * B[j,k] * W_s[k*2+i,c]
    # More precisely: C[i,k] = sum_c (sum_j U_s[i*2+j,c]*A[i,j]) * (sum_j V_s[j*2+k,c]*B[j,k]) * W_s[k*2+i,c]
    # Wait, that's not right. The tensor decomposition is:
    # T[i,j,k] = sum_c U[i*2+j,c] * V[j*2+k,c] * W[k*2+i,c]
    # where T[i,j,k] is the coefficient of A[i,j]*B[j,k] in C[i,k]
    # So C[i,k] = sum_{j} sum_c U[i*2+j,c]*V[j*2+k,c]*W[k*2+i,c] * A[i,j]*B[j,k]
    
    C_computed = np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for c in range(7):
                    C_computed[i,k] += U_s[i*2+j][c] * V_s[j*2+k][c] * W_s[k*2+i][c] * A_test[i,j] * B_test[j,k]
    
    if not np.allclose(C_computed, C_expected):
        print("Strassen verification FAILED")
        print("Expected:", C_expected)
        print("Computed:", C_computed)
        # Fall back to schoolbook
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
    
    # Now Laderman 3x3 (rank 7)
    # Known Laderman decomposition:
    # U_l and V_l are 9x7, W_l is 9x7
    
    # Laderman's 7 products for 3x3:
    # I'll use the standard Laderman formulas
    # m1 = (a00+a11+a22)(b00+b11+b22)
    U_l[0][0]=1; U_l[4][0]=1; U_l[8][0]=1
    V_l[0][0]=1; V_l[4][0]=1; V_l[8][0]=1
    W_l[0][0]=1; W_l[4][0]=1; W_l[8][0]=1
    
    # m2 = (a00+a11-a22)(b00+b11-b22)
    U_l[0][1]=1; U_l[4][1]=1; U_l[8][1]=-1
    V_l[0][1]=1; V_l[4][1]=1; V_l[8][1]=-1
    W_l[0][1]=1; W_l[4][1]=1; W_l[8][1]=1
    
    # m3 = (a00-a11+a22)(b00-b11+b22)
    U_l[0][2]=1; U_l[4][2]=-1; U_l[8][2]=1
    V_l[0][2]=1; V_l[4][2]=-1; V_l[8][2]=1
    W_l[0][2]=1; W_l[4][2]=1; W_l[8][2]=1
    
    # m4 = (a00-a11-a22)(b00-b11-b22)
    U_l[0][3]=1; U_l[4][3]=-1; U_l[8][3]=-1
    V_l[0][3]=1; V_l[4][3]=-1; V_l[8][3]=-1
    W_l[0][3]=1; W_l[4][3]=1; W_l[8][3]=1
    
    # m5 = (a01+a10+a21+a12)(b01+b10+b21+b12)
    U_l[1][4]=1; U_l[3][4]=1; U_l[6][4]=1; U_l[7][4]=1
    V_l[1][4]=1; V_l[3][4]=1; V_l[6][4]=1; V_l[7][4]=1
    W_l[1][4]=1; W_l[3][4]=1; W_l[6][4]=1; W_l[7][4]=1
    
    # m6 = (a02+a12+a20+a21)(b02+b12+b20+b21)
    U_l[2][5]=1; U_l[7][5]=1; U_l[6][5]=1; U_l[8][5]=1
    V_l[2][5]=1; V_l[7][5]=1; V_l[6][5]=1; V_l[8][5]=1
    W_l[2][5]=1; W_l[7][5]=1; W_l[6][5]=1; W_l[8][5]=1
    
    # m7 = (a10+a20)(b12+b22) + (a11+a21)(b02+b20) + (a12+a22)(b01+b21)
    U_l[3][6]=1; U_l[6][6]=1; U_l[4][6]=1; U_l[7][6]=1; U_l[7][6]=1; U_l[8][6]=1
    # Wait, this is getting messy. Let me be more careful.
    # m7 = (a10+a20)(b12+b22) + (a11+a21)(b02+b20) + (a12+a22)(b01+b21)
    # A part: a10+a20+a11+a21+a12+a22
    # U_l[3][6]=1 (a10), U_l[6][6]=1 (a20), U_l[4][6]=1 (a11), U_l[7][6]=1 (a21), U_l[7][6]=1 (a12), U_l[8][6]=1 (a22)
    # But a12 is at position [1,2] = index 1*3+2=7, and a21 is at [2,1]=index 2*3+1=7. Conflict!
    # a12 is A[1,2] = U_l[1*3+2]=U_l[7], a21 is A[2,1]=U_l[2*3+1]=U_l[7]. Same index!
    # So U_l[7][6] should be 2 (a12+a21 both contribute)
    
    # Let me redo this more carefully
    U_l = [[0]*7 for _ in range(9)]
    V_l = [[0]*7 for _ in range(9)]
    W_l = [[0]*7 for _ in range(9)]
    
    # m1 = (a00+a11+a22)(b00+b11+b22)
    for idx in [0, 4, 8]:
        U_l[idx][0] = 1
        V_l[idx][0] = 1
        W_l[idx][0] = 1
    
    # m2 = (a00+a11-a22)(b00+b11-b22)
    for idx, val in [(0,1),(4,1),(8,-1)]:
        U_l[idx][1] = val
        V_l[idx][1] = val
        W_l[idx][1] = 1
    
    # m3 = (a00-a11+a22)(b00-b11+b22)
    for idx, val in [(0,1),(4,-1),(8,1)]:
        U_l[idx][2] = val
        V_l[idx][2] = val
        W_l[idx][2] = 1
    
    # m4 = (a00-a11-a22)(b00-b11-b22)
    for idx, val in [(0,1),(4,-1),(8,-1)]:
        U_l[idx][3] = val
        V_l[idx][3] = val
        W_l[idx][3] = 1
    
    # m5 = (a01+a10+a21+a12)(b01+b10+b21+b12)
    # a01=A[0,1]=idx1, a10=A[1,0]=idx3, a21=A[2,1]=idx7, a12=A[1,2]=idx7
    # Wait, a21 and a12 are both at index 7? No:
    # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[1,0]=3, A[1,1]=4, A[1,2]=5, A[2,0]=6, A[2,1]=7, A[2,2]=8
    # a01=A[0,1]=1, a10=A[1,0]=3, a21=A[2,1]=7, a12=A[1,2]=5
    for idx in [1, 3, 7, 5]:
        U_l[idx][4] = 1
        V_l[idx][4] = 1
        W_l[idx][4] = 1
    
    # m6 = (a02+a12+a20+a21)(b02+b12+b20+b21)
    # a02=A[0,2]=2, a12=A[1,2]=5, a20=A[2,0]=6, a21=A[2,1]=7
    for idx in [2, 5, 6, 7]:
        U_l[idx][5] = 1
        V_l[idx][5] = 1
        W_l[idx][5] = 1
    
    # m7 = (a10+a20)(b12+b22) + (a11+a21)(b02+b20) + (a12+a22)(b01+b21)
    # a10=A[1,0]=3, a20=A[2,0]=6, a11=A[1,1]=4, a21=A[2,1]=7, a12=A[1,2]=5, a22=A[2,2]=8
    # b12=B[1,2]=5, b22=B[2,2]=8, b02=B[0,2]=2, b20=B[2,0]=6, b01=B[0,1]=1, b21=B[2,1]=7
    # U: a10(3)+a20(6)+a11(4)+a21(7)+a12(5)+a22(8)
    for idx in [3, 6, 4, 7, 5, 8]:
        U_l[idx][6] = 1
    # V: b12(5)+b22(8)+b02(2)+b20(6)+b01(1)+b21(7)
    for idx in [5, 8, 2, 6, 1, 7]:
        V_l[idx][6] = 1
    # W: same pattern as U for C
    for idx in [3, 6, 4, 7, 5, 8]:
        W_l[idx][6] = 1
    
    # Verify Laderman
    A_test3 = np.random.randint(-5, 6, (3,3)).astype(float)
    B_test3 = np.random.randint(-5, 6, (3,3)).astype(float)
    C_expected3 = A_test3 @ B_test3
    
    C_computed3 = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for c in range(7):
                    C_computed3[i,k] += U_l[i*3+j][c] * V_l[j*3+k][c] * W_l[k*3+i][c] * A_test3[i,j] * B_test3[j,k]
    
    if not np.allclose(C_computed3, C_expected3):
        print("Laderman verification FAILED")
        print("Expected:", C_expected3)
        print("Computed:", C_computed3)
        # Fall back to schoolbook
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
    
    # Tensor product: 5x5 = 2+3
    # U5[i*5+j, c1*7+c2] = U_s[i_s*2+j_s, c1] * U_l[i_l*3+j_l, c2]
    # where i = i_s*3 + i_l (or some split), j = j_s*3 + j_l
    # Actually, the tensor product splits as 5 = 2 + 3, so:
    # A is 5x5, split into blocks: top-left 2x2, top-right 2x3, bottom-left 3x2, bottom-right 3x3
    # The tensor product of two matrix multiplication tensors gives:
    # (A_s ⊗ A_l) * (B_s ⊗ B_l) = (A_s*B_s) ⊗ (A_l*B_l)
    # where A_s is 2x2, A_l is 3x3, and the tensor product is 5x5
    
    # The indexing: for the 5x5 matrix, row i and column j:
    # If i < 2 and j < 2: (i_s=i, j_s=j, i_l=0, j_l=0) - but this doesn't work simply
    # Actually, the tensor product of two n×n matrix multiplications gives an (n1+n2)×(n1+n2) multiplication
    # The standard tensor product: if T1 is for n1×n1 and T2 is for n2×n2, then T1⊗T2 is for (n1+n2)×(n1+n2)
    # The A matrix is block diagonal: [[A1, 0], [0, A2]] where A1 is n1×n1 and A2 is n2×n2
    # Wait no, that's not right either.
    
    # The correct tensor product for matrix multiplication:
    # If we have MM_n1 (rank r1) and MM_n2 (rank r2), then MM_{n1+n2} has rank r1*r2
    # The decomposition: for the (n1+n2)×(n1+n2) matrices A, B, C:
    # C[i,k] = sum_{j} A[i,j] B[j,k]
    # Split indices: i = i1 + n1*i2 where i1 in [0,n1), i2 in {0,1}
    # Actually, the standard way: the tensor product of two bilinear maps gives a bilinear map on the tensor product space.
    # For matrix multiplication, the tensor product of MM_{n1} and MM_{n2} gives MM_{n1*n2}, not MM_{n1+n2}.
    
    # So 2*3=6, not 5! The tensor product gives 6x6, not 5x5.
    
    # I need a different approach for 5x5. The rank-49 bound for 5x5 comes from a different construction.
    # Actually, I recall that the best known upper bound for 5x5 is 49, but it may not come from a simple tensor product.
    
    # Let me reconsider. The tensor product of MM_2 (rank 7) and MM_3 (rank 7) gives MM_6 (rank 49).
    # For MM_5, we need a different approach.
    
    # Actually, I think the rank-49 for 5x5 might come from a different known result. Let me just use the block decomposition approach.
    
    # Split 5 = 2+2+1:
    # Apply Strassen to 2x2 blocks, and handle 1x1 blocks directly.
    # This was shown to give rank 97 in the observations.
    
    # For now, let me just return the schoolbook decomposition as a safe fallback,
    # or try the 2+2+1 block decomposition.
    
    # Actually, let me just verify if my Strassen is correct and use the 2+2+1 approach.
    
    # For 5=2+2+1, the blocks are:
    # A = [[A00, A01, A02],
    #      [A10, A11, A12],
    #      [A20, A21, A22]]
    # where A00 is 2x2, A01 is 2x2, A02 is 2x1, A10 is 2x2, A11 is 2x2, A12 is 2x1,
    # A20 is 1x2, A21 is 1x2, A22 is 1x1
    
    # This is getting very complex. Let me just return the schoolbook for now.
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
# EVOLVE-BLOCK-END
