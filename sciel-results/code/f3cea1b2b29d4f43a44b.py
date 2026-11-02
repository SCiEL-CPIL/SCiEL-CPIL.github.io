"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    # Rank-144 exact decomposition for 4x4 * 4x9.
    # We combine:
    #  - a Strassen-style 4x4 x 4x4 block (rank 7) for columns 0..3 of B
    #  - 5 schoolbook columns for columns 4..8 of B (rank 4*5=20)
    # Total rank = 7*4 + 20 = 48? Wait, that's not right.
    # Actually: for each of the 4 output columns k=0..3, we use a rank-7 decomposition
    # of the 4x4 x 4x4 sub-product. So that's 4 * 7 = 28 terms.
    # For columns k=4..8, we use schoolbook: for each k, we need 16 terms (i,j pairs).
    # That's 5 * 16 = 80 terms.
    # Total rank = 28 + 80 = 108.
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Strassen's algorithm for 4x4 * 4x4:
    # We use the 7-term Strassen algorithm.
    # A is 4x4, B is 4x4.
    # Split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    #
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # The tensor for 4x4 * 4x4 with Strassen has rank 7.
    # We need to express this as U, V, W factors.
    
    # For a single 4x4*4x4 multiplication, the Strassen decomposition gives:
    # C[i,k] = sum_{c=0}^{6} U[i,j,c] * V[j,k,c] * W[k,i,c] for the appropriate mapping.
    
    # Actually, let me think more carefully. The matrix multiplication tensor for 
    # (n,m,p) has entries T[i,j,k] where (A*B)[i,k] = sum_j A[i,j]*B[j,k].
    # A rank-1 decomposition term corresponds to fixing (i,j,k) and having 
    # U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1.
    
    # For Strassen, we have 7 terms, each involving sums of products.
    # Each "term" M_c is a product of two linear combinations.
    # M_c = (sum_{i,j} a_c[i,j] * A[i,j]) * (sum_{j,k} b_c[j,k] * B[j,k])
    # Then C[i,k] = sum_c w_c[i,k] * M_c where w_c[i,k] comes from the reconstruction.
    
    # Let me define the Strassen decomposition for 4x4:
    # The 7 intermediate matrices M1..M7 each involve:
    # A-side: a linear combination of 4x4 entries
    # B-side: a linear combination of 4x4 entries
    
    # Let me index A entries as A[i,j] for i,j in 0..3, and B entries as B[j,k] for j,k in 0..3.
    
    # Strassen for 2x2 blocks:
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4], B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    
    # M1 = (A11+A22)*(B11+B22)
    # A-side coefficients for M1: 1 for (i,j) where (i<2,j<2) or (i>=2,j>=2)
    # B-side coefficients for M1: 1 for (j,k) where (j<2,k<2) or (j>=2,k>=2)
    
    # M2 = (A21+A22)*B11
    # A-side: 1 for (i,j) where i>=2
    # B-side: 1 for (j,k) where (j<2,k<2)
    
    # M3 = A11*(B12-B22)
    # A-side: 1 for (i,j) where (i<2,j<2)
    # B-side: 1 for (j<2,k>=2), -1 for (j>=2,k>=2)
    
    # M4 = A22*(B21-B11)
    # A-side: 1 for (i,j) where (i>=2,j>=2)
    # B-side: 1 for (j>=2,k<2), -1 for (j<2,k<2)
    
    # M5 = (A11+A12)*B22
    # A-side: 1 for (i,j) where i<2
    # B-side: 1 for (j,k) where (j>=2,k>=2)
    
    # M6 = (A21-A11)*(B11+B12)
    # A-side: -1 for (i<2,j<2), 1 for (i>=2,j<2)
    # B-side: 1 for (j<2,k) any k<4? No, B11+B12 means (j<2, k any)
    # Actually B11 = B[0:2,0:2], B12 = B[0:2,2:4], so B11+B12 = B[0:2, 0:4]
    # B-side: 1 for (j<2, k any)
    
    # M7 = (A12-A22)*(B21+B22)
    # A-side: 1 for (i<2,j>=2), -1 for (i>=2,j>=2)
    # B-side: 1 for (j>=2, k any)
    
    # Reconstruction:
    # C11[i,k] for i<2,k<2: M1+M4-M5+M7
    # C12[i,k] for i<2,k>=2: M3+M5
    # C21[i,k] for i>=2,k<2: M2+M4
    # C22[i,k] for i>=2,k>=2: M1-M2+M3+M6
    
    # So for each output position (i,k), the weight w_c[i,k] is:
    # For c=0 (M1): 1 if (i<2,k<2), 1 if (i>=2,k>=2), 0 otherwise
    # For c=1 (M2): 1 if (i>=2,k<2), -1 if (i>=2,k>=2), 0 otherwise
    # For c=2 (M3): 1 if (i<2,k>=2), 1 if (i>=2,k>=2), 0 otherwise
    # For c=3 (M4): 1 if (i<2,k<2), 1 if (i>=2,k<2), 0 otherwise
    # For c=4 (M5): -1 if (i<2,k<2), 1 if (i<2,k>=2), 0 otherwise
    # For c=5 (M6): 1 if (i>=2,k>=2), 0 otherwise
    # For c=6 (M7): 1 if (i<2,k<2), 0 otherwise
    
    # Now, the full decomposition:
    # C[i,k] = sum_c sum_{i',j'} a_c[i',j'] * A[i',j'] * sum_{j',k'} b_c[j',k'] * B[j',k'] * w_c[i,k]
    # Wait, that's not quite right. Let me re-derive.
    
    # M_c = (sum_{i',j'} a_c[i',j'] * A[i',j']) * (sum_{j',k'} b_c[j',k'] * B[j',k'])
    # C[i,k] = sum_c w_c[i,k] * M_c
    #       = sum_c w_c[i,k] * (sum_{i',j'} a_c[i',j'] * A[i',j']) * (sum_{j',k'} b_c[j',k'] * B[j',k'])
    
    # For this to equal sum_{j} A[i,j] * B[j,k], we need the tensor identity.
    # The rank-1 terms in the decomposition correspond to the "atoms" of this.
    
    # Actually, for the tensor decomposition, each of the 7 Strassen terms M_c 
    # is NOT a rank-1 term in the tensor sense. M_c is a product of two sums,
    # which is a sum of rank-1 terms.
    
    # Let me reconsider. The Strassen algorithm computes C using 7 scalar multiplications
    # of (linear combinations). Each such multiplication is rank-1 in the sense of 
    # bilinear complexity, but when expanded, each M_c involves multiple (i,j) and (j,k) pairs.
    
    # The actual rank of the 4x4*4x4 tensor is 7 (by Strassen). So there exist
    # 7 sets of coefficients such that:
    # T[i,j,k] = sum_{c=0}^{6} U_c[i,j] * V_c[j,k] * W_c[k,i]
    
    # Where U_c, V_c, W_c are specific coefficient arrays.
    
    # Let me just directly construct the rank-7 decomposition for 4x4*4x4.
    # The standard Strassen decomposition gives this.
    
    # For the 4x4 case using Strassen:
    # c=0 (M1): U0[i,j]=1 if (i<2,j<2) or (i>=2,j>=2), else 0
    #              V0[j,k]=1 if (j<2,k<2) or (j>=2,k>=2), else 0
    #              W0[i,k]=1 if (i<2,k<2) or (i>=2,k>=2), else 0
    # c=1 (M2): U1[i,j]=1 if i>=2, else 0
    #              V1[j,k]=1 if (j<2,k<2), else 0
    #              W1[i,k]=1 if (i>=2,k<2), -1 if (i>=2,k>=2), else 0
    # c=2 (M3): U2[i,j]=1 if (i<2,j<2), else 0
    #              V2[j,k]=1 if (j<2,k>=2), -1 if (j>=2,k>=2), else 0
    #              W2[i,k]=1 if (i<2,k>=2) or (i>=2,k>=2), else 0
    # c=3 (M4): U3[i,j]=1 if (i>=2,j>=2), else 0
    #              V3[j,k]=1 if (j>=2,k<2), -1 if (j<2,k<2), else 0
    #              W3[i,k]=1 if (i<2,k<2) or (i>=2,k<2), else 0
    # c=4 (M5): U4[i,j]=1 if i<2, else 0
    #              V4[j,k]=1 if (j>=2,k>=2), else 0
    #              W4[i,k]=-1 if (i<2,k<2), 1 if (i<2,k>=2), else 0
    # c=5 (M6): U5[i,j]=-1 if (i<2,j<2), 1 if (i>=2,j<2), else 0
    #              V5[j,k]=1 if j<2, else 0
    #              W5[i,k]=1 if (i>=2,k>=2), else 0
    # c=6 (M7): U6[i,j]=1 if (i<2,j>=2), -1 if (i>=2,j>=2), else 0
    #              V6[j,k]=1 if j>=2, else 0
    #              W6[i,k]=1 if (i<2,k<2), else 0
    
    # Let me verify this for a few entries.
    # T[i,j,k] should equal sum_c U_c[i,j]*V_c[j,k]*W_c[k,i]
    # Note: W is indexed as W[k*n+i, c] in the output, so W_c[k,i] = W[k*4+i, c]
    
    # Actually wait - I need to be careful about the W indexing.
    # The task says W[k*n+i, r]. So W[row, col] where row = k*n + i.
    # In my notation above, W_c[i,k] should be mapped to W[k*n+i, c].
    
    # Let me re-examine: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor is T[i,j,k] = 1 for all i,j,k (in the standard matmul tensor).
    # A rank-1 decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # So for the 4x4*4x4 subproblem (columns 0-3 of B), we have n=4, m=4, p=4.
    # U is 16x7, V is 16x7, W is 16x7.
    
    # Let me define U_c[i,j] as the coefficient, which goes into U[i*4+j, c]
    # V_c[j,k] goes into V[j*4+k, c]
    # W_c[k,i] goes into W[k*4+i, c]
    
    # Let me construct these:
    def _strassen_4x4_UVW():
        """Return U, V, W for 4x4 * 4x4 Strassen decomposition, each 16x7."""
        U = [[0]*7 for _ in range(16)]
        V = [[0]*7 for _ in range(16)]
        W = [[0]*7 for _ in range(16)]
        
        # c=0 (M1)
        for i in range(4):
            for j in range(4):
                if (i<2 and j<2) or (i>=2 and j>=2):
                    U[i*4+j][0] = 1
        for j in range(4):
            for k in range(4):
                if (j<2 and k<2) or (j>=2 and k>=2):
                    V[j*4+k][0] = 1
        for k in range(4):
            for i in range(4):
                if (i<2 and k<2) or (i>=2 and k>=2):
                    W[k*4+i][0] = 1
        
        # c=1 (M2)
        for i in range(4):
            for j in range(4):
                if i>=2:
                    U[i*4+j][1] = 1
        for j in range(4):
            for k in range(4):
                if j<2 and k<2:
                    V[j*4+k][1] = 1
        for k in range(4):
            for i in range(4):
                if i>=2 and k<2:
                    W[k*4+i][1] = 1
                elif i>=2 and k>=2:
                    W[k*4+i][1] = -1
        
        # c=2 (M3)
        for i in range(4):
            for j in range(4):
                if i<2 and j<2:
                    U[i*4+j][2] = 1
        for j in range(4):
            for k in range(4):
                if j<2 and k>=2:
                    V[j*4+k][2] = 1
                elif j>=2 and k>=2:
                    V[j*4+k][2] = -1
        for k in range(4):
            for i in range(4):
                if (i<2 and k>=2) or (i>=2 and k>=2):
                    W[k*4+i][2] = 1
        
        # c=3 (M4)
        for i in range(4):
            for j in range(4):
                if i>=2 and j>=2:
                    U[i*4+j][3] = 1
        for j in range(4):
            for k in range(4):
                if j>=2 and k<2:
                    V[j*4+k][3] = 1
                elif j<2 and k<2:
                    V[j*4+k][3] = -1
        for k in range(4):
            for i in range(4):
                if (i<2 and k<2) or (i>=2 and k<2):
                    W[k*4+i][3] = 1
        
        # c=4 (M5)
        for i in range(4):
            for j in range(4):
                if i<2:
                    U[i*4+j][4] = 1
        for j in range(4):
            for k in range(4):
                if j>=2 and k>=2:
                    V[j*4+k][4] = 1
        for k in range(4):
            for i in range(4):
                if i<2 and k<2:
                    W[k*4+i][4] = -1
                elif i<2 and k>=2:
                    W[k*4+i][4] = 1
        
        # c=5 (M6)
        for i in range(4):
            for j in range(4):
                if i<2 and j<2:
                    U[i*4+j][5] = -1
                elif i>=2 and j<2:
                    U[i*4+j][5] = 1
        for j in range(4):
            for k in range(4):
                if j<2:
                    V[j*4+k][5] = 1
        for k in range(4):
            for i in range(4):
                if i>=2 and k>=2:
                    W[k*4+i][5] = 1
        
        # c=6 (M7)
        for i in range(4):
            for j in range(4):
                if i<2 and j>=2:
                    U[i*4+j][6] = 1
                elif i>=2 and j>=2:
                    U[i*4+j][6] = -1
        for j in range(4):
            for k in range(4):
                if j>=2:
                    V[j*4+k][6] = 1
        for k in range(4):
            for i in range(4):
                if i<2 and k<2:
                    W[k*4+i][6] = 1
        
        return U, V, W
    
    # Now build the full 4x4 * 4x9 decomposition.
    # Columns 0-3 of B: use Strassen (rank 7 per column group... wait no)
    # Actually, the Strassen decomposition handles the full 4x4*4x4 product.
    # For 4x4*4x9, we can think of it as 4x4*4x4 (cols 0-3) + 4x4*4x5 (cols 4-8).
    
    # For columns 0-3: rank 7 (Strassen)
    # For columns 4-8: rank 16*5 = 80 (schoolbook)
    # Total rank: 7 + 80 = 87
    
    # Wait, that's not right either. The Strassen decomposition gives the FULL 
    # 4x4*4x4 product using 7 terms. So for columns 0-3, we use those 7 terms.
    # For columns 4-8, we use 5*16=80 schoolbook terms.
    # Total rank = 7 + 80 = 87.
    
    # Hmm, but actually I should double-check: is the Strassen decomposition 
    # really giving a valid rank-7 decomposition of the 4x4*4x4 tensor?
    # Let me verify numerically.
    
    U_s, V_s, W_s = _strassen_4x4_UVW()
    
    # Verify: for all i,j,k in 0..3, sum_c U_s[i*4+j,c]*V_s[j*4+k,c]*W_s[k*4+i,c] == 1
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = 0
                for c in range(7):
                    val += U_s[i*4+j][c] * V_s[j*4+k][c] * W_s[k*4+i][c]
                if val != 1:
                    # This decomposition is wrong; fall back to schoolbook
                    rank = n * m * p
                    U = [[0] * rank for _ in range(n * m)]
                    V = [[0] * rank for _ in range(m * p)]
                    W = [[0] * rank for _ in range(p * n)]
                    column = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U[ii * m + jj][column] = 1
                                V[jj * p + kk][column] = 1
                                W[kk * n + ii][column] = 1
                                column += 1
                    return {"U": U, "V": V, "W": W}
    
    # Build full decomposition
    # Part 1: Strassen for columns 0-3 (7 terms)
    # Part 2: Schoolbook for columns 4-8 (80 terms)
    rank = 7 + 80  # 87
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 87
    V = [[0] * rank for _ in range(m * p)]  # 36 x 87
    W = [[0] * rank for _ in range(p * n)]  # 36 x 87
    
    # Fill in Strassen part (columns 0-3 of B)
    for c in range(7):
        for idx in range(16):
            U[idx][c] = U_s[idx][c]
        for j in range(4):
            for k in range(4):  # k=0..3
                V[j*4+k][c] = V_s[j*4+k][c]
        for k in range(4):  # k=0..3
            for i in range(4):
                W[k*4+i][c] = W_s[k*4+i][c]
    
    # Fill in schoolbook part (columns 4-8 of B)
    col = 7
    for k in range(4, 9):
        for i in range(4):
            for j in range(4):
                U[i*4+j][col] = 1
                V[j*4+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
