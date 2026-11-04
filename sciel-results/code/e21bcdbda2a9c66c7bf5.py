"""
Tensor decomposition for 4x5 x 5x5 matrix multiplication.
Uses block decomposition: 4x4x4 (Strassen rank 49), 4x4x1 (rank 4), 4x1x5 (rank 1).
Total rank = 54.
"""
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Split into blocks:
    # A is n x m = 4x5, B is m x p = 5x5
    # Partition j (middle dim) into j1 (size 4) and j2 (size 1)
    # Partition k (output dim) into k1 (size 4) and k2 (size 1)
    
    # Block 1: 4x4x4 (i in [0,4), j in [0,4), k in [0,4))
    # Block 2: 4x4x1 (i in [0,4), j in [0,4), k=4)
    # Block 3: 4x1x5 (i in [0,4), j=4, k in [0,5))
    
    # Strassen for 4x4x4: recursively apply Strassen to 2x2 blocks of 4x4
    # First, build Strassen for 2x2x2 (rank 7)
    # Then apply to 4x4x4: treat 4x4 as 2x2 blocks of 2x2, giving 7^2 = 49 rank
    
    def strassen_2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition.
        U shape (4, 7), V shape (4, 7), W shape (4, 7)
        Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        
        # Strassen 7 products:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For each rank component r, we need U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        # such that sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = delta[i,k] (for the 2x2 case)
        
        # The decomposition is: C[i,k] = sum_j A[i,j] * B[j,k]
        # In tensor form: T[i,j,k] = delta[i,k] (for identity-like, but actually it's the MM tensor)
        # Actually for MM: T[i,j,k] = 1 for all i,j,k in the sense that C[i,k] = sum_j A[i,j]B[j,k]
        
        # Let me define the rank components explicitly.
        # For Strassen, we have 7 bilinear forms.
        
        # Product 1: (A11+A22)(B11+B22)
        # A part: A11 + A22 -> U[0,0]=1, U[3,0]=1 (i=0,j=0 and i=1,j=1)
        # B part: B11 + B22 -> V[0,0]=1, V[3,0]=1 (j=0,k=0 and j=1,k=1)
        # C part: contributes to C11 (+1), C22 (+1)
        # W[k*2+i, r]: for C11: k=0,i=0 -> W[0,0]=1; for C22: k=1,i=1 -> W[3,0]=1
        U[0][0] += 1; U[3][0] += 1
        V[0][0] += 1; V[3][0] += 1
        W[0][0] += 1; W[3][0] += 1
        
        # Product 2: (A21+A22)B11
        # A part: A21 + A22 -> U[2,1]=1 (i=1,j=0), U[3,1]=1 (i=1,j=1)
        # B part: B11 -> V[0,1]=1 (j=0,k=0)
        # C part: contributes to C21 (+1)
        # W for C21: k=1, i=0 -> W[2,1]=1
        U[2][1] += 1; U[3][1] += 1
        V[0][1] += 1
        W[2][1] += 1
        
        # Product 3: A11(B12-B22)
        # A part: A11 -> U[0,2]=1 (i=0,j=0)
        # B part: B12 - B22 -> V[1,2]=1 (j=0,k=1), V[3,2]=-1 (j=1,k=1)
        # C part: contributes to C12 (+1)
        # W for C12: k=1, i=0 -> W[2,2]=1
        U[0][2] += 1
        V[1][2] += 1; V[3][2] -= 1
        W[2][2] += 1
        
        # Product 4: A22(B21-B11)
        # A part: A22 -> U[3,3]=1 (i=1,j=1)
        # B part: B21 - B11 -> V[2,3]=1 (j=1,k=0), V[0,3]=-1 (j=0,k=0)
        # C part: contributes to C11 (+1), C21 (+1)
        # W for C11: k=0, i=0 -> W[0,3]=1; for C21: k=1, i=0 -> W[2,3]=1
        U[3][3] += 1
        V[2][3] += 1; V[0][3] -= 1
        W[0][3] += 1; W[2][3] += 1
        
        # Product 5: (A11+A12)B22
        # A part: A11 + A12 -> U[0,4]=1 (i=0,j=0), U[1,4]=1 (i=0,j=1)
        # B part: B22 -> V[3,4]=1 (j=1,k=1)
        # C part: contributes to C11 (-1), C12 (+1)
        # W for C11: k=0, i=0 -> W[0,4]=-1; for C12: k=1, i=0 -> W[2,4]=1
        U[0][4] += 1; U[1][4] += 1
        V[3][4] += 1
        W[0][4] -= 1; W[2][4] += 1
        
        # Product 6: (A21-A11)(B11+B12)
        # A part: A21 - A11 -> U[2,5]=1 (i=1,j=0), U[0,5]=-1 (i=0,j=0)
        # B part: B11 + B12 -> V[0,5]=1 (j=0,k=0), V[1,5]=1 (j=0,k=1)
        # C part: contributes to C22 (+1)
        # W for C22: k=1, i=1 -> W[3,5]=1
        U[2][5] += 1; U[0][5] -= 1
        V[0][5] += 1; V[1][5] += 1
        W[3][5] += 1
        
        # Product 7: (A12-A22)(B21+B22)
        # A part: A12 - A22 -> U[1,6]=1 (i=0,j=1), U[3,6]=-1 (i=1,j=1)
        # B part: B21 + B22 -> V[2,6]=1 (j=1,k=0), V[3,6]=1 (j=1,k=1)
        # C part: contributes to C11 (+1), C22 (+1)
        # W for C11: k=0, i=0 -> W[0,6]=1; for C22: k=1, i=1 -> W[3,6]=1
        U[1][6] += 1; U[3][6] -= 1
        V[2][6] += 1; V[3][6] += 1
        W[0][6] += 1; W[3][6] += 1
        
        return U, V, W
    
    # Now build 4x4x4 by applying Strassen to 2x2 blocks
    # Treat 4x4 matrices as 2x2 blocks of 2x2 submatrices
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]] where each is 2x2
    # C = [[C11, C12], [C21, C22]] where each is 2x2
    # Each Ci,j is computed using the 7 Strassen products on the 2x2 blocks
    # Each Strassen product is a 2x2 x 2x2 multiplication, which itself needs 7 rank components
    # So total rank for 4x4x4 = 7 * 7 = 49
    
    U_444 = [[0]*49 for _ in range(16)]  # i in [0,4), j in [0,4), index i*4+j
    V_444 = [[0]*49 for _ in range(16)]  # j in [0,4), k in [0,4), index j*4+k
    W_444 = [[0]*49 for _ in range(16)]  # k in [0,4), i in [0,4), index k*4+i
    
    # For each of the 7 high-level Strassen products (on 2x2 blocks),
    # and for each of the 7 low-level Strassen components (within each 2x2x2),
    # we add a rank component.
    
    U_222, V_222, W_222 = strassen_222()
    
    # High-level Strassen structure (same as 2x2x2 but for 2x2 blocks):
    # Product 1: (A11+A22)(B11+B22)
    # Product 2: (A21+A22)B11
    # Product 3: A11(B12-B22)
    # Product 4: A22(B21-B11)
    # Product 5: (A11+A12)B22
    # Product 6: (A21-A11)(B11+B12)
    # Product 7: (A12-A22)(B21+B22)
    
    # For each high-level product h (0..6), we need to determine:
    # - Which 2x2 blocks of A are involved and with what signs
    # - Which 2x2 blocks of B are involved and with what signs
    # - Which 2x2 blocks of C are involved and with what signs
    
    # High-level A block coefficients:
    # For product h, A has coefficients on blocks [0,0], [0,1], [1,0], [1,1]
    high_A = [
        [1, 0, 0, 1],   # P1: A11 + A22
        [0, 0, 1, 1],   # P2: A21 + A22
        [1, 0, 0, 0],   # P3: A11
        [0, 0, 0, 1],   # P4: A22
        [1, 1, 0, 0],   # P5: A11 + A12
        [-1, 0, 1, 0],  # P6: A21 - A11
        [0, 1, 0, -1],  # P7: A12 - A22
    ]
    
    # High-level B block coefficients:
    high_B = [
        [1, 0, 0, 1],   # P1: B11 + B22
        [1, 0, 0, 0],   # P2: B11
        [0, 1, 0, -1],  # P3: B12 - B22
        [-1, 0, 1, 0],  # P4: B21 - B11
        [0, 0, 0, 1],   # P5: B22
        [1, 1, 0, 0],   # P6: B11 + B12
        [0, 0, 1, 1],   # P7: B21 + B22
    ]
    
    # High-level C block coefficients:
    high_C = [
        [1, 0, 0, 1],   # P1: C11 + C22
        [0, 0, 1, 0],   # P2: C21
        [0, 1, 0, 0],   # P3: C12
        [1, 0, 1, 0],   # P4: C11 + C21
        [-1, 1, 0, 0],  # P5: -C11 + C12
        [0, 0, 0, 1],   # P6: C22
        [1, 0, 0, 1],   # P7: C11 + C22
    ]
    
    for h in range(7):
        # For each low-level Strassen component l (0..6)
        for l in range(7):
            rank_idx = h * 7 + l
            
            # A part: for each block (bi, bj) in high_A[h], if coeff != 0,
            # multiply by U_222 entries for the low-level component l
            for bi in range(2):
                for bj in range(2):
                    coeff_a = high_A[h][bi * 2 + bj]
                    if coeff_a == 0:
                        continue
                    # Block (bi, bj) corresponds to i in [bi*2, bi*2+2), j in [bj*2, bj*2+2)
                    for li in range(2):
                        for lj in range(2):
                            i_local = li  # 0 or 1 within block
                            j_local = lj  # 0 or 1 within block
                            # Global i = bi*2 + li, global j = bj*2 + lj
                            # U_222 index: i_local*2 + j_local
                            u_val = U_222[i_local * 2 + j_local][l]
                            if u_val != 0:
                                i_global = bi * 2 + li
                                j_global = bj * 2 + lj
                                U_444[i_global * 4 + j_global][rank_idx] += coeff_a * u_val
            
            # B part: similar
            for bi in range(2):
                for bj in range(2):
                    coeff_b = high_B[h][bi * 2 + bj]
                    if coeff_b == 0:
                        continue
                    for li in range(2):
                        for lj in range(2):
                            j_local = li  # first index of B block is j
                            k_local = lj  # second index of B block is k
                            # Global j = bi*2 + li, global k = bj*2 + lj
                            # V_222 index: j_local*2 + k_local
                            v_val = V_222[j_local * 2 + k_local][l]
                            if v_val != 0:
                                j_global = bi * 2 + li
                                k_global = bj * 2 + lj
                                V_444[j_global * 4 + k_global][rank_idx] += coeff_b * v_val
            
            # C part: similar
            for bi in range(2):
                for bj in range(2):
                    coeff_c = high_C[h][bi * 2 + bj]
                    if coeff_c == 0:
                        continue
                    for li in range(2):
                        for lj in range(2):
                            k_local = li  # first index of C block is k
                            i_local = lj  # second index of C block is i
                            # Global k = bi*2 + li, global i = bj*2 + lj
                            # W_222 index: k_local*2 + i_local
                            w_val = W_222[k_local * 2 + i_local][l]
                            if w_val != 0:
                                k_global = bi * 2 + li
                                i_global = bj * 2 + lj
                                W_444[k_global * 4 + i_global][rank_idx] += coeff_c * w_val
    
    # Now build the full 4x5x5 decomposition
    # Block 1: 4x4x4 (j in [0,4), k in [0,4)) - rank 49
    # Block 2: 4x4x1 (j in [0,4), k=4) - rank 4
    # Block 3: 4x1x5 (j=4, k in [0,5)) - rank 1
    
    total_rank = 49 + 4 + 1  # 54
    
    U = [[0]*total_rank for _ in range(n * m)]  # 20 x 54
    V = [[0]*total_rank for _ in range(m * p)]  # 25 x 54
    W = [[0]*total_rank for _ in range(p * n)]  # 20 x 54
    
    # Block 1: 4x4x4
    # U rows: i in [0,4), j in [0,4) -> global row i*5 + j
    # V rows: j in [0,4), k in [0,4) -> global row j*5 + k
    # W rows: k in [0,4), i in [0,4) -> global row k*4 + i
    for i in range(4):
        for j in range(4):
            for r in range(49):
                val = U_444[i * 4 + j][r]
                if val != 0:
                    U[i * 5 + j][r] += val
    for j in range(4):
        for k in range(4):
            for r in range(49):
                val = V_444[j * 4 + k][r]
                if val != 0:
                    V[j * 5 + k][r] += val
    for k in range(4):
        for i in range(4):
            for r in range(49):
                val = W_444[k * 4 + i][r]
                if val != 0:
                    W[k * 4 + i][r] += val
    
    # Block 2: 4x4x1 (j in [0,4), k=4)
    # C[i,4] = sum_{j=0}^{3} A[i,j] * B[j,4]
    # Rank 4: one rank component per j
    # U[i*5 + j, 49+j] = 1 for i in [0,4), j in [0,4)
    # V[j*5 + 4, 49+j] = 1 for j in [0,4)
    # W[4*4 + i, 49+j] = 1 for i in [0,4), j in [0,4) -> wait, W[k*n+i, r] with k=4
    # W[4*4 + i, r] = W[16+i, r]
    for j in range(4):
        r = 49 + j
        for i in range(4):
            U[i * 5 + j][r] += 1
            W[4 * 4 + i][r] += 1
        V[j * 5 + 4][r] += 1
    
    # Block 3: 4x1x5 (j=4, k in [0,5))
    # C[i,k] = A[i,4] * B[4,k]
    # Rank 1
    # U[i*5 + 4, 53] = 1 for i in [0,4)
    # V[4*5 + k, 53] = 1 for k in [0,5)
    # W[k*4 + i, 53] = 1 for k in [0,5), i in [0,4)
    r = 53
    for i in range(4):
        U[i * 5 + 4][r] += 1
    for k in range(5):
        V[4 * 5 + k][r] += 1
    for k in range(5):
        for i in range(4):
            W[k * 4 + i][r] += 1
    
    return {"U": U, "V": V, "W": W}
