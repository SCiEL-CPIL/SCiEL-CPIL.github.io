import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Uses Strassen's algorithm recursively to achieve rank 49 for each 4x4x4 block,
    resulting in a total rank of 98 for the 4x4x8 multiplication.
    """
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    # Helper to get rank-7 factors for 2x2x2 Strassen
    def get_strassen_2x2_factors():
        # U, V, W are 4x7 matrices
        # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00 + A11)(B00 + B11)
        # X = A00 + A11, Y = B00 + B11
        # U for X: A00 (0,0)=1, A11 (1,1)=1
        # V for Y: B00 (0,0)=1, B11 (1,1)=1
        # W for C: C00 = M1... 
        # Actually, we just need the factors for the multiplication X*Y = Z.
        # The factors are the same for all 7 terms, just with different coefficients in U and V.
        
        # Let's define the 7 terms explicitly for 2x2.
        # Term 0: M1
        U[:, 0] = [1, 0, 0, 1] # A00 + A11 -> (0,0)=1, (1,1)=1
        V[:, 0] = [1, 0, 0, 1] # B00 + B11 -> (0,0)=1, (1,1)=1
        W[:, 0] = [1, 1, 1, 1] # C00 += M1, C01 += M1? No.
        # Strassen reconstruction:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So for each M_t, we add its contribution to W.
        # M1 contributes to C00, C11
        W[:, 0] = [1, 0, 0, 1] # C00 (k=0,i=0), C11 (k=1,i=1)
        
        # Term 1: M2 = (A10 + A11)B00
        U[:, 1] = [0, 0, 1, 1] # A10 (1,0)=1, A11 (1,1)=1
        V[:, 1] = [1, 0, 0, 0] # B00 (0,0)=1
        W[:, 1] = [0, 0, 1, 0] # C10 (k=1,i=0) += M2
        
        # Term 2: M3 = A00(B01 - B11)
        U[:, 2] = [1, 0, 0, 0] # A00
        V[:, 2] = [0, 1, 0, -1] # B01 (0,1)=1, B11 (1,1)=-1
        W[:, 2] = [0, 1, 0, 0] # C01 (k=0,i=1) += M3
        
        # Term 3: M4 = A11(B10 - B00)
        U[:, 3] = [0, 0, 0, 1] # A11
        V[:, 3] = [0, 0, 1, -1] # B10 (1,0)=1, B00 (0,0)=-1
        W[:, 3] = [1, 0, 1, 0] # C00 += M4, C10 += M4
        
        # Term 4: M5 = (A00 + A01)B11
        U[:, 4] = [1, 1, 0, 0] # A00, A01
        V[:, 4] = [0, 0, 0, 1] # B11
        W[:, 4] = [0, 1, -1, 0] # C01 += M5, C00 -= M5
        
        # Term 5: M6 = (A10 - A00)(B00 + B01)
        U[:, 5] = [-1, 0, 1, 0] # A10=1, A00=-1
        V[:, 5] = [1, 1, 0, 0] # B00=1, B01=1
        W[:, 5] = [0, 0, 0, 1] # C11 += M6
        
        # Term 6: M7 = (A01 - A11)(B10 + B11)
        U[:, 6] = [0, 1, 0, -1] # A01=1, A11=-1
        V[:, 6] = [0, 0, 1, 1] # B10=1, B11=1
        W[:, 6] = [1, 0, 0, 1] # C00 += M7, C11 += M7
        
        return U, V, W

    U_2, V_2, W_2 = get_strassen_2x2_factors()
    
    # Now build 4x4x4 factors
    # Rank 49
    U_4 = np.zeros((16, 49))
    V_4 = np.zeros((16, 49))
    W_4 = np.zeros((16, 49))
    
    # Strassen coefficients for 4x4 (blocks are 2x2)
    # M1: X=A00+A11, Y=B00+B11
    # M2: X=A10+A11, Y=B00
    # M3: X=A00, Y=B01-B11
    # M4: X=A11, Y=B10-B00
    # M5: X=A00+A01, Y=B11
    # M6: X=A10-A00, Y=B00+B01
    # M7: X=A01-A11, Y=B10+B11
    
    # Coefficients for A blocks (2x2 matrix of coefficients)
    # Index [block_i, block_j]
    coeff_A = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, 1]], # M2
        [[1, 0], [0, 0]], # M3
        [[0, 0], [0, 1]], # M4
        [[1, 1], [0, 0]], # M5
        [[-1, 0], [1, 0]],# M6
        [[0, 1], [0, -1]] # M7
    ]
    
    # Coefficients for B blocks
    coeff_B = [
        [[1, 0], [0, 1]], # M1
        [[1, 0], [0, 0]], # M2
        [[0, 1], [0, -1]],# M3
        [[0, -1], [1, 0]],# M4
        [[0, 0], [0, 1]], # M5
        [[1, 1], [0, 0]], # M6
        [[0, 0], [1, 1]]  # M7
    ]
    
    rank_offset = 0
    for t in range(7):
        for r in range(7):
            col = rank_offset + r
            # For each local element in the 2x2 blocks
            for bi in range(2):
                for bj in range(2):
                    ca = coeff_A[t][bi][bj]
                    if ca == 0: continue
                    for bki in range(2):
                        for bkj in range(2):
                            cb = coeff_B[t][bki][bkj]
                            if cb == 0: continue
                            
                            # Local indices in 2x2 blocks
                            # U local index: bi*2 + x, bj*2 + y
                            # V local index: bki*2 + y, bkj*2 + z
                            # W local index: bkj*2 + z, bi*2 + x ? No.
                            # The product is X*Y.
                            # X[bi,bj] is a 2x2 matrix.
                            # Y[bki,bkj] is a 2x2 matrix.
                            # The term is ca * A[bi,bj] * cb * B[bki,bkj].
                            # This is a product of two 2x2 matrices.
                            # We use the 2x2 Strassen factors U_2, V_2, W_2.
                            # U_2[x,y,r] is the coefficient for A[x,y] in the r-th term.
                            # V_2[y,z,r] is the coefficient for B[y,z] in the r-th term.
                            # W_2[z,x,r] is the coefficient for C[z,x] in the r-th term.
                            
                            for x in range(2):
                                for y in range(2):
                                    u_val = ca * U_2[x*2+y, r]
                                    if u_val != 0:
                                        # Global U index: (bi*2+x)*4 + (bj*2+y)
                                        u_idx = (bi*2+x)*4 + (bj*2+y)
                                        U_4[u_idx, col] += u_val
                                        
                                    for z in range(2):
                                        v_val = cb * V_2[y*2+z, r]
                                        if v_val != 0:
                                            # Global V index: (bki*2+y)*4 + (bkj*2+z)
                                            v_idx = (bki*2+y)*4 + (bkj*2+z)
                                            V_4[v_idx, col] += v_val
                                        
                                        w_val = W_2[z*2+x, r]
                                        if w_val != 0:
                                            # Global W index: (bkj*2+z)*4 + (bi*2+x)
                                            w_idx = (bkj*2+z)*4 + (bi*2+x)
                                            W_4[w_idx, col] += w_val
            rank_offset += 7
            
    # Now combine two 4x4x4 blocks for 4x4x8
    # U is same for both
    # V: left block uses k=0..3, right block uses k=4..7
    # W: left block contributes to k=0..3, right block to k=4..7
    
    U_full = np.zeros((16, 98))
    V_full = np.zeros((32, 98))
    W_full = np.zeros((32, 98))
    
    # Left block (cols 0-3 of B)
    U_full[:, 0:49] = U_4
    # V indices for left block: j in 0..3, k in 0..3
    # V_4 has indices (j*4+k)
    # V_full has indices (j*8+k)
    for j in range(4):
        for k in range(4):
            v_idx_4 = j*4 + k
            v_idx_full = j*8 + k
            V_full[v_idx_full, 0:49] = V_4[v_idx_4, 0:49]
            
    # W indices for left block: k in 0..3, i in 0..3
    # W_4 has indices (k*4+i)
    # W_full has indices (k*4+i)
    W_full[0:16, 0:49] = W_4
    
    # Right block (cols 4-7 of B)
    U_full[:, 49:98] = U_4
    # V indices for right block: j in 0..3, k in 4..7
    for j in range(4):
        for k in range(4):
            v_idx_4 = j*4 + k
            v_idx_full = j*8 + (k+4)
            V_full[v_idx_full, 49:98] = V_4[v_idx_4, 0:49]
            
    # W indices for right block: k in 4..7, i in 0..3
    for k in range(4):
        for i in range(4):
            w_idx_4 = k*4 + i
            w_idx_full = (k+4)*4 + i
            W_full[w_idx_full, 49:98] = W_4[w_idx_4, 0:49]
            
    # Convert to lists of lists
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
