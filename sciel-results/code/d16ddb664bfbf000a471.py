import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def _get_strassen_2x2_2x2():
    """
    Returns U, V, W for 2x2 x 2x2 -> 2x2, rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, t], V[j*2+k, t], W[k*2+i, t]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Term 0: (A00+A11)(B00+B11) -> C00+=1, C11+=1
    # U: A00(0), A11(3)
    # V: B00(0), B11(3)
    # W: C00(0), C11(3)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    
    # Term 1: (A10+A11)B00 -> C10+=1, C11-=1
    # U: A10(2), A11(3)
    # V: B00(0)
    # W: C10(1), C11(3)
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[1,1]=1; W[3,1]=-1
    
    # Term 2: A00(B01-B11) -> C01+=1, C11+=1
    # U: A00(0)
    # V: B01(1), B11(3)
    # W: C01(2), C11(3)
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[2,2]=1; W[3,2]=1
    
    # Term 3: A11(B10-B00) -> C10+=1, C00+=1
    # U: A11(3)
    # V: B10(2), B00(0)
    # W: C10(1), C00(0)
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[1,3]=1; W[0,3]=1
    
    # Term 4: (A00+A01)B11 -> C01-=1, C11+=1
    # U: A00(0), A01(1)
    # V: B11(3)
    # W: C01(2), C11(3)
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[2,4]=-1; W[3,4]=1
    
    # Term 5: (A10-A00)(B00+B01) -> C11+=1
    # U: A10(2), A00(0)
    # V: B00(0), B01(1)
    # W: C11(3)
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[3,5]=1
    
    # Term 6: (A01-A11)(B10+B11) -> C00+=1
    # U: A01(1), A11(3)
    # V: B10(2), B11(3)
    # W: C00(0)
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=1
    
    return U, V, W

def _get_decomp_2x2_2x3():
    """
    Returns U, V, W for 2x2 x 2x3 -> 2x3, rank 11.
    U: (4, 11), V: (6, 11), W: (6, 11)
    Indexing: U[i*2+j, t], V[j*3+k, t], W[k*2+i, t]
    """
    U = np.zeros((4, 11))
    V = np.zeros((6, 11))
    W = np.zeros((6, 11))
    
    # Part A: 2x2 x 2x2 (cols 0-1 of B)
    UA, VA, WA = _get_strassen_2x2_2x2()
    
    # Map U (same indices)
    U[:, 0:7] = UA
    
    # Map V: j*3+k. k in {0,1}.
    # j=0: k=0->0, k=1->1.
    # j=1: k=0->3, k=1->4.
    # VA indices: 0,1,2,3 correspond to j=0,k=0; j=0,k=1; j=1,k=0; j=1,k=1.
    # Wait, VA is (4,7). Indices 0,1,2,3.
    # VA[0] -> j=0,k=0 -> V[0]
    # VA[1] -> j=0,k=1 -> V[1]
    # VA[2] -> j=1,k=0 -> V[3]
    # VA[3] -> j=1,k=1 -> V[4]
    V[0, 0:7] = VA[0, :]
    V[1, 0:7] = VA[1, :]
    V[3, 0:7] = VA[2, :]
    V[4, 0:7] = VA[3, :]
    
    # Map W: k*2+i. k in {0,1}.
    # k=0: i=0->0, i=1->1.
    # k=1: i=0->2, i=1->3.
    # WA indices: 0,1,2,3 correspond to k=0,i=0; k=0,i=1; k=1,i=0; k=1,i=1.
    # WA[0] -> W[0]
    # WA[1] -> W[1]
    # WA[2] -> W[2]
    # WA[3] -> W[3]
    W[0, 0:7] = WA[0, :]
    W[1, 0:7] = WA[1, :]
    W[2, 0:7] = WA[2, :]
    W[3, 0:7] = WA[3, :]
    
    # Part B: 2x2 x 1x2 (col 2 of B)
    # Schoolbook for 2x2 x 1x2 -> 2x1
    # n=2, m=2, p=1.
    # U_B (4,4), V_B (2,4), W_B (2,4)
    # U_B[i*2+j, t], V_B[j*1+0, t], W_B[0*2+i, t]
    
    # Term 7: A00 * B02
    # U: A00(0)
    # V: B02 (j=0, k=2) -> V[0*3+2]=V[2]
    # W: C02 (i=0, k=2) -> W[2*2+0]=W[4]
    U[0, 7] = 1
    V[2, 7] = 1
    W[4, 7] = 1
    
    # Term 8: A01 * B12
    # U: A01(1)
    # V: B12 (j=1, k=2) -> V[1*3+2]=V[5]
    # W: C02 (i=0, k=2) -> W[4]
    U[1, 8] = 1
    V[5, 8] = 1
    W[4, 8] = 1
    
    # Term 9: A10 * B02
    # U: A10(2)
    # V: B02 (j=0, k=2) -> V[2]
    # W: C12 (i=1, k=2) -> W[2*2+1]=W[5]
    U[2, 9] = 1
    V[2, 9] = 1
    W[5, 9] = 1
    
    # Term 10: A11 * B12
    # U: A11(3)
    # V: B12 (j=1, k=2) -> V[5]
    # W: C12 (i=1, k=2) -> W[5]
    U[3, 10] = 1
    V[5, 10] = 1
    W[5, 10] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Build the 4x4 x 4x5 decomposition
    # Rank 72
    total_rank = 72
    U = np.zeros((16, total_rank))
    V = np.zeros((20, total_rank))
    W = np.zeros((20, total_rank))
    
    U22, V22, W22 = _get_strassen_2x2_2x2()
    U23, V23, W23 = _get_decomp_2x2_2x3()
    
    # Blocks:
    # A: 2x2 blocks of 2x2.
    # B: 2x2 blocks of 2x3 (cols 0-1 are 2x2, cols 2-4 are 2x3? No, B is 4x5.
    # Rows 0-1: B00 (2x2, cols 0-1), B01 (2x3, cols 2-4).
    # Rows 2-3: B10 (2x2, cols 0-1), B11 (2x3, cols 2-4).
    
    # C: 2x2 blocks of 2x3.
    # Rows 0-1: C00 (2x2, cols 0-1), C01 (2x3, cols 2-4).
    # Rows 2-3: C10 (2x2, cols 0-1), C11 (2x3, cols 2-4).
    
    # Multiplications:
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    
    # We need to place the decompositions into U, V, W.
    
    # Helper to place a block decomposition
    def place_block(U_block, V_block, W_block, offset, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        # a_row_start, a_col_start: top-left of A block (2x2)
        # b_row_start, b_col_start: top-left of B block
        # c_row_start, c_col_start: top-left of C block
        
        # U indices: i*m + j.
        # A block is at rows a_row_start..a_row_start+1, cols a_col_start..a_col_start+1.
        # Local i in 0..1, j in 0..1.
        # Global I = a_row_start + i, J = a_col_start + j.
        # Global U index = I*4 + J.
        
        for i in range(2):
            for j in range(2):
                local_idx = i*2 + j
                global_u_idx = (a_row_start + i)*4 + (a_col_start + j)
                U[global_u_idx, offset:offset+U_block.shape[1]] = U_block[local_idx, :]
                
        # V indices: j*p + k.
        # B block dimensions depend on type.
        # If B block is 2x2 (cols 0-1 of 4x5 B):
        #   b_col_start is 0 or 2?
        #   If b_col_start=0, cols are 0,1.
        #   If b_col_start=2, cols are 2,3,4? No, 2x3 block.
        
        # Let's handle 2x2 and 2x3 separately.
        
    # We will manually place each of the 8 terms.
    
    # 1. A00 B00 (2x2 x 2x2)
    # A00: rows 0-1, cols 0-1.
    # B00: rows 0-1, cols 0-1.
    # C00: rows 0-1, cols 0-1.
    off = 0
    r = 7
    # U
    for i in range(2):
        for j in range(2):
            U[(0+i)*4 + (0+j), off:off+r] = U22[i*2+j, :]
    # V
    for j in range(2):
        for k in range(2):
            V[(0+j)*5 + (0+k), off:off+r] = V22[j*2+k, :]
    # W
    for k in range(2):
        for i in range(2):
            W[(0+k)*4 + (0+i), off:off+r] = W22[k*2+i, :]
            
    # 2. A01 B10 (2x2 x 2x2)
    # A01: rows 0-1, cols 2-3.
    # B10: rows 2-3, cols 0-1.
    # C00: rows 0-1, cols 0-1.
    off = 7
    r = 7
    for i in range(2):
        for j in range(2):
            U[(0+i)*4 + (2+j), off:off+r] = U22[i*2+j, :]
    for j in range(2):
        for k in range(2):
            V[(2+j)*5 + (0+k), off:off+r] = V22[j*2+k, :]
    for k in range(2):
        for i in range(2):
            W[(0+k)*4 + (0+i), off:off+r] = W22[k*2+i, :]
            
    # 3. A00 B01 (2x2 x 2x3)
    # A00: rows 0-1, cols 0-1.
    # B01: rows 0-1, cols 2-4.
    # C01: rows 0-1, cols 2-4.
    off = 14
    r = 11
    for i in range(2):
        for j in range(2):
            U[(0+i)*4 + (0+j), off:off+r] = U23[i*2+j, :]
    # V23 indices: j*3+k.
    # B01 global indices: (0+j)*5 + (2+k).
    for j in range(2):
        for k in range(3):
            V[(0+j)*5 + (2+k), off:off+r] = V23[j*3+k, :]
    # W23 indices: k*2+i.
    # C01 global indices: (2+k)*4 + (0+i).
    for k in range(3):
        for i in range(2):
            W[(2+k)*4 + (0+i), off:off+r] = W23[k*2+i, :]
            
    # 4. A01 B11 (2x2 x 2x3)
    # A01: rows 0-1, cols 2-3.
    # B11: rows 2-3, cols 2-4.
    # C01: rows 0-1, cols 2-4.
    off = 25
    r = 11
    for i in range(2):
        for j in range(2):
            U[(0+i)*4 + (2+j), off:off+r] = U23[i*2+j, :]
    for j in range(2):
        for k in range(3):
            V[(2+j)*5 + (2+k), off:off+r] = V23[j*3+k, :]
    for k in range(3):
        for i in range(2):
            W[(2+k)*4 + (0+i), off:off+r] = W23[k*2+i, :]
            
    # 5. A10 B00 (2x2 x 2x2)
    # A10: rows 2-3, cols 0-1.
    # B00: rows 0-1, cols 0-1.
    # C10: rows 2-3, cols 0-1.
    off = 36
    r = 7
    for i in range(2):
        for j in range(2):
            U[(2+i)*4 + (0+j), off:off+r] = U22[i*2+j, :]
    for j in range(2):
        for k in range(2):
            V[(0+j)*5 + (0+k), off:off+r] = V22[j*2+k, :]
    for k in range(2):
        for i in range(2):
            W[(0+k)*4 + (2+i), off:off+r] = W22[k*2+i, :]
            
    # 6. A11 B10 (2x2 x 2x2)
    # A11: rows 2-3, cols 2-3.
    # B10: rows 2-3, cols 0-1.
    # C10: rows 2-3, cols 0-1.
    off = 43
    r = 7
    for i in range(2):
        for j in range(2):
            U[(2+i)*4 + (2+j), off:off+r] = U22[i*2+j, :]
    for j in range(2):
        for k in range(2):
            V[(2+j)*5 + (0+k), off:off+r] = V22[j*2+k, :]
    for k in range(2):
        for i in range(2):
            W[(0+k)*4 + (2+i), off:off+r] = W22[k*2+i, :]
            
    # 7. A10 B01 (2x2 x 2x3)
    # A10: rows 2-3, cols 0-1.
    # B01: rows 0-1, cols 2-4.
    # C11: rows 2-3, cols 2-4.
    off = 50
    r = 11
    for i in range(2):
        for j in range(2):
            U[(2+i)*4 + (0+j), off:off+r] = U23[i*2+j, :]
    for j in range(2):
        for k in range(3):
            V[(0+j)*5 + (2+k), off:off+r] = V23[j*3+k, :]
    for k in range(3):
        for i in range(2):
            W[(2+k)*4 + (2+i), off:off+r] = W23[k*2+i, :]
            
    # 8. A11 B11 (2x2 x 2x3)
    # A11: rows 2-3, cols 2-3.
    # B11: rows 2-3, cols 2-4.
    # C11: rows 2-3, cols 2-4.
    off = 61
    r = 11
    for i in range(2):
        for j in range(2):
            U[(2+i)*4 + (2+j), off:off+r] = U23[i*2+j, :]
    for j in range(2):
        for k in range(3):
            V[(2+j)*5 + (2+k), off:off+r] = V23[j*3+k, :]
    for k in range(3):
        for i in range(2):
            W[(2+k)*4 + (2+i), off:off+r] = W23[k*2+i, :]
            
    # Verify
    def _verify(U, V, W, n, m, p):
        r = U.shape[1]
        res = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    res[i,j,k] = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
        return np.allclose(res, 1.0, atol=1e-9)
    
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook if construction failed
        rank_school = n * m * p
        U_sb = np.zeros((n*m, rank_school))
        V_sb = np.zeros((m*p, rank_school))
        W_sb = np.zeros((p*n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {
            "U": [[float(val) for val in row] for row in U_sb],
            "V": [[float(val) for val in row] for row in V_sb],
            "W": [[float(val) for val in row] for row in W_sb]
        }
        
    return {
        "U": [[float(val) for val in row] for row in U],
        "V": [[float(val) for val in row] for row in V],
        "W": [[float(val) for val in row] for row in W]
    }
