import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # For each (i,j,k), check sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                s = np.sum(u_val * v_val * w_val)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _get_strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    Uses the standard Strassen decomposition.
    """
    # Standard Strassen 2x2x2 rank-7 decomposition
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M4 + M5
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[3, 3] = -1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[1, 6] = -1
    
    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    try:
        # Get verified 2x2x2 Strassen decomposition
        U_222, V_222, W_222 = _get_strassen_2x2x2()
        
        if not _verify(U_222.tolist(), V_222.tolist(), W_222.tolist(), 2, 2, 2):
            raise ValueError("2x2x2 Strassen verification failed")
        
        # Construct 2x2x4 using two copies of 2x2x2
        U_224 = np.zeros((4, 14), dtype=int)
        V_224 = np.zeros((8, 14), dtype=int)
        W_224 = np.zeros((8, 14), dtype=int)
        
        U_224[:, 0:7] = U_222
        V_224[0:4, 0:7] = V_222
        W_224[0:4, 0:7] = W_222
        
        U_224[:, 7:14] = U_222
        V_224[4:8, 7:14] = V_222
        W_224[4:8, 7:14] = W_222
        
        if not _verify(U_224.tolist(), V_224.tolist(), W_224.tolist(), 2, 2, 4):
            raise ValueError("2x2x4 verification failed")
        
        # Now construct 4x4x8 by applying Strassen at the block level
        # A is 4x4, split into 2x2 blocks of 2x2
        # B is 4x8, split into 2x2 blocks of 2x4
        # C is 4x8, split into 2x2 blocks of 2x4
        #
        # Use Strassen formula:
        # M1 = (A00+A11)(B00+B11)  -> C00 += M1, C11 += M1
        # M2 = (A10+A11)B00        -> C10 += M2
        # M3 = A00(B01-B11)        -> C01 += M3
        # M4 = A11(B10-B00)        -> C11 -= M4
        # M5 = (A00+A01)B11        -> C00 += M5, C01 += M5
        # M6 = (A10-A00)(B00+B01)  -> C10 += M6
        # M7 = (A01-A11)(B10+B11)  -> C01 -= M7
        #
        # Wait, this is the same issue. Let me use the correct Strassen output:
        # C00 = M1 - M4 + M5
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3
        
        # Each block multiplication is 2x2 * 2x4 = 2x4, which uses 14 rank terms
        # So total rank = 7 * 14 = 98
        
        U = np.zeros((16, 98), dtype=int)
        V = np.zeros((32, 98), dtype=int)
        W = np.zeros((32, 98), dtype=int)
        
        # Define the 7 Strassen terms with their block coefficients
        # Each term: (A_block_coeffs, B_block_coeffs, C_block_coeffs)
        # A blocks: (row_block, col_block) where row_block in {0,1}, col_block in {0,1}
        # B blocks: (row_block, col_block) where row_block in {0,1}, col_block in {0,1}
        # C blocks: (row_block, col_block) where row_block in {0,1}, col_block in {0,1}
        
        strassen_terms = [
            # M1 = (A00+A11)(B00+B11)
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # M2 = (A10+A11)B00
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
            # M3 = A00(B01-B11)
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}),
            # M4 = A11(B10-B00)
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(1,1): -1}),
            # M5 = (A00+A01)B11
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): 1, (0,1): 1}),
            # M6 = (A10-A00)(B00+B01)
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,0): 1}),
            # M7 = (A01-A11)(B10+B11)
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,1): -1}),
        ]
        
        for term_idx, (a_coeffs, b_coeffs, c_coeffs) in enumerate(strassen_terms):
            col_start = term_idx * 14
            
            # Add A contributions to U
            for (a_row, a_col), coeff in a_coeffs.items():
                # A block (a_row, a_col) is a 2x2 block
                # Global row: a_row * 4 + local_row (0 or 1)
                # Global col: a_col * 2 + local_col (0 or 1)
                # U row index: global_row * m + global_col = (a_row*4 + lr) * 4 + (a_col*2 + lc)
                # = a_row*16 + lr*4 + a_col*2 + lc
                for lr in range(2):
                    for lc in range(2):
                        u_row = a_row * 16 + lr * 4 + a_col * 2 + lc
                        U[u_row, col_start:col_start+14] += coeff * U_224[lr*2+lc]
            
            # Add B contributions to V
            for (b_row, b_col), coeff in b_coeffs.items():
                # B block (b_row, b_col) is a 2x4 block
                # Global row: b_row * 4 + local_row (0 or 1)
                # Global col: b_col * 4 + local_col (0,1,2,3)
                # V row index: global_row * p + global_col = (b_row*4 + lr) * 8 + (b_col*4 + lc)
                # = b_row*32 + lr*8 + b_col*4 + lc
                for lr in range(2):
                    for lc in range(4):
                        v_row = b_row * 32 + lr * 8 + b_col * 4 + lc
                        V[v_row, col_start:col_start+14] += coeff * V_224[lr*4+lc]
            
            # Add C contributions to W
            for (c_row, c_col), coeff in c_coeffs.items():
                # C block (c_row, c_col) is a 2x4 block
                # Global row: c_row * 4 + local_row (0 or 1)
                # Global col: c_col * 4 + local_col (0,1,2,3)
                # W row index: global_col * n + global_row = (c_col*4 + lc) * 4 + (c_row*4 + lr)
                # = c_col*16 + lc*4 + c_row*4 + lr
                for lr in range(2):
                    for lc in range(4):
                        w_row = c_col * 16 + lc * 4 + c_row * 4 + lr
                        W[w_row, col_start:col_start+14] += coeff * W_224[lc*2+lr]
        
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            raise ValueError("4x4x8 Strassen verification failed")
            
    except Exception as e:
        # Fall back to schoolbook
        return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
