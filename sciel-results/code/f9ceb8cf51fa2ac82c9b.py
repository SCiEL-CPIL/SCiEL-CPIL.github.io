import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m  # 8
    V_dim = m * p  # 32
    W_dim = p * n  # 16

    start_time = time.time()
    deadline = start_time + 135

    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    def check_decomp(U, V, W, tol=1e-8):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Ur = U[i * m + j, :]
                    Vr = V[j * p + k, :]
                    Wr = W[k * n + i, :]
                    val = np.dot(Ur * Vr, Wr)
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def make_strassen_block():
        """
        Strassen for 2x2 * 2x2 -> 2x2 with 7 terms.
        Returns (U_block, V_block, W_block) each of shape (4, 7)
        Indexing:
          U: [i*m + j] where m=2, so U[i*2+j, t]
          V: [j*p + k] where p=2, so V[j*2+k, t]
          W: [k*n + i] where n=2, so W[k*2+i, t]
        """
        # Strassen terms for 2x2:
        # M1 = (A11-A22)(B21+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A22)(B11+B22)
        # M6 = (A12-A22)(B21+B22)
        # M7 = (A21-A11)(B11+B12)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M5 + M6
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # A indices: A[i,j] -> U[i*2+j]
        #   A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[2], A[1,1]=U[3]
        # B indices: B[j,k] -> V[j*2+k]
        #   B[0,0]=V[0], B[0,1]=V[1], B[1,0]=V[2], B[1,1]=V[3]
        # C indices: C[i,k] -> W[k*2+i]
        #   C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[2], C[1,1]=W[3]
        
        # M1: (A11-A22)(B21+B22)
        U[3, 0] = 1  # A[1,1]
        U[1, 0] = -1  # A[0,1]... wait, A11 is A[0,0] in 0-indexed? 
        # Let me use 0-indexed: A[i,j] where i,j in {0,1}
        # A11 in 1-indexed = A[0,0] in 0-indexed
        # A22 in 1-indexed = A[1,1] in 0-indexed
        # B21 in 1-indexed = B[1,0] in 0-indexed
        # B22 in 1-indexed = B[1,1] in 0-indexed
        
        # M1 = (A[0,0] - A[1,1]) * (B[1,0] + B[1,1])
        U[0, 0] = 1
        U[3, 0] = -1
        V[2, 0] = 1  # B[1,0]
        V[3, 0] = 1  # B[1,1]
        # C contribution: C[0,0] += M1, C[1,1] += M1
        W[0, 0] = 1  # C[0,0]
        W[3, 0] = 1  # C[1,1]
        
        # M2 = (A[1,0] + A[1,1]) * B[0,0]
        U[2, 1] = 1
        U[3, 1] = 1
        V[0, 1] = 1
        # C[1,0] += M2
        W[2, 1] = 1  # C[1,0]
        
        # M3 = A[0,0] * (B[0,1] - B[1,1])
        U[0, 2] = 1
        V[1, 2] = 1
        V[3, 2] = -1
        # C[0,1] += M3, C[1,1] -= M3
        W[1, 2] = 1  # C[0,1]
        W[3, 2] = -1  # C[1,1]
        
        # M4 = A[1,1] * (B[1,0] - B[0,0])
        U[3, 3] = 1
        V[2, 3] = 1
        V[0, 3] = -1
        # C[0,0] += M4, C[1,0] += M4
        W[0, 3] = 1
        W[2, 3] = 1
        
        # M5 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
        U[0, 4] = 1
        U[3, 4] = 1
        V[0, 4] = 1
        V[3, 4] = 1
        # C[0,0] -= M5, C[0,1] += M5, C[1,1] += M5
        W[0, 4] = -1
        W[1, 4] = 1
        W[3, 4] = 1
        
        # M6 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
        U[1, 5] = 1
        U[3, 5] = -1
        V[2, 5] = 1
        V[3, 5] = 1
        # C[1,1] += M6
        W[3, 5] = 1
        
        # M7 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
        U[2, 6] = 1
        U[0, 6] = -1
        V[0, 6] = 1
        V[1, 6] = 1
        # C[0,0] += M7
        W[0, 6] = 1
        
        return U, V, W

    U_block, V_block, W_block = make_strassen_block()
    
    # Now build the full 2x4 * 4x8 using 4 Strassen blocks.
    # A is 2x4: split into two 2x2 blocks: A0 (cols 0,1) and A1 (cols 2,3)
    # B is 4x8: split rows into B_top (rows 0,1) and B_bot (rows 2,3)
    # B_top is 2x8, B_bot is 2x8
    # C = A0 * B_top + A1 * B_bot, where each product is 2x8
    
    # For A0 * B_top: B_top is 2x8, treat as four 2x2 blocks: B_top[:, 0:2], B_top[:, 2:4], B_top[:, 4:6], B_top[:, 6:8]
    # For A1 * B_bot: similarly four 2x2 blocks
    
    # Total: 8 Strassen blocks, each with rank 7, total rank = 56
    
    rank_total = 56
    U_full = np.zeros((U_dim, rank_total))
    V_full = np.zeros((V_dim, rank_total))
    W_full = np.zeros((W_dim, rank_total))
    
    col_offset = 0
    for a_block in range(2):  # 0: cols 0-1 of A, 1: cols 2-3
        a_col_base = a_block * 2
        b_row_base = a_block * 2  # B_top for a_block=0, B_bot for a_block=1
        for c_block in range(4):  # 4 blocks of 2 columns in the 8-col output
            c_col_base = c_block * 2
            
            for t in range(7):
                # U: A[i, a_col_base + j]
                for i in range(2):
                    for j in range(2):
                        U_full[i * 4 + a_col_base + j, col_offset] += U_block[i * 2 + j, t]
                
                # V: B[b_row_base + j, c_col_base + k]
                for j in range(2):
                    for k in range(2):
                        V_full[(b_row_base + j) * 8 + c_col_base + k, col_offset] += V_block[j * 2 + k, t]
                
                # W: C[i, c_col_base + k] -> W[k*n + i] = W[k*2 + i]
                for i in range(2):
                    for k in range(2):
                        W_full[(c_col_base + k) * 2 + i, col_offset] += W_block[k * 2 + i, t]
                
                col_offset += 1
    
    # Verify
    if check_decomp(U_full, V_full, W_full, tol=1e-10):
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback to schoolbook
    U0, V0, W0, r0 = make_schoolbook()
    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
# EVOLVE-BLOCK-END
