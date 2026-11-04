"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    _start = time.time()
    _deadline = _start + 140.0  # 140s deadline, 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    def verify(U, V, W, tol=1e-9):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = U_arr.shape[1]
        
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
                    if abs(s - 1.0) > max_err:
                        max_err = abs(s - 1.0)
        
        return max_err < tol
    
    def schoolbook():
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
    
    def strassen_2x2x2():
        """
        Strassen's 7-rank decomposition for 2x2 * 2x2 matrix multiplication.
        Returns U, V, W arrays of shape (4, 7), (4, 7), (4, 7) respectively.
        U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        # Strassen's 7 products:
        # M1 = (A11)(B22 - B21)
        # M2 = (A11 + A12)(B22)
        # M3 = (A21 + A22)(B11)
        # M4 = (A22)(B12 - B11)
        # M5 = (A11 + A21)(B11 + B12)
        # M6 = (A12 - A21)(B21 + B22)
        # M7 = (A11 - A22)(B11 + B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M2 + M3
        # C21 = M3 + M6
        # C22 = M1 - M2 + M5 + M6
        
        # U: (2,2,7) -> (4,7) with U[i*2+j, r]
        # V: (2,2,7) -> (4,7) with V[j*2+k, r]
        # W: (2,2,7) -> (4,7) with W[k*2+i, r]
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1: A11 * (B22 - B21)
        U[0, 0] = 1  # A[0,0]
        V[3, 0] = 1  # B[1,1]
        V[2, 0] = -1  # B[1,0]
        W[0, 0] = 1  # C[0,0] contribution: +1
        W[3, 0] = 1  # C[1,1] contribution: +1
        
        # M2: (A11 + A12) * B22
        U[0, 1] = 1  # A[0,0]
        U[1, 1] = 1  # A[0,1]
        V[3, 1] = 1  # B[1,1]
        W[0, 1] = 1  # C[0,0]: +1
        W[1, 1] = 1  # C[0,1]: +1
        
        # M3: (A21 + A22) * B11
        U[2, 2] = 1  # A[1,0]
        U[3, 2] = 1  # A[1,1]
        V[0, 2] = 1  # B[0,0]
        W[1, 2] = 1  # C[0,1]: +1
        W[2, 2] = 1  # C[1,0]: +1
        
        # M4: A22 * (B12 - B11)
        U[3, 3] = 1  # A[1,1]
        V[1, 3] = 1  # B[0,1]
        V[0, 3] = -1  # B[0,0]
        W[0, 3] = 1  # C[0,0]: +1
        W[3, 3] = -1  # C[1,1]: -1
        
        # M5: (A11 + A21) * (B11 + B12)
        U[0, 4] = 1  # A[0,0]
        U[2, 4] = 1  # A[1,0]
        V[0, 4] = 1  # B[0,0]
        V[1, 4] = 1  # B[0,1]
        W[0, 4] = -1  # C[0,0]: -1
        W[3, 4] = 1  # C[1,1]: +1
        
        # M6: (A12 - A21) * (B21 + B22)
        U[1, 5] = 1  # A[0,1]
        U[2, 5] = -1  # A[1,0]
        V[2, 5] = 1  # B[1,0]
        V[3, 5] = 1  # B[1,1]
        W[2, 5] = 1  # C[1,0]: +1
        W[3, 5] = 1  # C[1,1]: +1
        
        # M7: (A11 - A22) * (B11 + B22)
        U[0, 6] = 1  # A[0,0]
        U[3, 6] = -1  # A[1,1]
        V[0, 6] = 1  # B[0,0]
        V[3, 6] = 1  # B[1,1]
        W[0, 6] = 1  # C[0,0]: +1
        W[3, 6] = -1  # C[1,1]: -1
        
        return U, V, W
    
    def rank_decomp_2x2x3():
        """
        Find a low-rank decomposition for 2x2 * 2x3 matrix multiplication.
        Schoolbook rank = 2*2*3 = 12.
        We'll try to find a rank < 12 decomposition.
        """
        nn, mm, pp = 2, 2, 3
        # Try random search for rank 10, 9, 8
        np.random.seed(123)
        
        for target_rank in [10, 9, 8]:
            for trial in range(200):
                if time.time() > _deadline - 20:
                    break
                
                U = np.random.randn(nn * mm, target_rank) * 0.5
                V = np.random.randn(mm * pp, target_rank) * 0.5
                W = np.random.randn(pp * nn, target_rank) * 0.5
                
                best_err = float('inf')
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                for step in range(50):
                    if time.time() > _deadline - 20:
                        break
                    
                    U_3d = U.reshape(nn, mm, target_rank)
                    V_3d = V.reshape(mm, pp, target_rank)
                    W_3d = W.reshape(pp, nn, target_rank)
                    
                    err = 0.0
                    for i in range(nn):
                        for j in range(mm):
                            for k in range(pp):
                                s = 0.0
                                for r in range(target_rank):
                                    s += U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
                                err += (s - 1.0) ** 2
                    
                    if err < best_err:
                        best_err = err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    
                    if best_err < 1e-9:
                        return best_U, best_V, best_W, target_rank
                    
                    # Gradient-like update
                    lr = 0.05
                    U += lr * np.random.randn(nn * mm, target_rank)
                    V += lr * np.random.randn(mm * pp, target_rank)
                    W += lr * np.random.randn(pp * nn, target_rank)
                
                if best_err < 1e-9:
                    return best_U, best_V, best_W, target_rank
        
        # Fall back to schoolbook for 2x2x3
        rank = nn * mm * pp
        U = np.zeros((nn * mm, rank))
        V = np.zeros((mm * pp, rank))
        W = np.zeros((pp * nn, rank))
        col = 0
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    U[i * mm + j, col] = 1
                    V[j * pp + k, col] = 1
                    W[k * nn + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    # Strategy: Split m=4 into 2+2, and p=5 into 2+3
    # The 4-dimensional inner product space is split into two 2-dim subspaces.
    # For each output column group, we handle the multiplication.
    #
    # A is 2x4, B is 4x5.
    # Split A into A1 (2x2) and A2 (2x2) along columns.
    # Split B into B1 (4x2) and B2 (4x3) along columns.
    # C = A*B = A1*B1_top + A2*B2_top for first 2 cols, etc.
    #
    # Actually, let's think more carefully.
    # A = [A_left | A_right] where A_left is 2x2, A_right is 2x2
    # B has 4 rows, 5 cols.
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    #        = sum_{j=0}^{1} A[i,j] * B[j,k] + sum_{j=2}^{3} A[i,j] * B[j,k]
    #
    # So C = A_left * B_top + A_right * B_bottom
    # where B_top is 2x5, B_bottom is 2x5.
    #
    # This is a sum of two 2x2 * 2x5 multiplications.
    # Each 2x2 * 2x5 can be decomposed further.
    # 2x2 * 2x5 = 2x2 * (2x2 + 2x3)
    # = (2x2 * 2x2) + (2x2 * 2x3)
    #
    # So total = 2 * (Strassen_2x2x2 + decomp_2x2x3)
    
    # Get Strassen decomposition for 2x2x2
    U_s, V_s, W_s = strassen_2x2x2()
    
    # Get decomposition for 2x2x3
    U_223, V_223, W_223, rank_223 = rank_decomp_2x2x3()
    
    # Now construct the full decomposition.
    # We have two copies: one for A_left * B_top, one for A_right * B_bottom.
    # For each copy, we split the 5 output columns into 2+3.
    
    total_rank = 2 * (7 + rank_223)
    
    U_full = np.zeros((n * m, total_rank))  # (8, total_rank)
    V_full = np.zeros((m * p, total_rank))  # (20, total_rank)
    W_full = np.zeros((p * n, total_rank))  # (10, total_rank)
    
    col_offset = 0
    
    for block in range(2):
        # block 0: A_left (cols 0,1) * B_top (rows 0,1)
        # block 1: A_right (cols 2,3) * B_bottom (rows 2,3)
        
        # For this block, we use Strassen for the first 2 output cols and rank_223 for the last 3.
        
        # Strassen part (2x2 * 2x2):
        # U_s has shape (4, 7), indexed by [i*2+j, r] where i in 0..1, j in 0..1
        # But our A has 4 columns total. For block 0, A_left uses cols 0,1.
        # For block 1, A_right uses cols 2,3.
        
        a_col_offset = block * 2  # 0 or 2
        
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    val = U_s[i * 2 + j, r]
                    if val != 0:
                        U_full[i * m + (a_col_offset + j), col_offset + r] = val
            
            for j in range(2):
                for k in range(2):
                    val = V_s[j * 2 + k, r]
                    if val != 0:
                        # B_top row j corresponds to global row (block*2 + j)
                        V_full[(block * 2 + j) * p + k, col_offset + r] = val
            
            for k in range(2):
                for i in range(2):
                    val = W_s[k * 2 + i, r]
                    if val != 0:
                        W_full[k * n + i, col_offset + r] = val
        
        col_offset += 7
        
        # 2x2x3 part:
        # U_223 has shape (4, rank_223), indexed by [i*2+j, r]
        for r in range(rank_223):
            for i in range(2):
                for j in range(2):
                    val = U_223[i * 2 + j, r]
                    if val != 0:
                        U_full[i * m + (a_col_offset + j), col_offset + r] = val
            
            for j in range(2):
                for k in range(3):
                    val = V_223[j * 3 + k, r]
                    if val != 0:
                        # B_top row j, output col k (but k is 0..2, global col is 2+k)
                        V_full[(block * 2 + j) * p + (2 + k), col_offset + r] = val
            
            for k in range(3):
                for i in range(2):
                    val = W_223[k * 2 + i, r]
                    if val != 0:
                        # Output col k maps to global col 2+k
                        W_full[(2 + k) * n + i, col_offset + r] = val
        
        col_offset += rank_223
    
    # Verify the construction
    if verify(U_full.tolist(), V_full.tolist(), W_full.tolist()):
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # If verification fails, fall back to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
