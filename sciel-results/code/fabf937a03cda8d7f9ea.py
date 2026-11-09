import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0  # seconds
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    def verify_decomposition(U, V, W, rank, n, m, p):
        """Check: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += W[k * n + i, r] * U[i * m + j, r] * V[j * p + k, r]
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True

    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank

    def strassen_2x2():
        """
        Return U, V, W for 2x2x2 Strassen with shapes (4,7), (4,7), (4,7).
        U indexed as U[i*2+j], V as V[j*2+k], W as W[k*2+i] for i,j,k in {0,1}.
        Must satisfy: for all (i,j,k) in {0,1}^3, sum_r U[i*2+j,r]*V[j*2+k,r]*W[k*2+i,r] = 1
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen 7 products:
        # M1 = (A00+A11)(B00+B11) -> contributes to C00 and C11
        # M2 = (A10+A11)B00 -> contributes to C10 and C11
        # M3 = A00(B01-B11) -> contributes to C01 and C11
        # M4 = A11(B10-B00) -> contributes to C00 and C10
        # M5 = (A00+A01)B11 -> contributes to C00 and C01
        # M6 = (A10-A00)(B00+B01) -> contributes to C11
        # M7 = (A01-A11)(B10+B11) -> contributes to C00
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # For M1: A-side = A00 + A11, B-side = B00 + B11
        # U[:,0]: U[0,0]=1 (A00), U[3,0]=1 (A11), rest 0
        # V[:,0]: V[0,0]=1 (B00), V[3,0]=1 (B11), rest 0
        # W[:,0]: W[0,0]=1 (C00), W[3,0]=1 (C11), rest 0
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # For M2: A-side = A10 + A11, B-side = B00
        # U[:,1]: U[2,1]=1 (A10), U[3,1]=1 (A11), rest 0
        # V[:,1]: V[0,1]=1 (B00), rest 0
        # W[:,1]: W[2,1]=1 (C10), W[3,1]=-1 (C11), rest 0
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # For M3: A-side = A00, B-side = B01 - B11
        # U[:,2]: U[0,2]=1 (A00), rest 0
        # V[:,2]: V[1,2]=1 (B01), V[3,2]=-1 (B11), rest 0
        # W[:,2]: W[1,2]=1 (C01), W[3,2]=1 (C11), rest 0
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # For M4: A-side = A11, B-side = B10 - B00
        # U[:,3]: U[3,3]=1 (A11), rest 0
        # V[:,3]: V[2,3]=1 (B10), V[0,3]=-1 (B00), rest 0
        # W[:,3]: W[0,3]=1 (C00), W[2,3]=1 (C10), rest 0
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # For M5: A-side = A00 + A01, B-side = B11
        # U[:,4]: U[0,4]=1 (A00), U[1,4]=1 (A01), rest 0
        # V[:,4]: V[3,4]=1 (B11), rest 0
        # W[:,4]: W[0,4]=-1 (C00), W[1,4]=1 (C01), rest 0
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # For M6: A-side = A10 - A00, B-side = B00 + B01
        # U[:,5]: U[2,5]=1 (A10), U[0,5]=-1 (A00), rest 0
        # V[:,5]: V[0,5]=1 (B00), V[1,5]=1 (B01), rest 0
        # W[:,5]: W[3,5]=1 (C11), rest 0
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # For M7: A-side = A01 - A11, B-side = B10 + B11
        # U[:,6]: U[1,6]=1 (A01), U[3,6]=-1 (A11), rest 0
        # V[:,6]: V[2,6]=1 (B10), V[3,6]=1 (B11), rest 0
        # W[:,6]: W[0,6]=1 (C00), rest 0
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def make_blocked_strassen():
        """
        Block structure for 3x4x6:
        A: 3x4, B: 4x6, C: 3x6
        
        Split A rows: [0:2] (2 rows), [2:3] (1 row)
        Split A cols: [0:2] (2 cols), [2:4] (2 cols)
        Split B rows: [0:2] (2 rows), [2:4] (2 rows)
        Split B cols: [0:2] (2 cols), [2:4] (2 cols), [4:6] (2 cols)
        
        For each combination of (a_row_block, a_col_block, b_col_block):
        - a_col_block determines which b_row_block to use
        - If a_row_block is 2 rows and b_col_block is 2 cols: use Strassen (rank 7)
        - If a_row_block is 1 row: use schoolbook (rank 4)
        
        Total rank = 6 * 7 + 6 * 4 = 42 + 24 = 66
        """
        # Get Strassen factors for 2x2x2
        sU, sV, sW = strassen_2x2()
        
        total_rank = 0
        # 6 blocks of type (2,2,2) using Strassen
        # 6 blocks of type (1,2,2) using schoolbook
        total_rank = 6 * 7 + 6 * 4  # 42 + 24 = 66
        
        U = np.zeros((n * m, total_rank), dtype=float)
        V = np.zeros((m * p, total_rank), dtype=float)
        W = np.zeros((p * n, total_rank), dtype=float)
        
        col_offset = 0
        
        # Process all block combinations
        # a_row_blocks: [0:2], [2:3]
        # a_col_blocks: [0:2], [2:4]
        # b_col_blocks: [0:2], [2:4], [4:6]
        
        a_row_blocks = [(0, 2), (2, 3)]
        a_col_blocks = [(0, 2), (2, 4)]
        b_col_blocks = [(0, 2), (2, 4), (4, 6)]
        
        for a_row_start, a_row_end in a_row_blocks:
            a_row_size = a_row_end - a_row_start
            
            for a_col_start, a_col_end in a_col_blocks:
                # b_row_block is determined by a_col_block
                b_row_start = a_col_start
                b_row_end = a_col_end
                
                for b_col_start, b_col_end in b_col_blocks:
                    if a_row_size == 2:
                        # Use Strassen for (2,2,2) block
                        for r_local in range(7):
                            # U: A[a_row_start:a_row_end, a_col_start:a_col_end]
                            # U is indexed as U[i*m+j, col]
                            for i_local in range(2):
                                for j_local in range(2):
                                    if sU[i_local * 2 + j_local, r_local] != 0:
                                        U[(a_row_start + i_local) * m + (a_col_start + j_local), col_offset] = sU[i_local * 2 + j_local, r_local]
                            
                            # V: B[b_row_start:b_row_end, b_col_start:b_col_end]
                            # V is indexed as V[j*p+k, col]
                            for j_local in range(2):
                                for k_local in range(2):
                                    if sV[j_local * 2 + k_local, r_local] != 0:
                                        V[(b_row_start + j_local) * p + (b_col_start + k_local), col_offset] = sV[j_local * 2 + k_local, r_local]
                            
                            # W: C[b_col_start:b_col_end, a_row_start:a_row_end]
                            # W is indexed as W[k*n+i, col]
                            for k_local in range(2):
                                for i_local in range(2):
                                    if sW[k_local * 2 + i_local, r_local] != 0:
                                        W[(b_col_start + k_local) * n + (a_row_start + i_local), col_offset] = sW[k_local * 2 + i_local, r_local]
                            
                            col_offset += 1
                    else:
                        # a_row_size == 1, use schoolbook for (1,2,2) block
                        # Rank = 1 * 2 * 2 = 4
                        for j_local in range(2):
                            for k_local in range(2):
                                # U: A[a_row_start, a_col_start+j_local]
                                U[a_row_start * m + (a_col_start + j_local), col_offset] = 1.0
                                # V: B[b_row_start+j_local, b_col_start+k_local]
                                V[(b_row_start + j_local) * p + (b_col_start + k_local), col_offset] = 1.0
                                # W: C[b_col_start+k_local, a_row_start]
                                W[(b_col_start + k_local) * n + a_row_start, col_offset] = 1.0
                                col_offset += 1
        
        return U, V, W, total_rank

    # Start with schoolbook
    best_U, best_V, best_W, best_rank = make_schoolbook()
    
    if time.time() - start_time > deadline - 10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try blocked Strassen
    try:
        bU, bV, bW, b_rank = make_blocked_strassen()
        if verify_decomposition(bU, bV, bW, b_rank, n, m, p):
            if b_rank < best_rank:
                best_U, best_V, best_W, best_rank = bU, bV, bW, b_rank
        else:
            # Verification failed, keep schoolbook
            pass
    except Exception:
        pass
    
    if time.time() - start_time > deadline - 5:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
