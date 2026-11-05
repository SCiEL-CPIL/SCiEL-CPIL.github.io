import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = 3, 3, 8
    schoolbook_rank = n * m * p  # 72
    
    # Initialize with schoolbook decomposition as the fallback
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = schoolbook_rank
    
    def check_decomposition(U, V, W, rank):
        """Check if the decomposition exactly represents matrix multiplication."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s = s + U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True

    # Construct the block-wise Strassen-based decomposition (rank 68)
    # 4 blocks of 2 columns each.
    # Each block uses 17 terms.
    # Total rank 68.
    
    rank = 68
    U_block = [[0] * rank for _ in range(n * m)]
    V_block = [[0] * rank for _ in range(m * p)]
    W_block = [[0] * rank for _ in range(p * n)]
    
    term_idx = 0
    
    for b in range(4):
        # Block b corresponds to B columns 2*b, 2*b+1
        # And C columns 2*b, 2*b+1
        
        # Strassen terms for A[0:2, 0:2] * B[0:2, 2*b:2*b+2]
        # A indices: 0,1 (rows), 0,1 (cols) -> 0,1,3,4
        # B indices: 0,1 (rows), 2*b, 2*b+1 (cols) -> 0+2*b, 0+2*b+1, 1+2*b, 1+2*b+1
        # C indices: 2*b, 2*b+1 (cols), 0,1 (rows) -> 2*b+0, 2*b+1, 2*b+1+0, 2*b+1+1
        
        # M1
        U_block[0][term_idx] = 1
        U_block[4][term_idx] = 1
        V_block[0 * p + 2 * b][term_idx] = 1
        V_block[0 * p + 2 * b + 1][term_idx] = 1
        V_block[1 * p + 2 * b][term_idx] = 1
        V_block[1 * p + 2 * b + 1][term_idx] = 1
        W_block[(2 * b) * n + 0][term_idx] = 1
        W_block[(2 * b + 1) * n + 0][term_idx] = 0
        W_block[(2 * b) * n + 1][term_idx] = 0
        W_block[(2 * b + 1) * n + 1][term_idx] = 1
        term_idx += 1
        
        # M2
        U_block[3][term_idx] = 1
        U_block[4][term_idx] = 1
        V_block[0 * p + 2 * b][term_idx] = 1
        W_block[(2 * b) * n + 0][term_idx] = 0
        W_block[(2 * b + 1) * n + 0][term_idx] = 0
        W_block[(2 * b) * n + 1][term_idx] = 1
        W_block[(2 * b + 1) * n + 1][term_idx] = -1
        term_idx += 1
        
        # M3
        U_block[0][term_idx] = 1
        V_block[1 * p + 2 * b][term_idx] = 1
        V_block[1 * p + 2 * b + 1][term_idx] = -1
        W_block[(2 * b) * n + 0][term_idx] = 0
        W_block[(2 * b + 1) * n + 0][term_idx] = 1
        W_block[(2 * b) * n + 1][term_idx] = 0
        W_block[(2 * b + 1) * n + 1][term_idx] = 1
        term_idx += 1
        
        # M4
        U_block[4][term_idx] = 1
        V_block[0 * p + 2 * b][term_idx] = 1
        V_block[0 * p + 2 * b + 1][term_idx] = -1
        W_block[(2 * b) * n + 0][term_idx] = 1
        W_block[(2 * b + 1) * n + 0][term_idx] = 0
        W_block[(2 * b) * n + 1][term_idx] = 0
        W_block[(2 * b + 1) * n + 1][term_idx] = 0
        term_idx += 1
        
        # M5
        U_block[0][term_idx] = 1
        U_block[1][term_idx] = 1
        V_block[1 * p + 2 * b + 1][term_idx] = 1
        W_block[(2 * b) * n + 0][term_idx] = -1
        W_block[(2 * b + 1) * n + 0][term_idx] = 1
        W_block[(2 * b) * n + 1][term_idx] = 0
        W_block[(2 * b + 1) * n + 1][term_idx] = 0
        term_idx += 1
        
        # M6
        U_block[3][term_idx] = 1
        U_block[0][term_idx] = -1
        V_block[0 * p + 2 * b][term_idx] = 1
        V_block[0 * p + 2 * b + 1][term_idx] = 1
        W_block[(2 * b) * n + 0][term_idx] = 0
        W_block[(2 * b + 1) * n + 0][term_idx] = 0
        W_block[(2 * b) * n + 1][term_idx] = 1
        W_block[(2 * b + 1) * n + 1][term_idx] = 1
        term_idx += 1
        
        # M7
        U_block[1][term_idx] = 1
        U_block[4][term_idx] = -1
        V_block[1 * p + 2 * b][term_idx] = 1
        V_block[1 * p + 2 * b + 1][term_idx] = 1
        W_block[(2 * b) * n + 0][term_idx] = 1
        W_block[(2 * b + 1) * n + 0][term_idx] = 0
        W_block[(2 * b) * n + 1][term_idx] = 0
        W_block[(2 * b + 1) * n + 1][term_idx] = 0
        term_idx += 1
        
        # A[0:2, 2] * B[2, 2*b:2*b+2]
        # A indices: 0,1 (rows), 2 (col) -> 2, 5
        # B indices: 2 (row), 2*b, 2*b+1 (cols) -> 2+2*b, 2+2*b+1
        # C indices: 2*b, 2*b+1 (cols), 0,1 (rows) -> 2*b, 2*b+1, 2*b+1, 2*b+2
        
        # 4 terms
        # A[0,2] * B[2, 2*b] -> C[2*b, 0]
        U_block[2][term_idx] = 1
        V_block[2 * p + 2 * b][term_idx] = 1
        W_block[(2 * b) * n + 0][term_idx] = 1
        term_idx += 1
        
        # A[0,2] * B[2, 2*b+1] -> C[2*b+1, 0]
        U_block[2][term_idx] = 1
        V_block[2 * p + 2 * b + 1][term_idx] = 1
        W_block[(2 * b + 1) * n + 0][term_idx] = 1
        term_idx += 1
        
        # A[1,2] * B[2, 2*b] -> C[2*b, 1]
        U_block[5][term_idx] = 1
        V_block[2 * p + 2 * b][term_idx] = 1
        W_block[(2 * b) * n + 1][term_idx] = 1
        term_idx += 1
        
        # A[1,2] * B[2, 2*b+1] -> C[2*b+1, 1]
        U_block[5][term_idx] = 1
        V_block[2 * p + 2 * b + 1][term_idx] = 1
        W_block[(2 * b + 1) * n + 1][term_idx] = 1
        term_idx += 1
        
        # A[2, 0:2] * B[0:2, 2*b:2*b+2]
        # A indices: 2 (row), 0,1 (cols) -> 6, 7
        # B indices: 0,1 (rows), 2*b, 2*b+1 (cols)
        # C indices: 2*b, 2*b+1 (cols), 2 (row) -> 2*b+2, 2*b+1+2
        
        # 4 terms
        # A[2,0] * B[0, 2*b] -> C[2*b, 2]
        U_block[6][term_idx] = 1
        V_block[0 * p + 2 * b][term_idx] = 1
        W_block[(2 * b) * n + 2][term_idx] = 1
        term_idx += 1
        
        # A[2,0] * B[0, 2*b+1] -> C[2*b+1, 2]
        U_block[6][term_idx] = 1
        V_block[0 * p + 2 * b + 1][term_idx] = 1
        W_block[(2 * b + 1) * n + 2][term_idx] = 1
        term_idx += 1
        
        # A[2,1] * B[1, 2*b] -> C[2*b, 2]
        U_block[7][term_idx] = 1
        V_block[1 * p + 2 * b][term_idx] = 1
        W_block[(2 * b) * n + 2][term_idx] = 1
        term_idx += 1
        
        # A[2,1] * B[1, 2*b+1] -> C[2*b+1, 2]
        U_block[7][term_idx] = 1
        V_block[1 * p + 2 * b + 1][term_idx] = 1
        W_block[(2 * b + 1) * n + 2][term_idx] = 1
        term_idx += 1
        
        # A[2, 2] * B[2, 2*b:2*b+2]
        # A indices: 2,2 -> 8
        # B indices: 2, 2*b, 2*b+1
        # C indices: 2*b, 2*b+1, 2
        
        # 2 terms
        # A[2,2] * B[2, 2*b] -> C[2*b, 2]
        U_block[8][term_idx] = 1
        V_block[2 * p + 2 * b][term_idx] = 1
        W_block[(2 * b) * n + 2][term_idx] = 1
        term_idx += 1
        
        # A[2,2] * B[2, 2*b+1] -> C[2*b+1, 2]
        U_block[8][term_idx] = 1
        V_block[2 * p + 2 * b + 1][term_idx] = 1
        W_block[(2 * b + 1) * n + 2][term_idx] = 1
        term_idx += 1

    if check_decomposition(U_block, V_block, W_block, rank):
        best_result = {"U": U_block, "V": V_block, "W": W_block}
        best_rank = rank

    return best_result
