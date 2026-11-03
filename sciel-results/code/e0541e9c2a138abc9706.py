import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = 3, 4, 4
    
    # Local decomposition for (3x2) x (2x2) -> (3x2) with rank 11
    # U_loc: (6, 11), V_loc: (4, 11), W_loc: (6, 11)
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))
    
    # Strassen part for top 2x2 x 2x2 -> 2x2 (cols 0-6)
    # Local indices: A rows 0-1 (idx 0,1,2,3), B rows 0-1 (idx 0,1,2,3), C rows 0-1 (idx 0,1,2,3)
    # Strassen:
    # M0 = (A00+A11)*(B00+B11)
    # M1 = (A10+A11)*B00
    # M2 = A00*(B01-B11)
    # M3 = A11*(B10-B00)
    # M4 = (A00+A10)*B11
    # M5 = (A01-A10)*(B00+B10)
    # M6 = (A01-A11)*(B01+B11)
    # C00 = M0+M3-M4-M6
    # C01 = M2+M4
    # C10 = M1+M5
    # C11 = M0-M1+M2+M5
    
    U_loc[0, 0] = 1; U_loc[3, 0] = 1   # A00+A11
    V_loc[0, 0] = 1; V_loc[3, 0] = 1   # B00+B11
    W_loc[0, 0] = 1; W_loc[3, 0] = 1   # C00+=1, C11+=1
    
    U_loc[2, 1] = 1; U_loc[3, 1] = 1   # A10+A11
    V_loc[0, 1] = 1                     # B00
    W_loc[2, 1] = 1; W_loc[3, 1] = -1  # C10+=1, C11-=-1
    
    U_loc[0, 2] = 1                     # A00
    V_loc[1, 2] = 1; V_loc[3, 2] = -1  # B01-B11
    W_loc[1, 2] = 1; W_loc[3, 2] = 1   # C01+=1, C11+=1
    
    U_loc[3, 3] = 1                     # A11
    V_loc[2, 3] = 1; V_loc[0, 3] = -1  # B10-B00
    W_loc[0, 3] = 1; W_loc[2, 3] = 1   # C00+=1, C10+=1
    
    U_loc[0, 4] = 1; U_loc[1, 4] = 1   # A00+A10
    V_loc[3, 4] = 1                     # B11
    W_loc[0, 4] = -1; W_loc[1, 4] = 1  # C00-=1, C01+=1
    
    U_loc[2, 5] = 1; U_loc[0, 5] = -1  # A01-A10
    V_loc[0, 5] = 1; V_loc[1, 5] = 1   # B00+B10
    W_loc[3, 5] = 1                    # C11+=1
    
    U_loc[1, 6] = 1; U_loc[3, 6] = -1  # A01-A11
    V_loc[2, 6] = 1; V_loc[3, 6] = 1   # B01+B11
    W_loc[0, 6] = 1                    # C00+=1
    
    # Trivial part for bottom row: A row 2 (idx 4,5) x B (2x2) -> C row 2 (idx 4,5)
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0]
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1]
    U_loc[4, 7] = 1; V_loc[0, 7] = 1; W_loc[4, 7] = 1  # A[2,0]*B[0,0] -> C[2,0]
    U_loc[5, 8] = 1; V_loc[2, 8] = 1; W_loc[4, 8] = 1  # A[2,1]*B[1,0] -> C[2,0]
    U_loc[4, 9] = 1; V_loc[1, 9] = 1; W_loc[5, 9] = 1  # A[2,0]*B[0,1] -> C[2,1]
    U_loc[5, 10] = 1; V_loc[3, 10] = 1; W_loc[5, 10] = 1  # A[2,1]*B[1,1] -> C[2,1]
    
    rank = 44
    U = np.zeros((n * m, rank))  # (12, 44)
    V = np.zeros((m * p, rank))  # (16, 44)
    W = np.zeros((p * n, rank))  # (12, 44)
    
    def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        col_offset = block_idx * 11
        # A: 3 rows, 2 cols -> local idx r*2+c, global idx (a_row_start+r)*m + (a_col_start+c)
        for r in range(3):
            for c in range(2):
                local_a_idx = r * 2 + c
                global_a_idx = (a_row_start + r) * m + (a_col_start + c)
                U[global_a_idx, col_offset:col_offset+11] += U_loc[local_a_idx, :]
        # B: 2 rows, 2 cols -> local idx r*2+c, global idx (b_row_start+r)*p + (b_col_start+c)
        for r in range(2):
            for c in range(2):
                local_b_idx = r * 2 + c
                global_b_idx = (b_row_start + r) * p + (b_col_start + c)
                V[global_b_idx, col_offset:col_offset+11] += V_loc[local_b_idx, :]
        # C: 3 rows, 2 cols -> local idx r*2+c, global idx (c_col_start+c)*n + (c_row_start+r) [trace convention]
        for r in range(3):
            for c in range(2):
                local_c_idx = r * 2 + c
                global_c_idx = (c_col_start + c) * n + (c_row_start + r)
                W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]
    
    # Block 0: A rows 0-2 cols 0-1, B rows 0-1 cols 0-1, C rows 0-2 cols 0-1
    add_block(0, 0, 0, 0, 0, 0, 0)
    # Block 1: A rows 0-2 cols 2-3, B rows 2-3 cols 0-1, C rows 0-2 cols 0-1
    add_block(1, 0, 2, 2, 0, 0, 0)
    # Block 2: A rows 0-2 cols 0-1, B rows 0-1 cols 2-3, C rows 0-2 cols 2-3
    add_block(2, 0, 0, 0, 2, 0, 2)
    # Block 3: A rows 0-2 cols 2-3, B rows 2-3 cols 2-3, C rows 0-2 cols 2-3
    add_block(3, 0, 2, 2, 2, 0, 2)
    
    sol = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify correctness
    def verify_solution(sol, max_tests=200):
        U_arr = np.array(sol['U'])
        V_arr = np.array(sol['V'])
        W_arr = np.array(sol['W'])
        
        T = np.zeros((12, 16, 12))
        for row_a in range(3):
            for col_a in range(4):
                for col_c in range(4):
                    i = row_a * 4 + col_a
                    j = col_a * 4 + col_c
                    k = col_c * 3 + row_a
                    T[i, j, k] = 1.0
        
        np.random.seed(12345)
        for _ in range(max_tests):
            i = np.random.randint(0, 12)
            j = np.random.randint(0, 16)
            k = np.random.randint(0, 12)
            computed = np.sum(U_arr[i, :] * V_arr[j, :] * W_arr[k, :])
            expected = T[i, j, k]
            if abs(computed - expected) > 1e-9:
                return False
        return True
    
    if not verify_solution(sol, max_tests=200):
        # Fallback to trivial rank-48 solution
        rank_triv = 48
        U_triv = np.zeros((12, rank_triv))
        V_triv = np.zeros((16, rank_triv))
        W_triv = np.zeros((12, rank_triv))
        idx = 0
        for row_a in range(3):
            for col_a in range(4):
                for col_c in range(4):
                    i = row_a * 4 + col_a
                    j = col_a * 4 + col_c
                    k = col_c * 3 + row_a
                    U_triv[i, idx] = 1
                    V_triv[j, idx] = 1
                    W_triv[k, idx] = 1
                    idx += 1
        sol = {"U": U_triv.tolist(), "V": V_triv.tolist(), "W": W_triv.tolist()}
    
    if time.time() > deadline:
        return sol
    
    return sol
