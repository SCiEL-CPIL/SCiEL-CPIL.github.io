import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = 3, 4, 4
    
    def get_block_solution():
        """
        Returns rank-11 decomposition for 3x2x2 matrix multiplication.
        A is 3x2, B is 2x2, C is 3x2.
        U shape: (6, 11), V shape: (4, 11), W shape: (6, 11)
        W index convention: W[col_c * n + row_c, r]
        """
        rank_loc = 11
        U_loc = np.zeros((6, rank_loc))
        V_loc = np.zeros((4, rank_loc))
        W_loc = np.zeros((6, rank_loc))
        
        # Strassen-like decomposition for 2x2x2 part (rows 0,1 of A and C)
        # Term 0: A[0,0]*B[0,0] -> C[0,0]
        U_loc[0, 0] = 1; V_loc[0, 0] = 1; W_loc[0, 0] = 1
        
        # Term 1: (A[1,0]+A[0,0])*(B[0,1]+B[0,0]) -> C[1,0]
        U_loc[1, 1] = 1; U_loc[0, 1] = 1
        V_loc[1, 1] = 1; V_loc[0, 1] = 1
        W_loc[1, 1] = 1
        
        # Term 2: (A[1,0]-A[0,0])*(B[0,0]+B[0,1]) -> C[1,1] - C[0,1]
        U_loc[1, 2] = 1; U_loc[0, 2] = -1
        V_loc[0, 2] = 1; V_loc[1, 2] = 1
        W_loc[1, 2] = 1; W_loc[0, 2] = -1
        
        # Term 3: (A[0,1]+A[1,1])*(B[1,0]+B[1,1]) -> C[0,1]
        U_loc[0, 3] = 1; U_loc[1, 3] = 1
        V_loc[2, 3] = 1; V_loc[3, 3] = 1
        W_loc[0, 3] = 1
        
        # Term 4: (A[0,1]-A[1,1])*(B[1,1]+B[1,0]) -> C[0,0] - C[1,0]
        U_loc[0, 4] = 1; U_loc[1, 4] = -1
        V_loc[3, 4] = 1; V_loc[2, 4] = 1
        W_loc[0, 4] = 1; W_loc[1, 4] = -1
        
        # Term 5: (A[1,1]+A[0,1])*(B[1,1]-B[1,0]) -> C[1,1]
        U_loc[1, 5] = 1; U_loc[0, 5] = 1
        V_loc[3, 5] = 1; V_loc[2, 5] = -1
        W_loc[1, 5] = 1
        
        # Term 6: (A[0,0]+A[1,1])*(B[1,1]-B[0,0]) -> C[0,1] - C[1,0]
        U_loc[0, 6] = 1; U_loc[1, 6] = 1
        V_loc[3, 6] = 1; V_loc[0, 6] = -1
        W_loc[0, 6] = 1; W_loc[1, 6] = -1
        
        # Terms 7-10: Handle row 2 of A (3rd row)
        # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0]
        # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1]
        
        # Term 7: A[2,0]*B[0,0] -> C[2,0]
        U_loc[2, 7] = 1; V_loc[0, 7] = 1; W_loc[4, 7] = 1
        
        # Term 8: A[2,0]*B[0,1] -> C[2,1]
        U_loc[2, 8] = 1; V_loc[1, 8] = 1; W_loc[5, 8] = 1
        
        # Term 9: A[2,1]*B[1,0] -> C[2,0]
        U_loc[2, 9] = 1; V_loc[2, 9] = 1; W_loc[4, 9] = 1
        
        # Term 10: A[2,1]*B[1,1] -> C[2,1]
        U_loc[2, 10] = 1; V_loc[3, 10] = 1; W_loc[5, 10] = 1
        
        return U_loc, V_loc, W_loc
    
    def build_full_solution():
        """
        Build the full 3x4x4 solution by tiling 4 blocks of 3x2x2.
        Block (a_row_start, a_col_start, b_row_start, b_col_start, c_col_start):
        - A rows: a_row_start to a_row_start+2 (3 rows)
        - A cols: a_col_start to a_col_start+1 (2 cols)
        - B rows: b_row_start to b_row_start+1 (2 rows)  
        - B cols: b_col_start to b_col_start+1 (2 cols)
        - C cols: c_col_start to c_col_start+1 (2 cols)
        - C rows: 0 to 2 (all 3 rows)
        """
        rank = 44
        U = np.zeros((n * m, rank))  # (12, 44)
        V = np.zeros((m * p, rank))  # (16, 44)
        W = np.zeros((p * n, rank))  # (12, 44)
        
        U_loc, V_loc, W_loc = get_block_solution()
        
        # Four blocks covering the 4 columns of A/B and 4 columns of C:
        # Block 0: A cols 0-1, B rows 0-1, C cols 0-1
        # Block 1: A cols 2-3, B rows 2-3, C cols 0-1
        # Block 2: A cols 0-1, B rows 0-1, C cols 2-3
        # Block 3: A cols 2-3, B rows 2-3, C cols 2-3
        
        blocks = [
            (0, 0, 0, 0, 0),  # (a_col_start, b_row_start, b_col_start, c_col_start)
            (2, 2, 0, 0),
            (0, 0, 2, 2),
            (2, 2, 2, 2),
        ]
        
        for block_idx, (a_col_start, b_row_start, b_col_start, c_col_start) in enumerate(blocks):
            col_offset = block_idx * 11
            
            # A indices: row_a * 4 + col_a, where col_a in [a_col_start, a_col_start+1]
            # B indices: row_b * 4 + col_b, where row_b in [b_row_start, b_row_start+1], col_b in [b_col_start, b_col_start+1]
            # C indices: col_c * 3 + row_c, where col_c in [c_col_start, c_col_start+1], row_c in [0,1,2]
            
            for row_a in range(3):
                for col_a_local in range(2):
                    col_a = a_col_start + col_a_local
                    local_a_idx = row_a * 2 + col_a_local
                    global_a_idx = row_a * m + col_a
                    U[global_a_idx, col_offset:col_offset+11] += U_loc[local_a_idx, :]
            
            for row_b_local in range(2):
                for col_b_local in range(2):
                    row_b = b_row_start + row_b_local
                    col_b = b_col_start + col_b_local
                    local_b_idx = row_b_local * 2 + col_b_local
                    global_b_idx = row_b * p + col_b
                    V[global_b_idx, col_offset:col_offset+11] += V_loc[local_b_idx, :]
            
            for row_c in range(3):
                for col_c_local in range(2):
                    col_c = c_col_start + col_c_local
                    local_c_idx = row_c * 2 + col_c_local
                    global_c_idx = col_c * n + row_c
                    W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify_solution(sol, max_samples=200):
        """Verify the decomposition is correct."""
        U = np.array(sol['U'], dtype=float)
        V = np.array(sol['V'], dtype=float)
        W = np.array(sol['W'], dtype=float)
        rank = U.shape[1]
        
        # Build target tensor T[i, j, k]
        # i = row_a * 4 + col_a (0..11)
        # j = row_b * 4 + col_b (0..15) where row_b = col_a, col_b = col_c
        # k = col_c * 3 + row_a (0..11)
        T = np.zeros((12, 16, 12))
        for row_a in range(3):
            for col_a in range(4):
                for col_c in range(4):
                    i = row_a * 4 + col_a
                    j = col_a * 4 + col_c
                    k = col_c * 3 + row_a
                    T[i, j, k] = 1.0
        
        # Check random samples
        np.random.seed(42)
        for _ in range(max_samples):
            i = np.random.randint(0, 12)
            j = np.random.randint(0, 16)
            k = np.random.randint(0, 12)
            computed = np.sum(U[i, :] * V[j, :] * W[k, :])
            expected = T[i, j, k]
            if abs(computed - expected) > 1e-9:
                return False
        return True
    
    sol = build_full_solution()
    
    # Verify correctness
    if not verify_solution(sol, max_samples=500):
        # Fallback to schoolbook if verification fails
        rank = 48
        U = np.zeros((12, rank))
        V = np.zeros((16, rank))
        W = np.zeros((12, rank))
        for row_a in range(3):
            for col_a in range(4):
                for col_c in range(4):
                    idx = (row_a * 4 + col_a) * 4 + col_c
                    i = row_a * 4 + col_a
                    j = col_a * 4 + col_c
                    k = col_c * 3 + row_a
                    U[i, idx] = 1
                    V[j, idx] = 1
                    W[k, idx] = 1
        sol = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if time.time() > deadline:
        return sol
    
    return sol
