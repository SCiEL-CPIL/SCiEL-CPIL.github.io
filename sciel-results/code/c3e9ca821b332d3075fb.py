import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s limit
    
    # Schoolbook baseline
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
        
    def generate_strassen_4x4x4():
        """
        Generates U, V, W for 4x4 * 4x4 multiplication with rank 28.
        U: (16, 28), V: (16, 28), W: (16, 28)
        Indices:
        A: i in 0..3, j in 0..3. Index = i*4 + j.
        B: j in 0..3, k in 0..3. Index = j*4 + k.
        C: i in 0..3, k in 0..3. Index = i*4 + k.
        """
        rank = 28
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(16)]
        W = [[0] * rank for _ in range(16)]
        
        def add_term(u_indices, v_indices, w_index, sign, col):
            for idx in u_indices:
                U[idx][col] += sign
            for idx in v_indices:
                V[idx][col] += sign
            W[w_index][col] += sign

        col = 0
        # M1 = (A11+A22)(B11+B22)
        # A11+A22:
        # (0,0): [0, 10]
        # (0,1): [1, 11]
        # (1,0): [4, 14]
        # (1,1): [5, 15]
        # B11+B22:
        # (0,0): [0, 10]
        # (0,1): [1, 11]
        # (1,0): [4, 14]
        # (1,1): [5, 15]
        # C11:
        # (0,0): 0, (0,1): 1, (1,0): 4, (1,1): 5
        
        add_term([0, 10], [0, 10], 0, 1, col); col += 1
        add_term([1, 11], [4, 14], 0, 1, col); col += 1
        add_term([0, 10], [1, 11], 1, 1, col); col += 1
        add_term([1, 11], [5, 15], 1, 1, col); col += 1
        add_term([4, 14], [0, 10], 4, 1, col); col += 1
        add_term([5, 15], [4, 14], 4, 1, col); col += 1
        add_term([4, 14], [1, 11], 5, 1, col); col += 1
        add_term([5, 15], [5, 15], 5, 1, col); col += 1
        
        # M2 = (A21+A22)B11
        # A21+A22:
        # (0,0): [8, 10]
        # (0,1): [9, 11]
        # (1,0): [12, 14]
        # (1,1): [13, 15]
        # B11:
        # (0,0): [0], (0,1): [1], (1,0): [4], (1,1): [5]
        
        add_term([8, 10], [0], 0, 1, col); col += 1
        add_term([9, 11], [4], 0, 1, col); col += 1
        add_term([8, 10], [1], 1, 1, col); col += 1
        add_term([9, 11], [5], 1, 1, col); col += 1
        add_term([12, 14], [0], 4, 1, col); col += 1
        add_term([13, 15], [4], 4, 1, col); col += 1
        add_term([12, 14], [1], 5, 1, col); col += 1
        add_term([13, 15], [5], 5, 1, col); col += 1
        
        # M3 = A11(B12-B22)
        # A11:
        # (0,0): [0], (0,1): [1], (1,0): [4], (1,1): [5]
        # B12-B22:
        # (0,0): [2, -10], (0,1): [3, -11], (1,0): [6, -14], (1,1): [7, -15]
        
        add_term([0], [2, -10], 0, 1, col); col += 1
        add_term([1], [6, -14], 0, 1, col); col += 1
        add_term([0], [3, -11], 1, 1, col); col += 1
        add_term([1], [7, -15], 1, 1, col); col += 1
        add_term([4], [2, -10], 4, 1, col); col += 1
        add_term([5], [6, -14], 4, 1, col); col += 1
        add_term([4], [3, -11], 5, 1, col); col += 1
        add_term([5], [7, -15], 5, 1, col); col += 1
        
        # M4 = A22(B21-B11)
        # A22:
        # (0,0): [10], (0,1): [11], (1,0): [14], (1,1): [15]
        # B21-B11:
        # (0,0): [8, -0], (0,1): [9, -1], (1,0): [12, -4], (1,1): [13, -5]
        
        add_term([10], [8, -0], 0, 1, col); col += 1
        add_term([11], [12, -4], 0, 1, col); col += 1
        add_term([10], [9, -1], 1, 1, col); col += 1
        add_term([11], [13, -5], 1, 1, col); col += 1
        add_term([14], [8, -0], 4, 1, col); col += 1
        add_term([15], [12, -4], 4, 1, col); col += 1
        add_term([14], [9, -1], 5, 1, col); col += 1
        add_term([15], [13, -5], 5, 1, col); col += 1
        
        # M5 = (A11+A12)B22
        # A11+A12:
        # (0,0): [0, 2], (0,1): [1, 3], (1,0): [4, 6], (1,1): [5, 7]
        # B22:
        # (0,0): [10], (0,1): [11], (1,0): [14], (1,1): [15]
        
        add_term([0, 2], [10], 0, 1, col); col += 1
        add_term([1, 3], [14], 0, 1, col); col += 1
        add_term([0, 2], [11], 1, 1, col); col += 1
        add_term([1, 3], [15], 1, 1, col); col += 1
        add_term([4, 6], [10], 4, 1, col); col += 1
        add_term([5, 7], [14], 4, 1, col); col += 1
        add_term([4, 6], [11], 5, 1, col); col += 1
        add_term([5, 7], [15], 5, 1, col); col += 1
        
        # M6 = (A21-A11)(B11+B12)
        # A21-A11:
        # (0,0): [8, -0], (0,1): [9, -1], (1,0): [12, -4], (1,1): [13, -5]
        # B11+B12:
        # (0,0): [0, 2], (0,1): [1, 3], (1,0): [4, 6], (1,1): [5, 7]
        
        add_term([8, -0], [0, 2], 0, 1, col); col += 1
        add_term([9, -1], [4, 6], 0, 1, col); col += 1
        add_term([8, -0], [1, 3], 1, 1, col); col += 1
        add_term([9, -1], [5, 7], 1, 1, col); col += 1
        add_term([12, -4], [0, 2], 4, 1, col); col += 1
        add_term([13, -5], [4, 6], 4, 1, col); col += 1
        add_term([12, -4], [1, 3], 5, 1, col); col += 1
        add_term([13, -5], [5, 7], 5, 1, col); col += 1
        
        # M7 = (A12-A22)(B21+B22)
        # A12-A22:
        # (0,0): [2, -10], (0,1): [3, -11], (1,0): [6, -14], (1,1): [7, -15]
        # B21+B22:
        # (0,0): [8, 10], (0,1): [9, 11], (1,0): [12, 14], (1,1): [13, 15]
        
        add_term([2, -10], [8, 10], 0, 1, col); col += 1
        add_term([3, -11], [12, 14], 0, 1, col); col += 1
        add_term([2, -10], [9, 11], 1, 1, col); col += 1
        add_term([3, -11], [13, 15], 1, 1, col); col += 1
        add_term([6, -14], [8, 10], 4, 1, col); col += 1
        add_term([7, -15], [12, 14], 4, 1, col); col += 1
        add_term([6, -14], [9, 11], 5, 1, col); col += 1
        add_term([7, -15], [13, 15], 5, 1, col); col += 1
        
        return U, V, W

    try:
        U_str, V_str, W_str = generate_strassen_4x4x4()
        
        # Construct full 4x4*4x7
        # C[:, 0:4] uses Strassen (rank 28)
        # C[:, 4:7] uses Schoolbook (rank 48)
        total_rank = 28 + 48
        U_full = [[0] * total_rank for _ in range(16)]
        V_full = [[0] * total_rank for _ in range(28)] # m*p = 4*7 = 28
        W_full = [[0] * total_rank for _ in range(28)] # p*n = 7*4 = 28
        
        # Copy Strassen part
        # U, V, W from generate_strassen are 16x28, 16x28, 16x28
        # U_full[i*4+j, c] = U_str[i*4+j, c]
        # V_full[j*4+k, c] = V_str[j*4+k, c] for k in 0..3
        # W_full[k*4+i, c] = W_str[i*4+k, c] for k in 0..3
        
        for c in range(28):
            for idx in range(16):
                U_full[idx][c] = U_str[idx][c]
                V_full[idx][c] = V_str[idx][c]
            # Map W indices
            # W_str index is i*4+k (C[i,k])
            # W_full index is k*4+i (C[k,i] in trace convention? No, W[k*n+i] is C[i,k]?)
            # Task says: W[k*n+i, r]. So W index = k*4 + i.
            # Our Strassen W is indexed by C[i,k] which is i*4+k.
            # So we need to transpose the indices for W.
            for i in range(4):
                for k in range(4):
                    W_full[k*4 + i][c] = W_str[i*4 + k][c]
                    
        # Add Schoolbook part for C[:, 4:7]
        col_offset = 28
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    # A[i,j] * B[j, 4+k] -> C[i, 4+k]
                    U_full[i*4+j][col_offset] = 1
                    V_full[j*4+(4+k)][col_offset] = 1
                    W_full[(4+k)*4+i][col_offset] = 1
                    col_offset += 1
                    
        # Verify
        valid = True
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    total = 0
                    for c in range(total_rank):
                        total += U_full[i*4+j][c] * V_full[j*7+k][c] * W_full[k*4+i][c]
                    if total != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
                
        if valid:
            return {"U": U_full, "V": V_full, "W": W_full}
        else:
            return {"U": U_sb, "V": V_sb, "W": W_sb}
            
    except Exception as e:
        return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
