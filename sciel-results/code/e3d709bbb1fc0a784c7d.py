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
        Generates U, V, W for 4x4 * 4x4 multiplication using Strassen's algorithm.
        Each 2x2 block multiplication uses 7 scalar multiplications.
        4x4 = 4 blocks of 2x2, so 4 * 7 = 28 total rank.
        
        U: (16, 28), V: (16, 28), W: (16, 28)
        A indices: i*4 + j for A[i,j]
        B indices: j*4 + k for B[j,k]
        C indices: i*4 + k for C[i,k]
        """
        rank = 28
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(16)]
        W = [[0] * rank for _ in range(16)]
        
        # Block indices for 4x4 matrix split into 2x2 blocks of size 2x2:
        # Block (bi, bj) covers A[bi*2:bi*2+2, bj*2:bj*2+2]
        # A element (i,j) is in block (i//2, j//2) at local pos (i%2, j%2)
        # A global index = i*4+j, local in block = (i%2)*2 + (j%2)
        
        def block_a_indices(bi, bj):
            """Return global indices of A elements in block (bi,bj)"""
            return [(bi*2 + di)*4 + (bj*2 + dj) for di in range(2) for dj in range(2)]
        
        def block_b_indices(bi, bj):
            """Return global indices of B elements in block (bi,bj) - B[j,k] where j is row, k is col"""
            # B[j,k] with j in block bi rows, k in block bj cols
            return [(bi*2 + dj)*4 + (bj*2 + dk) for dj in range(2) for dk in range(2)]
        
        def block_c_indices(bi, bj):
            """Return global indices of C elements in block (bi,bj) - C[i,k] where i is row, k is col"""
            return [(bi*2 + di)*4 + (bj*2 + dk) for di in range(2) for dk in range(2)]
        
        col = 0
        # For each pair of 2x2 blocks (A_block, B_block) -> C_block
        # Strassen's 7 multiplications:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We have 4 block positions for A (rows) and 4 for B (cols), but A and B share the middle dimension.
        # A is 4x4: blocks A[bi, bj] for bi,bj in {0,1}
        # B is 4x4: blocks B[bj, bk] for bj,bk in {0,1}  
        # C is 4x4: blocks C[bi, bk] for bi,bk in {0,1}
        # For each (bi, bj, bk), we compute C[bi,bk] += A[bi,bj] * B[bj,bk]
        
        for bi in range(2):
            for bk in range(2):
                for bj in range(2):
                    # Compute C_block[bi,bk] += A_block[bi,bj] * B_block[bj,bk]
                    a_idx = block_a_indices(bi, bj)
                    b_idx = block_b_indices(bj, bk)
                    c_idx = block_c_indices(bi, bk)
                    
                    # Local indices within 2x2 block:
                    # pos 0: (0,0), pos 1: (0,1), pos 2: (1,0), pos 3: (1,1)
                    a11, a12, a21, a22 = a_idx[0], a_idx[1], a_idx[2], a_idx[3]
                    b11, b12, b21, b22 = b_idx[0], b_idx[1], b_idx[2], b_idx[3]
                    c11, c12, c21, c22 = c_idx[0], c_idx[1], c_idx[2], c_idx[3]
                    
                    # M1 = (A11+A22)(B11+B22)
                    U[a11][col] += 1; U[a22][col] += 1
                    V[b11][col] += 1; V[b22][col] += 1
                    W[c11][col] += 1; W[c22][col] += 1
                    col += 1
                    
                    # M2 = (A21+A22)B11
                    U[a21][col] += 1; U[a22][col] += 1
                    V[b11][col] += 1
                    W[c21][col] += 1
                    col += 1
                    
                    # M3 = A11(B12-B22)
                    U[a11][col] += 1
                    V[b12][col] += 1; V[b22][col] -= 1
                    W[c12][col] += 1
                    col += 1
                    
                    # M4 = A22(B21-B11)
                    U[a22][col] += 1
                    V[b21][col] += 1; V[b11][col] -= 1
                    W[c11][col] += 1; W[c21][col] += 1
                    col += 1
                    
                    # M5 = (A11+A12)B22
                    U[a11][col] += 1; U[a12][col] += 1
                    V[b22][col] += 1
                    W[c11][col] -= 1; W[c12][col] += 1
                    col += 1
                    
                    # M6 = (A21-A11)(B11+B12)
                    U[a21][col] += 1; U[a11][col] -= 1
                    V[b11][col] += 1; V[b12][col] += 1
                    W[c22][col] += 1
                    col += 1
                    
                    # M7 = (A12-A22)(B21+B22)
                    U[a12][col] += 1; U[a22][col] -= 1
                    V[b21][col] += 1; V[b22][col] += 1
                    W[c11][col] += 1
                    col += 1
        
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
        for c in range(28):
            for idx in range(16):
                U_full[idx][c] = U_str[idx][c]
                V_full[idx][c] = V_str[idx][c]
            # Map W indices: W_str[i*4+k] -> W_full[k*4+i]
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
