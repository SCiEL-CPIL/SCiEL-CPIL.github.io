import time
import sys
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 24
    pn = p * n  # 24
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p  # 96
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Strassen-based decomposition for 4x4x6
    # Top level: 7 Strassen multiplications of (2x2) * (2x3)
    # Each (2x2)*(2x3) is decomposed as:
    #   * Inner Strassen (rank 7) for the first 2 columns: (2x2)*(2x2)
    #   * Schoolbook (rank 4) for the last 1 column: (2x2)*(2x1)
    # Total per (2x2)*(2x3): 7 + 4 = 11
    # Total rank: 7 * 11 = 77
    
    R = 77
    U = np.zeros((nm, R), dtype=int)
    V = np.zeros((mp, R), dtype=int)
    W = np.zeros((pn, R), dtype=int)
    
    # Strassen coefficients for 2x2x2
    alpha = [
        [1, 0, 0, 1],  # s0: A00 + A11
        [0, 0, 1, 1],  # s1: A10 + A11
        [1, 0, 0, 0],  # s2: A00
        [0, 0, 0, 1],  # s3: A11
        [1, 1, 0, 0],  # s4: A00 + A01
        [-1, 0, 1, 0], # s5: A10 - A00
        [0, 1, 0, -1]  # s6: A01 - A11
    ]
    beta = [
        [1, 0, 0, 1],  # s0: B00 + B11
        [1, 0, 0, 0],  # s1: B00
        [0, 1, 0, -1], # s2: B01 - B11
        [-1, 0, 1, 0], # s3: B10 - B00
        [0, 0, 0, 1],  # s4: B11
        [1, 1, 0, 0],  # s5: B00 + B01
        [0, 0, 1, 1]   # s6: B10 + B11
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],     # s0
        [(1, 0, 1), (1, 1, -1)],    # s1
        [(0, 1, 1), (1, 1, 1)],     # s2
        [(0, 0, 1), (1, 0, 1)],     # s3
        [(0, 0, -1), (0, 1, 1)],    # s4
        [(1, 1, 1)],                # s5
        [(0, 0, 1)]                 # s6
    ]
    
    col = 0
    for s in range(7):
        # Part 1: Inner Strassen for (2x2) * (2x2) [columns 0,1 of Y_s]
        for si in range(7):
            if time.time() > deadline:
                break
            
            # U part
            for a in range(4):
                if alpha[si][a] != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]
            
            # V part
            for b in range(4):
                if beta[si][b] != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += beta[si][b] * beta[s][c]
            
            # W part
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
        
        # Part 2: Schoolbook for (2x2) * (2x1) [column 2 of Y_s]
        for u in range(2):
            if time.time() > deadline:
                break
            for v in range(2):
                # U part
                for r in range(4):
                    if alpha[s][r] != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += alpha[s][r]
                
                # V part
                for c in range(4):
                    if beta[s][c] != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += beta[s][c]
                
                # W part
                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
    
    # Fast Verification
    def verify_decomp(U_mat, V_mat, W_mat):
        rank = U_mat.shape[1]
        # We need to check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U_mat[i * m + j, r] * V_mat[j * p + k, r] * W_mat[k * n + i, r]
                    if s != 1:
                        return False
        return True
    
    if not verify_decomp(U, V, W):
        return schoolbook()
    
    # Attempt rank reduction by checking for linearly dependent terms
    # We check if removing any single term breaks the decomposition.
    # This is a necessary condition for a term to be redundant.
    # If removing term r makes the decomposition invalid, term r is essential.
    # If removing term r keeps the decomposition valid, term r is redundant.
    
    # To speed this up, we can precompute the contribution of each term to the tensor.
    # But for R=77, checking 77 terms * (4*4*6*76) ops is ~ 1.5M ops, which is fast.
    
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = current_U.shape[1]
    
    # We iterate until no more terms can be removed
    # To avoid infinite loops or excessive time, we limit iterations
    max_removal_attempts = 10
    for _ in range(max_removal_attempts):
        if time.time() > deadline:
            break
        
        removed = False
        for r in range(current_rank):
            # Try removing term r
            test_U = np.delete(current_U, r, axis=1)
            test_V = np.delete(current_V, r, axis=1)
            test_W = np.delete(current_W, r, axis=1)
            
            if verify_decomp(test_U, test_V, test_W):
                current_U = test_U
                current_V = test_V
                current_W = test_W
                current_rank -= 1
                removed = True
                break # Restart the loop with the new rank
        
        if not removed:
            break
    
    # Final check
    if not verify_decomp(current_U, current_V, current_W):
        return schoolbook()
    
    U_list = current_U.tolist()
    V_list = current_V.tolist()
    W_list = current_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
