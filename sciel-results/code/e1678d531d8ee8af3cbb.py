"""Matrix multiplication 4x4x6 - Winograd/Strassen-based rank reduction."""
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
    # Strategy: 
    #   - Top level: treat A (4x4) as 2x2 blocks of 2x2, B (4x6) as 2x3 blocks of 2x3
    #   - Use 7 Strassen multiplications of (2x2) * (2x3)
    #   - Each (2x2)*(2x3) is decomposed as:
    #       * Inner Strassen (rank 7) for the first 2 columns: (2x2)*(2x2)
    #       * Schoolbook (rank 4) for the last 1 column: (2x2)*(2x1)
    #     Total per (2x2)*(2x3): 7 + 4 = 11
    #   - Total rank: 7 * 11 = 77
    
    R = 77
    U = np.zeros((nm, R), dtype=int)
    V = np.zeros((mp, R), dtype=int)
    W = np.zeros((pn, R), dtype=int)
    
    # Strassen coefficients for 2x2x2
    # A blocks indexed 0..3: 00, 01, 10, 11
    # B blocks indexed 0..3: 00, 01, 10, 11
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
        # For each top-level Strassen term s:
        # X_s is 2x2, Y_s is 2x3
        # Compute X_s * Y_s using inner Strassen (cols 0-1) + schoolbook (col 2)
        
        # Part 1: Inner Strassen for (2x2) * (2x2) [columns 0,1 of Y_s]
        for si in range(7):
            # U part: inner_alpha[si][a] * alpha[s][r]
            # a indexes position in X_s: row=a//2, col=a%2
            # r indexes block in A: ar=r//2, br=r%2
            for a in range(4):
                if alpha[si][a] != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]
            
            # V part: inner_beta[si][b] * beta[s][c]
            # b indexes position in Y_s[:,0:2]: row=b//2, col=b%2
            # c indexes block in B: ac=c//2, bc=c%2
            for b in range(4):
                if beta[si][b] != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += beta[si][b] * beta[s][c]
            
            # W part: recon[s] and inner_recon[si]
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    # za, zb are positions in the 2x2 inner output
                    # This is P_s[za, zb]
                    # It contributes to C block (a,b) at position (za, zb)
                    # C global: row = a*2 + za, col = b*3 + zb
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
        
        # Part 2: Schoolbook for (2x2) * (2x1) [column 2 of Y_s]
        # X_s (2x2) * Y_s[:,2] (2x1) -> Z (2x1)
        # 4 schoolbook multiplications
        for u in range(2):
            for v in range(2):
                # Term: X_s[u,v] * Y_s[v,2]
                
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
                        # B_c[v, 2]
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += beta[s][c]
                
                # W part
                for a, b, gamma in recon[s]:
                    # Output Z[u] contributes to C block (a,b) at row u, col 2
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
    
    # Verify the decomposition
    def verify_decomp(U, V, W):
        """Check if U, V, W exactly decompose the tensor."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if s != 1:
                        return False
        return True
    
    if not verify_decomp(U, V, W):
        # Fallback to schoolbook if verification fails
        return schoolbook()
    
    # Convert to lists of lists for the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
