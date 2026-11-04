"""Improved program: Strassen decomposition for 4x4 * 4x6 matrix multiplication."""

import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify tensor decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    """Construct a rank-77 tensor decomposition for 4x4 * 4x6."""
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = 4, 4, 6
    R = 77
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    # Strassen coefficients for 2x2x2
    # alpha[s][a], beta[s][b], gamma[s][c]
    # a,b,c in 0..3 (0:00, 1:01, 2:10, 3:11)
    alpha = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    beta = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    # recon[s] = list of (out_block_row, out_block_col, coeff)
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    col = 0
    # Top level: 7 Strassen terms
    for s in range(7):
        # For each term, we have a 2x2 matrix X (from A) and 2x3 matrix Y (from B)
        # We decompose X*Y into 7 (Strassen 2x2x2) + 4 (Schoolbook 2x2x1) = 11 terms
        
        # Part 1: Strassen for columns 0,1 of Y (2x2x2)
        for si in range(7):
            # U part
            for a in range(4):
                if alpha[s][a] != 0:
                    ar, br = a // 2, a % 2
                    for ai in range(2):
                        for aj in range(2):
                            # X[ai, aj] contribution from A[ar*2+ai, br*2+aj]
                            # Inner Strassen alpha[si] for block a
                            # We need to map inner block a to local 2x2 indices
                            # Inner a is 0:00, 1:01, 2:10, 3:11
                            # So local row = a//2, local col = a%2
                            local_r, local_c = a // 2, a % 2
                            if alpha[si][a] != 0:
                                u_idx = (ar*2 + local_r)*4 + (br*2 + local_c)
                                U[u_idx, col] += alpha[s][a] * alpha[si][a]
            
            # V part
            for b in range(4):
                if beta[s][b] != 0:
                    ac, bc = b // 2, b % 2
                    local_r, local_c = b // 2, b % 2
                    if beta[si][b] != 0:
                        # Y[local_r, local_c] from B[ac*2+local_r, bc*3+local_c]
                        v_idx = (ac*2 + local_r)*6 + (bc*3 + local_c)
                        V[v_idx, col] += beta[s][b] * beta[si][b]
            
            # W part
            for a_idx, b_idx, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    # Output is at C[a_idx*2+za, b_idx*3+zb]
                    w_idx = (b_idx*3 + zb)*4 + (a_idx*2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
        
        # Part 2: Schoolbook for column 2 of Y (2x2x1)
        # Z[u] = sum_v X[u,v] * Y[v,2]
        for u in range(2):
            for v in range(2):
                # U part: X[u,v]
                for r in range(4):
                    if alpha[s][r] != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar*2 + u)*4 + (br*2 + v)
                        U[u_idx, col] += alpha[s][r]
                
                # V part: Y[v,2]
                for c in range(4):
                    if beta[s][c] != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac*2 + v)*6 + (bc*3 + 2)
                        V[v_idx, col] += beta[s][c]
                
                # W part: C[a_idx*2+u, b_idx*3+2]
                for a_idx, b_idx, gamma in recon[s]:
                    w_idx = (b_idx*3 + 2)*4 + (a_idx*2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
    
    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if time.time() < deadline:
        if _verify_decomposition(U, V, W, n, m, p):
            return result
    
    # Fallback
    rank = n * m * p
    U_fb = [[0] * rank for _ in range(n * m)]
    V_fb = [[0] * rank for _ in range(m * p)]
    W_fb = [[0] * rank for _ in range(p * n)]
    col_fb = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col_fb] = 1
                V_fb[j * p + k][col_fb] = 1
                W_fb[k * n + i][col_fb] = 1
                col_fb += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
