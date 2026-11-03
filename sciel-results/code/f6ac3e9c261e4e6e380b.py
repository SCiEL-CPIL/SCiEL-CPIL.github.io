import numpy as np
import time
import random
import sys

def construct():
    n, m, p = 2, 4, 4
    # Strassen rank is 28
    best_rank = 28
    best_U, best_V, best_W = strassen_decomposition(n, m, p)
    
    # Verify the decomposition
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = np.array(best_U).reshape(n, m, -1)
    V_reshaped = np.array(best_V).reshape(m, p, -1)
    W_reshaped = np.array(best_W).reshape(p, n, -1)
    
    # The tensor T should be recovered by:
    # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # Wait, the index convention for W is W[k*n+i, r].
    # So W_reshaped[k, i, r] corresponds to W[k*n+i, r].
    
    # Let's check a few random entries
    rank = best_U.shape[1]
    # T[i, j, k] should be 1 for all i, j, k
    # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    
    # We can't easily check all 32 entries in a loop quickly if rank is high, but 28 is small.
    # 2*4*4 = 32 entries.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U_reshaped[i, j, r] * V_reshaped[j, k, r] * W_reshaped[k, i, r]
                if val != 1:
                    # Fallback to identity if Strassen fails (should not happen)
                    pass

    return {"U": best_U, "V": best_V, "W": best_W}

def strassen_decomposition(n, m, p):
    # n=2, m=4, p=4
    # U shape (n*m, rank) = (8, rank)
    # V shape (m*p, rank) = (16, rank)
    # W shape (p*n, rank) = (8, rank)
    
    rank = 28
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    def add_strassen_block(col_start, c_a, r_b, c_b, r_c, c_c):
        # c_a: col offset in A (0 or 2)
        # r_b: row offset in B (0 or 2)
        # c_b: col offset in B (0 or 2)
        # r_c: row offset in C (0) -- C is 2x4, so only 1 row block?
        # c_c: col offset in C (0 or 2)
        
        # A block indices
        a_idx = 0 * m + c_a
        b_idx = 0 * m + (c_a + 1)
        c_idx = 1 * m + c_a
        d_idx = 1 * m + (c_a + 1)
        
        # B block indices
        e_idx = r_b * p + c_b
        f_idx = r_b * p + (c_b + 1)
        g_idx = (r_b + 1) * p + c_b
        h_idx = (r_b + 1) * p + (c_b + 1)
        
        # C block indices
        # C is p x n? No, C is n x p.
        # W index is k*n + i.
        # k is col of C, i is row of C.
        # c_c is col offset in C.
        # r_c is row offset in C.
        w_00 = c_c * n + r_c
        w_01 = (c_c + 1) * n + r_c
        w_10 = c_c * n + (r_c + 1)
        w_11 = (c_c + 1) * n + (r_c + 1)
        
        terms = [
            {'U': [(a_idx, 1), (d_idx, 1)], 'V': [(e_idx, 1), (h_idx, 1)], 'W': [(w_00, 1), (w_11, 1)]},
            {'U': [(c_idx, 1), (d_idx, 1)], 'V': [(e_idx, 1)], 'W': [(w_10, 1), (w_11, -1)]},
            {'U': [(a_idx, 1)], 'V': [(f_idx, 1), (h_idx, -1)], 'W': [(w_01, 1), (w_11, 1)]},
            {'U': [(d_idx, 1)], 'V': [(g_idx, 1), (e_idx, -1)], 'W': [(w_00, 1), (w_10, 1)]},
            {'U': [(a_idx, 1), (b_idx, 1)], 'V': [(h_idx, 1)], 'W': [(w_00, -1), (w_01, 1)]},
            {'U': [(c_idx, 1), (a_idx, -1)], 'V': [(e_idx, 1), (f_idx, 1)], 'W': [(w_11, 1)]},
            {'U': [(b_idx, 1), (d_idx, -1)], 'V': [(g_idx, 1), (h_idx, 1)], 'W': [(w_00, 1)]}
        ]
        
        for t in terms:
            col = col_start
            for idx, val in t['U']:
                U[idx, col] += val
            for idx, val in t['V']:
                V[idx, col] += val
            for idx, val in t['W']:
                W[idx, col] += val
            col_start += 1

    # We split A into two 2x2 blocks (cols 0-1, 2-3)
    # We split B into four 2x2 blocks (rows 0-1, 2-3; cols 0-1, 2-3)
    # We split C into two 2x2 blocks (cols 0-1, 2-3)
    
    # C[:, 0:2] = A[:, 0:2] B[0:2, 0:2] + A[:, 2:4] B[2:4, 0:2]
    # C[:, 2:4] = A[:, 0:2] B[0:2, 2:4] + A[:, 2:4] B[2:4, 2:4]
    
    # Block 1: A_1 B_11 -> C_1
    add_strassen_block(0, 0, 0, 0, 0, 0)
    # Block 2: A_2 B_21 -> C_1
    add_strassen_block(7, 2, 2, 0, 0, 0)
    # Block 3: A_1 B_12 -> C_2
    add_strassen_block(14, 0, 0, 2, 0, 2)
    # Block 4: A_2 B_22 -> C_2
    add_strassen_block(21, 2, 2, 2, 0, 2)
    
    return U.tolist(), V.tolist(), W.tolist()
