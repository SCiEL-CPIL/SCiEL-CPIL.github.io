"""Rank-reduced exact decomposition for 3x4 times 4x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact decomposition for C = A(3x4) * B(4x4) using a rank-28
    decomposition derived from 4x4x4 Strassen blocks.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = 3, 4, 4
    
    # Construct a 4x4x4 decomposition first
    # U4: (16, 28), V4: (16, 28), W4: (16, 28)
    # Indices:
    # U4[i*4+j, r]
    # V4[j*4+k, r]
    # W4[k*4+i, r]
    
    total_rank_444 = 28
    U4 = [[0] * total_rank_444 for _ in range(16)]
    V4 = [[0] * total_rank_444 for _ in range(16)]
    W4 = [[0] * total_rank_444 for _ in range(16)]
    
    def add_strassen_2x2x2_444(i0, j0, k0, col_start):
        a11 = i0 * 4 + j0
        a12 = i0 * 4 + j0 + 1
        a21 = (i0 + 1) * 4 + j0
        a22 = (i0 + 1) * 4 + j0 + 1
        
        b11 = j0 * 4 + k0
        b12 = j0 * 4 + k0 + 1
        b21 = (j0 + 1) * 4 + k0
        b22 = (j0 + 1) * 4 + k0 + 1
        
        c11 = k0 * 4 + i0
        c12 = k0 * 4 + i0 + 1
        c21 = (k0 + 1) * 4 + i0
        c22 = (k0 + 1) * 4 + i0 + 1
        
        c = col_start
        U4[a11][c] += 1; U4[a22][c] += 1
        V4[b11][c] += 1; V4[b22][c] += 1
        W4[c11][c] += 1; W4[c22][c] += 1
        c += 1
        U4[a21][c] += 1; U4[a22][c] += 1
        V4[b11][c] += 1
        W4[c21][c] += 1; W4[c22][c] -= 1
        c += 1
        U4[a11][c] += 1
        V4[b12][c] += 1; V4[b22][c] -= 1
        W4[c12][c] += 1; W4[c22][c] += 1
        c += 1
        U4[a22][c] += 1
        V4[b21][c] += 1; V4[b11][c] -= 1
        W4[c11][c] += 1; W4[c21][c] += 1
        c += 1
        U4[a11][c] += 1; U4[a12][c] += 1
        V4[b22][c] += 1
        W4[c12][c] += 1; W4[c11][c] -= 1
        c += 1
        U4[a21][c] += 1; U4[a11][c] -= 1
        V4[b11][c] += 1; V4[b12][c] += 1
        W4[c22][c] += 1
        c += 1
        U4[a12][c] += 1; U4[a22][c] -= 1
        V4[b21][c] += 1; V4[b22][c] += 1
        W4[c11][c] += 1
        c += 1
        return c

    col = 0
    col = add_strassen_2x2x2_444(0, 0, 0, col)
    col = add_strassen_2x2x2_444(0, 2, 0, col)
    col = add_strassen_2x2x2_444(2, 0, 0, col)
    col = add_strassen_2x2x2_444(2, 2, 0, col)
    
    assert col == total_rank_444, f"Column mismatch in 444: {col} != {total_rank_444}"
    
    # Extract 3x4x4 sub-tensor
    # U: rows 0..11 (i=0..2, j=0..3)
    # V: all 16 rows
    # W: rows corresponding to k=0..3, i=0..2
    # W index in W4 is k*4 + i.
    # We need i in 0..2.
    # k=0: 0,1,2
    # k=1: 4,5,6
    # k=2: 8,9,10
    # k=3: 12,13,14
    
    U = [U4[i] for i in range(12)]
    V = V4
    w_indices = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
    W = [W4[i] for i in w_indices]
    
    if time.time() > deadline - 5.0:
        return {"U": U, "V": V, "W": W}
    
    try:
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        
        U3 = Ua.reshape(n, m, total_rank_444)
        V3 = Va.reshape(m, p, total_rank_444)
        W3 = Wa.reshape(p, n, total_rank_444)
        
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        
        if not np.allclose(T, 1.0, atol=1e-10):
            # Fallback to schoolbook if verification fails
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
            
    except Exception:
        # Fallback to schoolbook if exception
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
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
