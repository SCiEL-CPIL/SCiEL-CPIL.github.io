import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 28 decomposition for 2x4x4 matrix multiplication.
    # This is achieved by decomposing into 4 independent 2x2x2 multiplications,
    # each using the rank-7 Strassen decomposition.
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen decomposition for 2x2x2:
    # A = [[a1, a2], [a3, a4]], B = [[b1, b2], [b3, b4]], C = A*B
    # Terms (U_idx, V_idx, W_idx) with coefficients
    # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    strassen_2x2 = [
        # (u_indices, v_indices, w_indices)
        # Each is a list of (local_index, coefficient)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    # Mapping functions for 2x2 blocks
    # A is 2x4. Blocks are 2x2.
    # A_L is columns 0,1. A_R is columns 2,3.
    # U index mapping for a 2x2 block at (row_offset, col_offset)
    def u_idx(row_off, col_off, local_r, local_c):
        return (row_off + local_r) * m + (col_off + local_c)
        
    # B is 4x4. Blocks are 2x2.
    # B_TL is rows 0,1 cols 0,1. B_TR is rows 0,1 cols 2,3.
    # B_BL is rows 2,3 cols 0,1. B_BR is rows 2,3 cols 2,3.
    def v_idx(row_off, col_off, local_r, local_c):
        return (row_off + local_r) * p + (col_off + local_c)
        
    # C is 2x4. Blocks are 2x2.
    # C_L is cols 0,1. C_R is cols 2,3.
    def w_idx(col_off, local_r, local_c):
        return (col_off + local_c) * n + local_r

    col = 0
    # 4 block products:
    # 1. A_L * B_TL -> C_L
    # 2. A_R * B_BL -> C_L
    # 3. A_L * B_TR -> C_R
    # 4. A_R * B_BR -> C_R
    
    block_configs = [
        # (A_col_off, B_row_off, B_col_off, C_col_off)
        (0, 0, 0, 0), # A_L * B_TL -> C_L
        (2, 2, 0, 0), # A_R * B_BL -> C_L
        (0, 0, 2, 2), # A_L * B_TR -> C_R
        (2, 2, 2, 2), # A_R * B_BR -> C_R
    ]
    
    for a_co, b_ro, b_co, c_co in block_configs:
        for u_terms, v_terms, w_terms in strassen_2x2:
            for idx, coeff in u_terms:
                r = idx // 2
                c = idx % 2
                U[u_idx(0, a_co, r, c), col] += coeff
            for idx, coeff in v_terms:
                r = idx // 2
                c = idx % 2
                V[v_idx(b_ro, b_co, r, c), col] += coeff
            for idx, coeff in w_terms:
                r = idx // 2
                c = idx % 2
                W[w_idx(c_co, r, c), col] += coeff
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
