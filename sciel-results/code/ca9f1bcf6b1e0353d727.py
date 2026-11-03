"""Direct construction of rank-minimal (or near-minimal) tensor for 4x4 * 4x8."""
import time

DIMENSIONS = [4, 4, 8]

def _make_strassen_2x2():
    """Rank-7 decomposition for 2x2 * 2x2.
    U: 4x7 (rows indexed i*2+j), V: 4x7 (rows j*2+k), W: 4x7 (rows k*2+i)
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    # Index: U[i*2+j], V[j*2+k], W[k*2+i]
    terms = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 1:1, 2:-1, 3:1}),   # M1
        ({2:1, 3:1}, {0:1},   {0:0, 1:0, 2:1, 3:-1}),      # M2
        ({0:1}, {1:-1, 3:1},  {0:0, 1:1, 2:1, 3:0}),       # M3
        ({3:1}, {0:-1, 2:1},  {0:-1, 1:0, 2:1, 3:0}),      # M4
        ({0:1, 2:1}, {3:1},   {0:1, 1:-1, 2:0, 3:0}),      # M5
        ({0:-1, 2:1}, {0:1, 2:1}, {0:1, 1:0, 2:0, 3:1}),   # M6
        ({1:-1, 3:1}, {1:1, 3:1}, {0:0, 1:1, 2:0, 3:1}),   # M7
    ]
    for t, (u_c, v_c, w_c) in enumerate(terms):
        for idx, val in u_c.items():
            U[idx][t] = val
        for idx, val in v_c.items():
            V[idx][t] = val
        for idx, val in w_c.items():
            W[idx][t] = val
    return U, V, W

def _strassen_4x4_4x4():
    """
    Rank-49 decomposition for 4x4 * 4x4 using Strassen's algorithm.
    
    U: (16, 49) - rows indexed by i*4+j for A[i,j]
    V: (16, 49) - rows indexed by j*4+k for B[j,k]
    W: (16, 49) - rows indexed by k*4+i for C[i,k] (note: output convention is W[k*n+i])
    """
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    u2, v2, w2 = _make_strassen_2x2()
    
    strassen_mults = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)]),
        ([(0, 1)], [(1, 1), (3, -1)]),
        ([(3, 1)], [(2, 1), (0, -1)]),
        ([(0, 1), (2, 1)], [(3, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(1, 1), (3, -1)], [(1, 1), (3, 1)]),
    ]
    
    c_assignments = [
        [(0, 1), (1, 1), (2, -1), (3, 1)],
        [(2, 1), (3, -1)],
        [(1, 1), (2, 1)],
        [(0, -1), (2, 1)],
        [(0, 1), (1, -1)],
        [(0, 1), (3, 1)],
        [(1, 1), (3, 1)],
    ]
    
    block_pos = {
        0: (0, 0),
        1: (0, 2),
        2: (2, 0),
        3: (2, 2),
    }
    
    for m in range(7):
        a_terms, b_terms = strassen_mults[m]
        c_terms = c_assignments[m]
        
        for (a_block, a_sign) in a_terms:
            a_row_off, a_col_off = block_pos[a_block]
            for (b_block, b_sign) in b_terms:
                b_row_off, b_col_off = block_pos[b_block]
                for (c_block, c_sign) in c_terms:
                    c_row_off, c_col_off = block_pos[c_block]
                    
                    scale = a_sign * b_sign * c_sign
                    if scale == 0:
                        continue
                    
                    for r in range(7):
                        col_idx = m * 7 + r
                        
                        for i in range(2):
                            for j in range(2):
                                u2_val = u2[i*2+j][r]
                                if u2_val != 0:
                                    U[(a_row_off+i)*4 + (a_col_off+j)][col_idx] += scale * u2_val
                        
                        for j in range(2):
                            for k in range(2):
                                v2_val = v2[j*2+k][r]
                                if v2_val != 0:
                                    V[(b_row_off+j)*4 + (b_col_off+k)][col_idx] += scale * v2_val
                        
                        for k in range(2):
                            for i in range(2):
                                w2_val = w2[k*2+i][r]
                                if w2_val != 0:
                                    W[(c_col_off+k)*4 + (c_row_off+i)][col_idx] += scale * w2_val
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    deadline = time.time() + 100
    
    # Use 2x Strassen 4x4*4x4 for the two halves of B (columns 0-3 and 4-7)
    # Total rank = 98
    
    U4, V4, W4 = _strassen_4x4_4x4()
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]  # 16 x 98
    V = [[0] * rank for _ in range(m * p)]  # 32 x 98
    W = [[0] * rank for _ in range(p * n)]  # 32 x 98
    
    # First half: B columns 0-3 (k=0,1,2,3)
    for i in range(16):
        for r in range(49):
            U[i][r] = U4[i][r]
            for j in range(4):
                for k in range(4):
                    V[j * 8 + k][r] = V4[j * 4 + k][r]
            for k in range(4):
                for i2 in range(4):
                    W[k * 4 + i2][r] = W4[k * 4 + i2][r]
    
    # Second half: B columns 4-7 (k=4,5,6,7)
    for i in range(16):
        for r in range(49):
            r2 = 49 + r
            U[i][r2] = U4[i][r]
            for j in range(4):
                for k in range(4):
                    V[j * 8 + (k + 4)][r2] = V4[j * 4 + k][r]
            for k in range(4):
                for i2 in range(4):
                    W[(k + 4) * 4 + i2][r2] = W4[k * 4 + i2][r]
    
    # Verify the full decomposition if time permits
    if time.time() < deadline - 20:
        try:
            import numpy as np
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            for i in range(4):
                for j in range(4):
                    for k in range(8):
                        val = sum(U_arr[i*4+j, r] * V_arr[j*8+k, r] * W_arr[k*4+i, r] for r in range(rank))
                        if abs(val - 1.0) > 1e-10:
                            raise ValueError(f"Full decomposition mismatch at ({i},{j},{k}): got {val}")
        except Exception as e:
            # If verification fails, fall back to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
            return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
