"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]
import time
import numpy as np

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return Strassen's 7-term decomposition for 2x2 x 2x2.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Index convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (a00+a11)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (a10+a11)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3: a00(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4: a11(b10-b00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5: (a00+a01)b11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (a10-a00)(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (a01-a11)(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = 1
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check shapes
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Dimensions
    # n=4, m=4, p=8
    # Block sizes: 2, 2, 2
    bn, bm, bp = 2, 2, 2
    
    num_blocks_i = n // bn
    num_blocks_j = m // bm
    num_blocks_k = p // bp
    
    block_rank = 7
    total_rank = num_blocks_i * num_blocks_j * num_blocks_k * block_rank
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    base_U, base_V, base_W = _strassen_2x2()
    
    for bi in range(num_blocks_i):
        for bj in range(num_blocks_j):
            for bk in range(num_blocks_k):
                # Block index
                b_idx = (bi * num_blocks_j + bj) * num_blocks_k + bk
                col_offset = b_idx * block_rank
                
                for r in range(block_rank):
                    # U: rows (i, j)
                    for li in range(bn):
                        for lj in range(bm):
                            i = bi * bn + li
                            j = bj * bm + lj
                            u_val = base_U[li * bm + lj][r]
                            if u_val != 0:
                                U[i * m + j][col_offset + r] = u_val
                                
                    # V: rows (j, k)
                    for lj in range(bm):
                        for lk in range(bp):
                            j = bj * bm + lj
                            k = bk * bp + lk
                            v_val = base_V[lj * bp + lk][r]
                            if v_val != 0:
                                V[j * p + k][col_offset + r] = v_val
                                
                    # W: rows (k, i)
                    for lk in range(bp):
                        for li in range(bn):
                            k = bk * bp + lk
                            i = bi * bn + li
                            w_val = base_W[lk * bn + li][r]
                            if w_val != 0:
                                W[k * n + i][col_offset + r] = w_val
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return result
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        rank = n * m * p
        U_s = [[0] * rank for _ in range(n * m)]
        V_s = [[0] * rank for _ in range(m * p)]
        W_s = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][col] = 1
                    V_s[j * p + k][col] = 1
                    W_s[k * n + i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}

# EVOLVE-BLOCK-END
