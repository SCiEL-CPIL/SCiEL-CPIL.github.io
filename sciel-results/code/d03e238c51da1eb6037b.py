"""Matrix multiplication 6x6x6 using Strassen on 2x2 blocks in a 3x3 grid."""
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We decompose 6x6 matrices into a 3x3 grid of 2x2 blocks.
    # Each 2x2 multiplication uses Strassen's algorithm (rank 7).
    # Total rank = 3^3 * 7 = 189.
    
    bs = 2  # block size
    nb = 3  # number of blocks per dimension
    
    total_rank = (nb ** 3) * 7
    
    # Initialize U, V, W as lists of lists for integer precision
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen 2x2 multiplication patterns
    # Each pattern: (u_dict, v_dict, w_dict) where keys are (row, col) within 2x2 block
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)(B00)
    # M2 = (A00)(B01-B11)
    # M3 = (A11)(B10-B00)
    # M4 = (A00+A01)(B11)
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    patterns = [
        # M0
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M2
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M3
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col = 0
    for bi in range(nb):
        for bj in range(nb):
            for bk in range(nb):
                for u_pat, v_pat, w_pat in patterns:
                    # U: rows are (i, j) for A[i,j], i in [bi*bs, (bi+1)*bs), j in [bj*bs, (bj+1)*bs)
                    for (ri, rj), val in u_pat.items():
                        i_global = bi * bs + ri
                        j_global = bj * bs + rj
                        idx_u = i_global * m + j_global
                        U[idx_u][col] += val
                    
                    # V: rows are (j, k) for B[j,k], j in [bj*bs, (bj+1)*bs), k in [bk*bs, (bk+1)*bs)
                    for (ri, rj), val in v_pat.items():
                        j_global = bj * bs + ri
                        k_global = bk * bs + rj
                        idx_v = j_global * p + k_global
                        V[idx_v][col] += val
                    
                    # W: rows are (k, i) for C[k,i] following trace convention W[k*n+i, r]
                    # C[i,k] where i in [bi*bs, (bi+1)*bs), k in [bk*bs, (bk+1)*bs)
                    for (ri, rj), val in w_pat.items():
                        i_global = bi * bs + ri
                        k_global = bk * bs + rj
                        idx_w = k_global * n + i_global
                        W[idx_w][col] += val
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
