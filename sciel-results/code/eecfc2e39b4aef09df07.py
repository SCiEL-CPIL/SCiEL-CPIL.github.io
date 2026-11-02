"""Matrix multiplication 4x5x5 with rank-reduced decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def _schoolbook(n, m, p, i_off=0, j_off=0, k_off=0):
    """Generate schoolbook decomposition for n×m × m×p sub-problem."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[(i + i_off) * m + (j + j_off)][col] = 1
                V[(j + j_off) * p + (k + k_off)][col] = 1
                W[(k + k_off) * n + (i + i_off)][col] = 1
                col += 1
    return U, V, W

def _add_decompositions(decomps):
    """Add multiple decompositions (direct sum)."""
    total_rank = sum(d[0].__len__() if isinstance(d[0], list) else len(d[0]) for d in decomps)
    # Actually, let's just concatenate
    U_list = []
    V_list = []
    W_list = []
    for U, V, W in decomps:
        U_list.append(U)
        V_list.append(V)
        W_list.append(W)
    # Concatenate along the rank dimension (columns)
    max_U_rows = max(len(U) for U in U_list)
    max_V_rows = max(len(V) for V in V_list)
    max_W_rows = max(len(W) for W in W_list)
    
    total_rank = sum(len(U[0]) for U in U_list)
    
    U_out = [[0] * total_rank for _ in range(max_U_rows)]
    V_out = [[0] * total_rank for _ in range(max_V_rows)]
    W_out = [[0] * total_rank for _ in range(max_W_rows)]
    
    col_off = 0
    for U, V, W in decomps:
        n_rows_U = len(U)
        n_rows_V = len(V)
        n_rows_W = len(W)
        n_cols = len(U[0])
        for i in range(n_rows_U):
            for c in range(n_cols):
                U_out[i][col_off + c] = U[i][c]
        for i in range(n_rows_V):
            for c in range(n_cols):
                V_out[i][col_off + c] = V[i][c]
        for i in range(n_rows_W):
            for c in range(n_cols):
                W_out[i][col_off + c] = W[i][c]
        col_off += n_cols
    
    return U_out, V_out, W_out

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 5
    
    # Split: (3,5,5) + (1,5,5)
    # (3,5,5) = (3,3,5) + (3,2,5)
    # (3,3,5) = (3,3,3) + (3,3,2)
    # (3,3,3) = rank 19 decomposition
    # Others = schoolbook
    
    # For now, use schoolbook for all sub-problems to get rank 92
    # (3,3,3): 27, (3,3,2): 18, (3,2,5): 30, (1,5,5): 25 => total 100
    # Wait, that's still 100. I need the 19-rank for 3x3x3.
    
    # Let me just use schoolbook for everything first to verify the framework,
    # then improve the 3x3x3 part.
    
    # Actually, let me just do schoolbook for the whole thing to start:
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
# EVOLVE-BLOCK-END
