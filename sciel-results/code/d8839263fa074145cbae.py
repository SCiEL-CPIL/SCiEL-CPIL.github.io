import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """
    Return 7 Strassen terms for 2x2 x 2x2 multiplication.
    """
    terms = []
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))
    terms.append(([1,0,0,0], [0,0,1,-1], [0,1,0,1]))
    terms.append(([0,0,0,1], [-1,1,0,0], [1,0,1,0]))
    terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
    terms.append([[-1,0,1,0], [1,1,0,0], [0,0,0,1]])
    terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
    return terms

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    
    rank = 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Term 1: A[:,0:1] @ B[0:1,:]
    U[0][col] = 1
    V[0][col] = 1; V[1][col] = 1; V[2][col] = 1; V[3][col] = 1
    W[0][col] = 1; W[1][col] = 1; W[4][col] = 1; W[5][col] = 1
    col += 1
    
    U[3][col] = 1
    V[0][col] = 1; V[1][col] = 1; V[2][col] = 1; V[3][col] = 1
    W[2][col] = 1; W[3][col] = 1; W[6][col] = 1; W[7][col] = 1
    col += 1
    
    # Term 2a
    strassen_terms = _strassen_2x2x2()
    u_row_map_2a = [1, 2, 4, 5]
    v_row_map_2a = [4, 5, 8, 9]
    w_row_map_2a = [0, 1, 2, 3]
    
    for u_vec, v_vec, w_vec in strassen_terms:
        for local_idx, u_val in enumerate(u_vec):
            if u_val != 0: U[u_row_map_2a[local_idx]][col] = u_val
        for local_idx, v_val in enumerate(v_vec):
            if v_val != 0: V[v_row_map_2a[local_idx]][col] = v_val
        for local_idx, w_val in enumerate(w_vec):
            if w_val != 0: W[w_row_map_2a[local_idx]][col] = w_val
        col += 1
    
    # Term 2b
    u_row_map_2b = [1, 2, 4, 5]
    v_row_map_2b = [6, 7, 10, 11]
    w_row_map_2b = [4, 5, 6, 7]
    
    for u_vec, v_vec, w_vec in strassen_terms:
        for local_idx, u_val in enumerate(u_vec):
            if u_val != 0: U[u_row_map_2b[local_idx]][col] = u_val
        for local_idx, v_val in enumerate(v_vec):
            if v_val != 0: V[v_row_map_2b[local_idx]][col] = v_val
        for local_idx, w_val in enumerate(w_vec):
            if w_val != 0: W[w_row_map_2b[local_idx]][col] = w_val
        col += 1
    
    assert col == rank
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_reconstructed = np.zeros((n, p))
    for r in range(rank):
        u_col = U_arr[:, r]
        v_col = V_arr[:, r]
        w_col = W_arr[:, r]
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    s += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
                C_reconstructed[i, k] += s
    
    max_err = np.max(np.abs(C_reconstructed - C_expected))
    if max_err > 1e-10:
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][column] = 1
                    V_fb[j * p + k][column] = 1
                    W_fb[k * n + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
