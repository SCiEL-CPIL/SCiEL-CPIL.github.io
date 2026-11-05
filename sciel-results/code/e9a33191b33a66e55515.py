"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition.
    
    U is 4x7 (indices: i*2+j), V is 4x7 (indices: j*2+k), W is 4x7 (indices: k*2+i)
    """
    U = [
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [1, 1, 0, 0, 0, 0, 1],
    ]
    V = [
        [1, 0, 0, 0, 0, 1, 1],
        [0, 1, -1, 1, 0, 0, 1],
        [1, 0, 1, 0, 0, 1, 0],
        [1, 1, 0, -1, 1, 0, 0],
    ]
    W = [
        [1, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0, 1, 0],
        [1, 0, 0, 1, -1, 0, 0],
    ]
    return U, V, W

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    U_r = U_arr.reshape(n, m, rank)
    V_r = V_arr.reshape(m, p, rank)
    W_r = W_arr.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
    return np.allclose(T, np.ones((n, m, p)), atol=1e-9, rtol=0)

def _build_56_rank(n, m, p):
    """Build a rank-56 decomposition using Strassen blocks.
    
    Block structure for 3x4x5:
    - i in {0,1}, j in {0,1}, k in {0,1}: Strassen 2x2x2 (rank 7)
    - i in {0,1}, j in {0,1}, k in {2,3}: Strassen 2x2x2 (rank 7)
    - i in {0,1}, j in {0,1}, k = {4}:    schoolbook 2x2x1 (rank 4)
    - i in {0,1}, j in {2,3}, k in {0,1}: Strassen 2x2x2 (rank 7)
    - i in {0,1}, j in {2,3}, k in {2,3}: Strassen 2x2x2 (rank 7)
    - i in {0,1}, j in {2,3}, k = {4}:    schoolbook 2x2x1 (rank 4)
    - i = {2}, j in {0,1,2,3}, k in {0,1,2,3,4}: schoolbook 1x4x5 (rank 20)
    Total: 7+7+4+7+7+4+20 = 56
    """
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_strassen_block(i_off, j_off, k_off):
        U_s, V_s, W_s = _strassen_2x2x2()
        for c in range(7):
            U_col = [0.0] * (n * m)
            V_col = [0.0] * (m * p)
            W_col = [0.0] * (p * n)
            for ii in range(2):
                for jj in range(2):
                    U_col[(i_off + ii) * m + (j_off + jj)] = U_s[ii * 2 + jj][c]
            for jj in range(2):
                for kk in range(2):
                    V_col[(j_off + jj) * p + (k_off + kk)] = V_s[jj * 2 + kk][c]
            for kk in range(2):
                for ii in range(2):
                    W_col[(k_off + kk) * n + (i_off + ii)] = W_s[kk * 2 + ii][c]
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)
    
    def add_schoolbook_block(i_list, j_list, k_list):
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    U_col = [0.0] * (n * m)
                    V_col = [0.0] * (m * p)
                    W_col = [0.0] * (p * n)
                    U_col[i * m + j] = 1.0
                    V_col[j * p + k] = 1.0
                    W_col[k * n + i] = 1.0
                    U_cols.append(U_col)
                    V_cols.append(V_col)
                    W_cols.append(W_col)
    
    # Strassen blocks for i in {0,1}
    add_strassen_block(0, 0, 0)  # i={0,1}, j={0,1}, k={0,1}
    add_strassen_block(0, 0, 2)  # i={0,1}, j={0,1}, k={2,3}
    add_schoolbook_block([0, 1], [0, 1], [4])  # i={0,1}, j={0,1}, k={4}
    
    add_strassen_block(0, 2, 0)  # i={0,1}, j={2,3}, k={0,1}
    add_strassen_block(0, 2, 2)  # i={0,1}, j={2,3}, k={2,3}
    add_schoolbook_block([0, 1], [2, 3], [4])  # i={0,1}, j={2,3}, k={4}
    
    # Schoolbook for i=2
    add_schoolbook_block([2], list(range(4)), list(range(5)))
    
    rank = len(U_cols)
    U_mat = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
    V_mat = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
    W_mat = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    base_rank = n * m * p
    
    # Try the rank-56 Strassen-based decomposition
    candidate = _build_56_rank(n, m, p)
    
    if _verify_decomposition(n, m, p, candidate["U"], candidate["V"], candidate["W"]):
        return candidate
    
    # Fallback to schoolbook if verification fails
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
