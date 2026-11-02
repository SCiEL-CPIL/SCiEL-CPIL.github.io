import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    num_u = n * m  # 6
    num_v = m * p  # 21
    num_w = p * n  # 14

    def verify(U, V, W):
        R = U.shape[1]
        u_indices = []
        v_indices = []
        w_indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_indices.append(i * m + j)
                    v_indices.append(j * p + k)
                    w_indices.append(k * n + i)
        u_indices = np.array(u_indices)
        v_indices = np.array(v_indices)
        w_indices = np.array(w_indices)
        
        U_sel = U[u_indices, :]
        V_sel = V[v_indices, :]
        W_sel = W[w_indices, :]
        vals = np.sum(U_sel * V_sel * W_sel, axis=1)
        return np.max(np.abs(vals - 1.0))
    
    R = 39
    U = np.zeros((num_u, R))
    V = np.zeros((num_v, R))
    W = np.zeros((num_w, R))
    
    def add_strassen(col_start, i_off, j_off, k_off):
        """Add Strassen 2x2x2 terms. i_off,j_off,k_off are offsets into the i,j,k spaces."""
        alphas = [
            [[1, 0], [0, 1]],
            [[0, 0], [1, 1]],
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 1], [-1, 0]],
        ]
        betas = [
            [[1, 1], [1, 1]],
            [[1, 0], [0, 0]],
            [[0, 1], [-1, 1]],
            [[0, -1], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[0, 1], [1, 1]],
        ]
        gammas = [
            [[1, 0], [0, 1]],
            [[0, 0], [1, -1]],
            [[0, 1], [1, 1]],
            [[1, 0], [1, 0]],
            [[-1, 1], [0, 0]],
            [[0, 0], [0, 1]],
            [[1, 0], [0, 0]],
        ]
        
        for s in range(7):
            col = col_start + s
            for ii in range(2):
                for jj in range(2):
                    U[(i_off + ii) * m + (j_off + jj), col] = alphas[s][ii][jj]
            for jj in range(2):
                for kk in range(2):
                    V[(j_off + jj) * p + (k_off + kk), col] = betas[s][jj][kk]
            for kk in range(2):
                for ii in range(2):
                    W[(k_off + kk) * n + (i_off + ii), col] = gammas[s][ii][kk]
        return col_start + 7
    
    col = 0
    # Part 1: j=0,1, k=0,1
    col = add_strassen(col, i_off=0, j_off=0, k_off=0)  # col=7
    # Part 2: j=0,1, k=2,3
    col = add_strassen(col, i_off=0, j_off=0, k_off=2)  # col=14
    # Part 3: j=0,1, k=4,5 (Strassen) + k=6 (schoolbook)
    col = add_strassen(col, i_off=0, j_off=0, k_off=4)  # col=21
    for ii in range(2):
        for jj in range(2):
            U[ii * m + jj, col] = 1
            V[jj * p + 6, col] = 1
            W[6 * n + ii, col] = 1
            col += 1
    # col=25
    
    # Part 4: j=2, k=0..6 (schoolbook)
    for ii in range(2):
        for kk in range(7):
            U[ii * m + 2, col] = 1
            V[2 * p + kk, col] = 1
            W[kk * n + ii, col] = 1
            col += 1
    # col=39
    
    assert col == R, f"Expected {R}, got {col}"
    
    err = verify(U, V, W)
    if err < 1e-10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback
    R42 = n * m * p
    U42 = np.zeros((num_u, R42))
    V42 = np.zeros((num_v, R42))
    W42 = np.zeros((num_w, R42))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U42[i * m + j, c] = 1
                V42[j * p + k, c] = 1
                W42[k * n + i, c] = 1
                c += 1
    return {"U": U42.tolist(), "V": V42.tolist(), "W": W42.tolist()}
