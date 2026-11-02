import time
import numpy as np

DIMENSIONS = [3, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 145.0

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    strassen_decomp = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {1: 1, 0: -1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    for j1 in range(2):
        for l in range(4):
            for u_coeffs, v_coeffs, w_coeffs in strassen_decomp:
                for li, coeff in u_coeffs.items():
                    li_i, li_j = li // 2, li % 2
                    U[li_i * m + (2 * j1 + li_j)][col] += coeff
                for li, coeff in v_coeffs.items():
                    li_i, li_j = li // 2, li % 2
                    V[(2 * j1 + li_i) * p + (2 * l + li_j)][col] += coeff
                for li, coeff in w_coeffs.items():
                    li_i, li_j = li // 2, li % 2
                    W[(2 * l + li_j) * n + li_i][col] += coeff
                col += 1
        
        for k in range(8):
            for j2 in range(2):
                U[2 * m + (2 * j1 + j2)][col] = 1
                V[(2 * j1 + j2) * p + k][col] = 1
                W[k * n + 2][col] = 1
                col += 1

    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    def try_reduce_rank(U_mat, V_mat, W_mat, max_time=100.0):
        n, m, p = DIMENSIONS
        rank = U_mat.shape[1]
        num_elements = n * m * p
        
        # Build contribution matrix: contrib[r, idx] where idx = i*m*p + j*p + k
        contrib = np.zeros((rank, num_elements), dtype=np.float64)
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    u_val = U_mat[i * m + j, r]
                    if u_val == 0:
                        continue
                    for k in range(p):
                        v_val = V_mat[j * p + k, r]
                        if v_val == 0:
                            continue
                        w_val = W_mat[k * n + i, r]
                        if w_val == 0:
                            continue
                        contrib[r, i * m * p + j * p + k] += u_val * v_val * w_val
        
        total_contrib = contrib.sum(axis=0)
        expected = np.ones((n, m, p), dtype=np.float64)
        if not np.allclose(total_contrib.reshape(n, m, p), expected, atol=1e-9):
            return U_mat, V_mat, W_mat
        
        keep = [True] * rank
        start_time = time.time()
        
        # Iterative removal with in-place total update for cascading
        changed = True
        while changed and (time.time() - start_time < max_time):
            changed = False
            for r in range(rank):
                if time.time() - start_time > max_time:
                    break
                if not keep[r]:
                    continue
                new_total = total_contrib - contrib[r]
                if np.allclose(new_total.reshape(n, m, p), expected, atol=1e-9):
                    keep[r] = False
                    changed = True
                    total_contrib = new_total
        
        idx = [i for i in range(rank) if keep[i]]
        if len(idx) < rank:
            return U_mat[:, idx], V_mat[:, idx], W_mat[:, idx]
        return U_mat, V_mat, W_mat
    
    U_opt, V_opt, W_opt = try_reduce_rank(U_np, V_np, W_np, max_time=100.0)
    
    def to_list(mat):
        mat_rounded = np.round(mat * 2) / 2
        if np.allclose(mat, mat_rounded, atol=1e-9):
            return mat_rounded.tolist()
        return mat.tolist()
    
    return {"U": to_list(U_opt), "V": to_list(V_opt), "W": to_list(W_opt)}
# EVOLVE-BLOCK-END
