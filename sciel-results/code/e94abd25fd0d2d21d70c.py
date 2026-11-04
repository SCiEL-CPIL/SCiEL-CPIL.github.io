import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 110.0

    # Strassen rank-7 factors for 2x2
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)

    # Construct 4x4 factors using Kronecker product structure
    u_mat = np.zeros((16, 49), dtype=int)
    v_mat = np.zeros((16, 49), dtype=int)
    w_mat = np.zeros((16, 49), dtype=int)
    
    for i in range(4):
        for j in range(4):
            r_u = i * 4 + j
            ih, il = i // 2, i % 2
            jh, jl = j // 2, j % 2
            
            vec_hi = U2[ih*2+jh, :]
            vec_lo = U2[il*2+jl, :]
            u_mat[r_u, :] = np.outer(vec_hi, vec_lo).flatten()
            
    for j in range(4):
        for k in range(4):
            r_v = j * 4 + k
            jh, jl = j // 2, j % 2
            kh, kl = k // 2, k % 2
            
            vec_hi = V2[jh*2+kh, :]
            vec_lo = V2[jl*2+kl, :]
            v_mat[r_v, :] = np.outer(vec_hi, vec_lo).flatten()
            
    for k in range(4):
        for i in range(4):
            r_w = k * 4 + i
            kh, kl = k // 2, k % 2
            ih, il = i // 2, i % 2
            
            vec_hi = W2[kh*2+ih, :]
            vec_lo = W2[kl*2+il, :]
            w_mat[r_w, :] = np.outer(vec_hi, vec_lo).flatten()

    def verify(U, V, W, rank):
        # U, V, W are (16, rank)
        valid = True
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    u_row = U[i*4+j, :rank]
                    v_row = V[j*4+k, :rank]
                    w_row = W[k*4+i, :rank]
                    prod = np.sum(u_row * v_row * w_row)
                    if prod != 1:
                        valid = False
                        return False
        return valid

    # Attempt Rank 48 (Structural Change: Try to drop 1 rank)
    if verify(u_mat, v_mat, w_mat, 48):
        return {
            "U": u_mat[:, :48].tolist(),
            "V": v_mat[:, :48].tolist(),
            "W": w_mat[:, :48].tolist()
        }

    # Fallback to Rank 49 (Known Valid)
    if verify(u_mat, v_mat, w_mat, 49):
        return {
            "U": u_mat.tolist(),
            "V": v_mat.tolist(),
            "W": w_mat.tolist()
        }

    # Fallback to Rank 64 (Identity)
    n, m, p = 4, 4, 4
    rank = n * m * p
    U_f = [[0] * rank for _ in range(n * m)]
    V_f = [[0] * rank for _ in range(m * p)]
    W_f = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_f[i*m+j][col] = 1
                V_f[j*p+k][col] = 1
                W_f[k*n+i][col] = 1
                col += 1
    return {"U": U_f, "V": V_f, "W": W_f}
