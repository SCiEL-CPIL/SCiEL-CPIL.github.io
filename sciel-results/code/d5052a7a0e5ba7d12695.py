import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 5, 5, 5
    
    # Block sizes: 5 = 2 + 2 + 1
    block_sizes = [2, 2, 1]
    offsets = [0]
    for s in block_sizes:
        offsets.append(offsets[-1] + s)
    
    # Strassen factors for 2x2x2
    # Each factor is (u_flat, v_flat, w_flat) where flat index = row*2 + col
    strassen_factors = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Calculate total rank
    total_rank = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni, mj, pk = block_sizes[i], block_sizes[j], block_sizes[k]
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk
    
    # Allocate using numpy arrays for speed
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if time.time() > deadline:
                    break
                ni, mj, pk = block_sizes[i], block_sizes[j], block_sizes[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    for (uf, vf, wf) in strassen_factors:
                        for idx in range(4):
                            lr, lc = divmod(idx, 2)
                            U_global = (offsets[i] + lr) * m + (offsets[j] + lc)
                            V_global = (offsets[j] + lr) * p + (offsets[k] + lc)
                            W_global = (offsets[k] + lr) * n + (offsets[i] + lc)
                            U[U_global, col] += uf[idx]
                            V[V_global, col] += vf[idx]
                            W[W_global, col] += wf[idx]
                        col += 1
                else:
                    for li in range(ni):
                        for lj in range(mj):
                            for lk in range(pk):
                                U_idx = (offsets[i] + li) * m + (offsets[j] + lj)
                                V_idx = (offsets[j] + lj) * p + (offsets[k] + lk)
                                W_idx = (offsets[k] + lk) * n + (offsets[i] + li)
                                U[U_idx, col] = 1.0
                                V[V_idx, col] = 1.0
                                W[W_idx, col] = 1.0
                                col += 1
    
    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"
    
    # Fast numpy-based verification using einsum
    def _verify_numpy(U_arr, V_arr, W_arr):
        rank = U_arr.shape[1]
        U_mat = U_arr.reshape(n, m, rank)
        V_mat = V_arr.reshape(m, p, rank)
        W_mat = W_arr.reshape(p, n, rank)
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        T = np.einsum('ijc,jkc,kic->ijk', U_mat, V_mat, W_mat)
        return np.allclose(T, 1.0, atol=1e-9)
    
    if _verify_numpy(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook (rank 125)
    U_fallback = np.zeros((25, 125))
    V_fallback = np.zeros((25, 125))
    W_fallback = np.zeros((25, 125))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fallback[i * m + j, col] = 1.0
                V_fallback[j * p + k, col] = 1.0
                W_fallback[k * n + i, col] = 1.0
                col += 1
    return {"U": U_fallback.tolist(), "V": V_fallback.tolist(), "W": W_fallback.tolist()}
