import time
import numpy as np

DIMENSIONS = [3, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 130.0

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 88
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
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
        # Top part: 2x2x8
        for l in range(4):
            for u_coeffs, v_coeffs, w_coeffs in strassen_decomp:
                for li, coeff in u_coeffs.items():
                    li_i, li_j = li // 2, li % 2
                    global_i = li_i
                    global_j = 2 * j1 + li_j
                    U[global_i * m + global_j, col] += coeff
                
                for li, coeff in v_coeffs.items():
                    li_i, li_j = li // 2, li % 2
                    global_j = 2 * j1 + li_i
                    global_k = 2 * l + li_j
                    V[global_j * p + global_k, col] += coeff
                
                for li, coeff in w_coeffs.items():
                    li_i, li_j = li // 2, li % 2
                    global_i = li_i
                    global_k = 2 * l + li_j
                    W[global_k * n + global_i, col] += coeff
                
                col += 1
        
        # Bot part: 1x2x8
        for k in range(8):
            for j2 in range(2):
                global_i = 2
                global_j = 2 * j1 + j2
                U[global_i * m + global_j, col] = 1
                
                global_j = 2 * j1 + j2
                global_k = k
                V[global_j * p + global_k, col] = 1
                
                global_i = 2
                global_k = k
                W[global_k * n + global_i, col] = 1
                
                col += 1

    # Verify
    def verify(U_mat, V_mat, W_mat):
        # Reshape to 3D
        U_3d = U_mat.reshape(n, m, -1)
        V_3d = V_mat.reshape(m, p, -1)
        W_3d = W_mat.reshape(p, n, -1)
        
        # Compute T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # Note: W is indexed by k, i, r. In the tensor T[i,j,k], the third index is k.
        # The term is U[i,j,r] * V[j,k,r] * W[k,i,r].
        # np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        return True

    if not verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to reduce rank
    def try_reduce_rank(U_mat, V_mat, W_mat, max_time=20.0):
        rank = U_mat.shape[1]
        keep = [True] * rank
        start_time = time.time()
        
        # Precompute U_3d, V_3d, W_3d for faster checks? 
        # Slicing 3D arrays is fast.
        
        for r in range(rank):
            if time.time() - start_time > max_time:
                break
            if not keep[r]:
                continue
            
            keep[r] = False
            idx = [i for i in range(rank) if keep[i]]
            if len(idx) == 0:
                keep[r] = True
                continue
                
            U_red = U_mat[:, idx]
            V_red = V_mat[:, idx]
            W_red = W_mat[:, idx]
            
            U_3d = U_red.reshape(n, m, -1)
            V_3d = V_red.reshape(m, p, -1)
            W_3d = W_red.reshape(p, n, -1)
            
            T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
            
            if not np.allclose(T, 1.0, atol=1e-9):
                keep[r] = True
        
        idx = [i for i in range(rank) if keep[i]]
        if len(idx) < rank:
            return U_mat[:, idx], V_mat[:, idx], W_mat[:, idx]
        return U_mat, V_mat, W_mat

    U_opt, V_opt, W_opt = try_reduce_rank(U, V, W, max_time=30.0)
    
    def to_list(mat):
        mat_rounded = np.round(mat * 2) / 2
        if np.allclose(mat, mat_rounded, atol=1e-9):
            return mat_rounded.tolist()
        else:
            return mat.tolist()
    
    return {"U": to_list(U_opt), "V": to_list(V_opt), "W": to_list(W_opt)}
# EVOLVE-BLOCK-END
