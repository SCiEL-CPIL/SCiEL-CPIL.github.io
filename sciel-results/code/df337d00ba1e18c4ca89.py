import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = 4, 4, 4
    rank = 49
    
    # 1. Define 2x2 Strassen factors
    # U2: rows (0,0), (0,1), (1,0), (1,1) -> indices 0, 1, 2, 3
    # V2: rows (0,0), (0,1), (1,0), (1,1) -> indices 0, 1, 2, 3
    # W2: rows (0,0), (0,1), (1,0), (1,1) -> indices 0, 1, 2, 3
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    # 2. Compute Kronecker products
    U_kron = np.kron(U2, U2)
    V_kron = np.kron(V2, V2)
    W_kron = np.kron(W2, W2)
    
    # 3. Permute rows to match standard row-major ordering (i*4+j)
    # The Kronecker product `np.kron(A, B)` where A and B are 4x7 results in a 16x49 matrix.
    # The row index in the result is `r_A * 4 + r_B` where r_A is row index in A, r_B in B.
    # We need to map this to the standard index `S = i * 4 + j` for U, `S = j * 4 + k` for V, `S = k * 4 + i` for W.
    
    # For U (i, j):
    # i = i_out * 2 + i_in
    # j = j_out * 2 + j_in
    # Standard S = i * 4 + j
    # Kron row index K = (i_out * 2 + j_out) * 4 + (i_in * 2 + j_in)
    # We need to find K for each S.
    
    def get_permutation_U():
        perm = []
        for S in range(16):
            i = S // 4
            j = S % 4
            i_out, i_in = i // 2, i % 2
            j_out, j_in = j // 2, j % 2
            K = (i_out * 2 + j_out) * 4 + (i_in * 2 + j_in)
            perm.append(K)
        return perm

    def get_permutation_V():
        perm = []
        for S in range(16):
            j = S // 4
            k = S % 4
            j_out, j_in = j // 2, j % 2
            k_out, k_in = k // 2, k % 2
            K = (j_out * 2 + k_out) * 4 + (j_in * 2 + k_in)
            perm.append(K)
        return perm

    def get_permutation_W():
        perm = []
        for S in range(16):
            k = S // 4
            i = S % 4
            k_out, k_in = k // 2, k % 2
            i_out, i_in = i // 2, i % 2
            K = (k_out * 2 + i_out) * 4 + (k_in * 2 + i_in)
            perm.append(K)
        return perm

    perm_U = get_permutation_U()
    perm_V = get_permutation_V()
    perm_W = get_permutation_W()
    
    U = U_kron[perm_U, :]
    V = V_kron[perm_V, :]
    W = W_kron[perm_W, :]
    
    # 4. Verification
    if time.time() - start_time > deadline - 5.0:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    max_err = 0.0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = np.sum(U[i*4+j, :] * V[j*4+k, :] * W[k*4+i, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
                    
    if max_err > 1e-9:
        # Fallback to schoolbook rank 64
        rank_fb = 64
        U_fb = np.zeros((16, rank_fb), dtype=float)
        V_fb = np.zeros((16, rank_fb), dtype=float)
        W_fb = np.zeros((16, rank_fb), dtype=float)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_fb[i*4+j, col] = 1
                    V_fb[j*4+k, col] = 1
                    W_fb[k*4+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
