import time
import numpy as np

def construct():
    t0 = time.time()
    deadline = 120.0

    n, m, p = 4, 4, 4

    # Strassen rank-7 factors for 2x2 matrix multiplication
    # U2: rows (i,j) -> i*2+j
    # V2: rows (j,k) -> j*2+k
    # W2: rows (k,i) -> k*2+i
    U2 = np.array([
        [1,  0,  1,  0,  1, -1,  0],
        [0,  0,  0,  0,  1,  0,  1],
        [0,  1,  0,  0,  0,  1,  0],
        [1,  1,  0,  1,  0,  0, -1]
    ], dtype=int)

    V2 = np.array([
        [1,  1,  0, -1,  0,  1,  0],
        [0,  0,  1,  0,  0,  1,  0],
        [0,  0,  0,  1,  0,  0,  1],
        [1,  0, -1,  0,  1,  0,  1]
    ], dtype=int)

    W2 = np.array([
        [1,  0,  0,  1, -1,  0,  1],
        [0,  1,  0,  1,  0,  0,  0],
        [0,  0,  1,  0,  1,  0,  0],
        [1, -1, 1,  0,  0,  1,  0]
    ], dtype=int)

    # Kronecker product gives 4x4 from 2x2: rank 49
    U_kron = np.kron(U2, U2)  # shape (16, 49)
    V_kron = np.kron(V2, V2)  # shape (16, 49)
    W_kron = np.kron(W2, W2)  # shape (16, 49)

    # Permute rows to standard row-major ordering
    # U: (i, j) -> i*4+j. i = i_out*2 + i_in, j = j_out*2 + j_in
    # Kron row index: (i_out*2 + j_out)*4 + (i_in*2 + j_in)
    
    def get_permutation_for_U():
        perm = []
        for S in range(16):
            i = S // 4
            j = S % 4
            i_out, i_in = i // 2, i % 2
            j_out, j_in = j // 2, j % 2
            block = i_out * 2 + j_out
            sub = i_in * 2 + j_in
            perm.append(block * 4 + sub)
        return perm

    def get_permutation_for_V():
        perm = []
        for S in range(16):
            j = S // 4
            k = S % 4
            j_out, j_in = j // 2, j % 2
            k_out, k_in = k // 2, k % 2
            block = j_out * 2 + k_out
            sub = j_in * 2 + k_in
            perm.append(block * 4 + sub)
        return perm

    def get_permutation_for_W():
        perm = []
        for S in range(16):
            k = S // 4
            i = S % 4
            k_out, k_in = k // 2, k % 2
            i_out, i_in = i // 2, i % 2
            block = k_out * 2 + i_out
            sub = k_in * 2 + i_in
            perm.append(block * 4 + sub)
        return perm

    perm_U = get_permutation_for_U()
    perm_V = get_permutation_for_V()
    perm_W = get_permutation_for_W()

    U = U_kron[perm_U, :].astype(float)
    V = V_kron[perm_V, :].astype(float)
    W = W_kron[perm_W, :].astype(float)

    # Verification function
    def verify(U, V, W):
        max_err = 0.0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    val = np.dot(U[i * 4 + j, :], V[j * 4 + k, :] * W[k * 4 + i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    # Initial verification
    if time.time() - t0 > deadline - 10.0:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    max_err = verify(U, V, W)
    if max_err > 1e-9:
        # Fallback: schoolbook rank 64
        rank_fb = 64
        U_fb = np.zeros((16, rank_fb), dtype=float)
        V_fb = np.zeros((16, rank_fb), dtype=float)
        W_fb = np.zeros((16, rank_fb), dtype=float)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_fb[i * 4 + j, col] = 1.0
                    V_fb[j * 4 + k, col] = 1.0
                    W_fb[k * 4 + i, col] = 1.0
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    # Attempt rank reduction via column elimination
    current_rank = 49
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    remaining = list(range(49))
    
    # Greedy elimination
    for col_to_test in range(49):
        if time.time() - t0 > deadline - 15.0:
            break
        if col_to_test not in remaining:
            continue
        
        keep = [c for c in remaining if c != col_to_test]
        if len(keep) == 0:
            continue
        
        U_cand = U_cur[:, keep]
        V_cand = V_cur[:, keep]
        W_cand = W_cur[:, keep]
        
        # Check validity
        valid = True
        for i in range(4):
            if not valid:
                break
            for j in range(4):
                if not valid:
                    break
                for k in range(4):
                    ui = U_cand[i * 4 + j, :]
                    vj = V_cand[j * 4 + k, :]
                    wk = W_cand[k * 4 + i, :]
                    val = np.dot(ui, vj * wk)
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
        
        if valid:
            U_cur = U_cand
            V_cur = V_cand
            W_cur = W_cand
            remaining = keep
            current_rank = len(keep)
    
    if time.time() - t0 > deadline - 5.0:
        return {"U": U_cur.tolist(), "V": V_cur.tolist(), "W": W_cur.tolist()}
    
    # Final verification
    max_err = verify(U_cur, V_cur, W_cur)
    if max_err > 1e-9:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U_cur.tolist(), "V": V_cur.tolist(), "W": W_cur.tolist()}

if __name__ == "__main__":
    construct()
