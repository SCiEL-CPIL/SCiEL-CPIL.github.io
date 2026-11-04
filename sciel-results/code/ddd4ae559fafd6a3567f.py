import numpy as np
import time

def construct():
    n, m, p = 2, 4, 4
    
    T = np.zeros((8, 16, 8), dtype=int)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                x = i * m + k
                y = k * p + j
                z = j * n + i
                T[x, y, z] = 1

    def verify_exact(U, V, W):
        if not U or not V or not W:
            return False
        r = len(U[0])
        if len(U) != 8 or len(V) != 16 or len(W) != 8:
            return False
        if any(len(row) != r for row in U) or any(len(row) != r for row in V) or any(len(row) != r for row in W):
            return False
        U_np = np.array(U, dtype=int)
        V_np = np.array(V, dtype=int)
        W_np = np.array(W, dtype=int)
        T_est = np.einsum('xr,yr,zr->xyz', U_np, V_np, W_np)
        return np.array_equal(T_est, T)

    def get_strassen_28():
        rank = 28
        U = [[0] * rank for _ in range(8)]
        V = [[0] * rank for _ in range(16)]
        W = [[0] * rank for _ in range(8)]
        
        def add_block(col_offset, X_rows, Y_rows, Z_rows):
            U[X_rows[0]][col_offset + 0] += 1
            U[X_rows[3]][col_offset + 0] += 1
            V[Y_rows[0]][col_offset + 0] += 1
            V[Y_rows[3]][col_offset + 0] += 1
            W[Z_rows[0]][col_offset + 0] += 1
            W[Z_rows[3]][col_offset + 0] += 1
            U[X_rows[2]][col_offset + 1] += 1
            U[X_rows[3]][col_offset + 1] += 1
            V[Y_rows[0]][col_offset + 1] += 1
            W[Z_rows[2]][col_offset + 1] += 1
            W[Z_rows[3]][col_offset + 1] -= 1
            U[X_rows[0]][col_offset + 2] += 1
            V[Y_rows[1]][col_offset + 2] += 1
            V[Y_rows[3]][col_offset + 2] -= 1
            W[Z_rows[1]][col_offset + 2] += 1
            W[Z_rows[3]][col_offset + 2] += 1
            U[X_rows[3]][col_offset + 3] += 1
            V[Y_rows[2]][col_offset + 3] += 1
            V[Y_rows[0]][col_offset + 3] -= 1
            W[Z_rows[0]][col_offset + 3] += 1
            W[Z_rows[2]][col_offset + 3] += 1
            U[X_rows[0]][col_offset + 4] += 1
            U[X_rows[1]][col_offset + 4] += 1
            V[Y_rows[3]][col_offset + 4] += 1
            W[Z_rows[0]][col_offset + 4] -= 1
            W[Z_rows[1]][col_offset + 4] += 1
            U[X_rows[2]][col_offset + 5] += 1
            U[X_rows[0]][col_offset + 5] -= 1
            V[Y_rows[0]][col_offset + 5] += 1
            V[Y_rows[1]][col_offset + 5] += 1
            W[Z_rows[3]][col_offset + 5] += 1
            U[X_rows[1]][col_offset + 6] += 1
            U[X_rows[3]][col_offset + 6] -= 1
            V[Y_rows[2]][col_offset + 6] += 1
            V[Y_rows[3]][col_offset + 6] += 1
            W[Z_rows[0]][col_offset + 6] += 1

        add_block(0, [0, 1, 4, 5], [0, 1, 4, 5], [0, 2, 1, 3])
        add_block(7, [2, 3, 6, 7], [8, 9, 12, 13], [0, 2, 1, 3])
        add_block(14, [0, 1, 4, 5], [2, 3, 6, 7], [4, 6, 5, 7])
        add_block(21, [2, 3, 6, 7], [10, 11, 14, 15], [4, 6, 5, 7])
        
        return U, V, W

    best_U, best_V, best_W = get_strassen_28()
    best_rank = 28
    
    if not verify_exact(best_U, best_V, best_W):
        return {"U": best_U, "V": best_V, "W": best_W}

    start_time = time.time()
    deadline = start_time + 130

    # Try to find rank 27 by checking linear dependence of Strassen products
    # Represent each of the 28 rank-1 tensors as a 1024-dim vector
    # and check if they are linearly dependent
    
    U_np = np.array(best_U, dtype=int)
    V_np = np.array(best_V, dtype=int)
    W_np = np.array(best_W, dtype=int)
    
    # Build the 28 rank-1 tensors as vectors
    tensors = []
    for r in range(28):
        t = np.outer(np.outer(U_np[:, r], V_np[:, r]), W_np[:, r]).flatten()
        tensors.append(t)
    
    tensors_np = np.array(tensors, dtype=int)  # shape (28, 1024)
    
    # Check rank of the 28x1024 matrix
    # If rank < 28, we can reduce the rank
    try:
        rank_check = np.linalg.matrix_rank(tensors_np)
        if rank_check < 28:
            # We found a linear dependence, but we still need to construct
            # an explicit rank-27 decomposition. For now, just note it.
            pass
    except:
        pass

    # Random search for rank 27 (very unlikely to succeed, but try)
    import random
    random.seed(12345)
    
    found = False
    for attempt in range(10000):
        if time.time() > deadline:
            break
        U = [[random.randint(-1, 1) for _ in range(27)] for _ in range(8)]
        V = [[random.randint(-1, 1) for _ in range(27)] for _ in range(16)]
        W = [[random.randint(-1, 1) for _ in range(27)] for _ in range(8)]
        if verify_exact(U, V, W):
            found = True
            best_U, best_V, best_W = U, V, W
            best_rank = 27
            break

    return {"U": best_U, "V": best_V, "W": best_W}
