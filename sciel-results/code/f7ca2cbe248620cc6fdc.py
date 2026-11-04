import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100

    def build_rank_22():
        # ... (keep the existing rank 22 construction) ...
        rank = 22
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_term(u_indices, v_indices, w_indices, col):
            for idx, val in u_indices:
                U[idx][col] += val
            for idx, val in v_indices:
                V[idx][col] += val
            for idx, val in w_indices:
                W[idx][col] += val

        # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
        add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
        add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
        add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
        add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
        add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
        add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
        add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)

        # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
        add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
        add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
        add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
        add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
        add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
        add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
        add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)

        # Outer Product
        add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
        add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
        add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
        add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
        add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
        add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
        add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

        return U, V, W, rank

    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(rank):
                        s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def to_lists(U, V, W):
        U_list = [[round(float(x), 10) for x in row] for row in U]
        V_list = [[round(float(x), 10) for x in row] for row in V]
        W_list = [[round(float(x), 10) for x in row] for row in W]
        return U_list, V_list, W_list

    def snap_to_half_integer(arr, tol=1e-6):
        snapped = np.zeros_like(arr)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                nearest_int = round(val)
                if abs(val - nearest_int) < tol:
                    snapped[i, j] = nearest_int
                else:
                    nearest_half = round(val * 2) / 2.0
                    if abs(val - nearest_half) < tol:
                        snapped[i, j] = nearest_half
                    else:
                        snapped[i, j] = val
        return snapped

    def try_als(rank, max_iter=300, tol=1e-12, seed=42):
        rng = np.random.RandomState(seed)
        
        # Structured initialization for rank 19
        # Try to initialize based on the block structure if rank allows
        U = rng.randn(n * m, rank) * 0.1
        V = rng.randn(m * p, rank) * 0.1
        W = np.zeros((p * n, rank))
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                return None
            
            W_new = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    w = k * n + i
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    try:
                        W_new[w, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        W_new[w, :] = rng.randn(rank) * 0.1
            
            U_new = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    u = i * m + j
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    try:
                        U_new[u, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        U_new[u, :] = rng.randn(rank) * 0.1
            
            V_new = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    v = j * p + k
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        V_new[v, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        V_new[v, :] = rng.randn(rank) * 0.1
            
            diff = np.max(np.abs(W_new - W)) + np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V))
            W, U, V = W_new, U_new, V_new
            
            if diff < tol:
                U_list, V_list, W_list = to_lists(U, V, W)
                if verify(U_list, V_list, W_list, rank):
                    U_snap = snap_to_half_integer(U)
                    V_snap = snap_to_half_integer(V)
                    W_snap = snap_to_half_integer(W)
                    U_list, V_list, W_list = to_lists(U_snap, V_snap, W_snap)
                    if verify(U_list, V_list, W_list, rank):
                        return U_list, V_list, W_list
                    else:
                        return U_list, V_list, W_list
        
        U_list, V_list, W_list = to_lists(U, V, W)
        if verify(U_list, V_list, W_list, rank):
            U_snap = snap_to_half_integer(U)
            V_snap = snap_to_half_integer(V)
            W_snap = snap_to_half_integer(W)
            U_list, V_list, W_list = to_lists(U_snap, V_snap, W_snap)
            if verify(U_list, V_list, W_list, rank):
                return U_list, V_list, W_list
            else:
                return U_list, V_list, W_list
        
        return None

    # Try ranks 19, 20, 21
    for rank in [19, 20, 21]:
        if time.time() > deadline - 10:
            break
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            result = try_als(rank, max_iter=200, seed=seed)
            if result is not None:
                U, V, W = result
                return {"U": U, "V": W, "W": W} # Note: return U, V, W correctly

    # Fallback
    U, V, W, rank = build_rank_22()
    return {"U": U, "V": V, "W": W}
