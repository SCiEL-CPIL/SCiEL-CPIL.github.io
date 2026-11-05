import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    DEADLINE = 130

    def elapsed():
        return time.time() - t0

    def verify(U, V, W, seed=12345, trials=12):
        try:
            Ua = np.array(U, dtype=float)
            Va = np.array(V, dtype=float)
            Wa = np.array(W, dtype=float)
            rank = Ua.shape[1]
            rng = np.random.RandomState(seed)
            for _ in range(trials):
                A = rng.rand(n, m)
                B = rng.rand(m, p)
                C_expected = A @ B
                x = (A.reshape(n * m) @ Ua)
                y = (B.reshape(m * p) @ Va)
                prod = x * y
                C_computed = (Wa * prod[None, :]).sum(axis=1).reshape(p, n)
                C_expected_T = C_expected.T
                if not np.allclose(C_computed, C_expected_T, atol=1e-9, rtol=1e-9):
                    return False
            return True
        except Exception:
            return False

    def reduce_rank(U, V, W, max_iter=800):
        """Attempt to reduce rank via sparse null-space elimination."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        r0 = Ua.shape[1]
        best_r = r0
        best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()

        for _ in range(max_iter):
            if elapsed() > DEADLINE:
                break
            r = Ua.shape[1]
            eliminated = False
            for c in range(r):
                if eliminated:
                    break
                U_others = np.delete(Ua, c, axis=1)
                V_others = np.delete(Va, c, axis=1)
                W_others = np.delete(Wa, c, axis=1)

                coeff_u, _, _, _ = np.linalg.lstsq(U_others, Ua[:, c], rcond=None)
                pred_v = V_others @ coeff_u
                if np.allclose(pred_v, Va[:, c], atol=1e-8, rtol=1e-8):
                    pred_w = W_others @ coeff_u
                    if np.allclose(pred_w, Wa[:, c], atol=1e-8, rtol=1e-8):
                        Ua = np.delete(Ua, c, axis=1)
                        Va = np.delete(Va, c, axis=1)
                        Wa = np.delete(Wa, c, axis=1)
                        eliminated = True
                        break

            if not eliminated:
                break

        if Ua.shape[1] < best_r:
            best_r = Ua.shape[1]
            best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()

        return best_U, best_V, best_W, best_r

    def to_list(arr):
        return arr.tolist()

    # Winograd F(4) for 4x4x4 (rank 16) + rank-4 for 4x4x1 = rank 20
    rank = 20
    U = [[0] * rank for _ in range(n * m)]  # 16 x 20
    V = [[0] * rank for _ in range(m * p)]  # 20 x 20
    W = [[0] * rank for _ in range(p * n)]  # 20 x 20

    # Winograd F(4) transform
    T = [[1, 0, 1, 0],
         [0, 1, 0, -1],
         [1, 0, -1, 0],
         [0, 1, 0, 1]]

    col = 0
    for r in range(4):
        for s in range(4):
            for i in range(4):
                for j in range(4):
                    U[i * 4 + j][col] += T[r][i] * T[s][j]
            for j in range(4):
                for k in range(4):
                    V[j * 5 + k][col] += T[r][j] * T[s][k]
            for i in range(4):
                for k in range(4):
                    W[k * 4 + i][col] += (1.0/4.0) * T[r][i] * T[s][k]
            col += 1

    # Part 2: 4x4x1 rank-4 decomposition
    # For each column of the 4x5 matrix (the 5th column, index 4)
    # We need to compute A * B[:,4] where B[:,4] is a 4x1 vector
    # This can be done with rank 4 using the structure
    for j in range(4):
        # U part: for each i in 0..3, U[i*4 + j] = 1
        for i in range(4):
            U[i * 4 + j][col] = 1
        # V part: V[j*5 + 4] = 1
        V[j * 5 + 4][col] = 1
        # W part: for each i in 0..3, W[4*4 + i] = 1 if i == j else 0
        W[4 * 4 + j][col] = 1
        col += 1

    # Verify
    ok = verify(U, V, W, seed=99, trials=8)
    if ok:
        if elapsed() < DEADLINE:
            Ua = np.array(U, dtype=float)
            Va = np.array(V, dtype=float)
            Wa = np.array(W, dtype=float)
            Ua, Va, Wa, new_r = reduce_rank(Ua, Va, Wa, max_iter=300)
            if new_r < rank:
                ok2 = verify(to_list(Ua), to_list(Va), to_list(Wa), seed=777, trials=10)
                if ok2:
                    return {"U": to_list(Ua), "V": to_list(Va), "W": to_list(Wa)}
        return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook rank 80
    rank = 80
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(5):
                U[i * 4 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
