"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time

    _deadline = time.time() + 135
    n, m, p = DIMENSIONS  # 3, 4, 6

    U66 = [[0.0] * 66 for _ in range(n * m)]
    V66 = [[0.0] * 66 for _ in range(m * p)]
    W66 = [[0.0] * 66 for _ in range(p * n)]

    col = 0

    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = a_r * m + a_c
        A01 = a_r * m + a_c + 1
        A10 = (a_r + 1) * m + a_c
        A11 = (a_r + 1) * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        C00 = c_c * n + c_r
        C01 = (c_c + 1) * n + c_r
        C10 = c_c * n + (c_r + 1)
        C11 = (c_c + 1) * n + (c_r + 1)

        U66[A00][col] += 1; U66[A11][col] += 1
        V66[B00][col] += 1; V66[B11][col] += 1
        W66[C00][col] += 1; W66[C11][col] += 1
        col += 1

        U66[A10][col] += 1; U66[A11][col] += 1
        V66[B00][col] += 1
        W66[C10][col] += 1; W66[C11][col] -= 1
        col += 1

        U66[A00][col] += 1
        V66[B01][col] += 1; V66[B11][col] -= 1
        W66[C01][col] += 1; W66[C11][col] += 1
        col += 1

        U66[A11][col] += 1
        V66[B10][col] += 1; V66[B00][col] -= 1
        W66[C00][col] += 1; W66[C10][col] += 1
        col += 1

        U66[A00][col] += 1; U66[A01][col] += 1
        V66[B11][col] += 1
        W66[C00][col] -= 1; W66[C01][col] += 1
        col += 1

        U66[A10][col] += 1; U66[A00][col] -= 1
        V66[B00][col] += 1; V66[B01][col] += 1
        W66[C11][col] += 1
        col += 1

        U66[A01][col] += 1; U66[A11][col] -= 1
        V66[B10][col] += 1; V66[B11][col] += 1
        W66[C00][col] += 1
        col += 1

    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = a_r * m + a_c
        A1 = a_r * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        C0 = c_c * n + c_r
        C1 = (c_c + 1) * n + c_r

        U66[A0][col] += 1; V66[B00][col] += 1; W66[C0][col] += 1
        col += 1
        U66[A1][col] += 1; V66[B10][col] += 1; W66[C0][col] += 1
        col += 1
        U66[A0][col] += 1; V66[B01][col] += 1; W66[C1][col] += 1
        col += 1
        U66[A1][col] += 1; V66[B11][col] += 1; W66[C1][col] += 1
        col += 1

    for t0 in [0, 2, 4]:
        add_1x2_2x2(0, 0, 0, t0, 0, t0)
        add_1x2_2x2(0, 2, 2, t0, 0, t0)

    for t0 in [0, 2, 4]:
        add_strassen(1, 0, 0, t0, 1, t0)
        add_strassen(1, 2, 2, t0, 1, t0)

    U = np.array(U66, dtype=float)
    V = np.array(V66, dtype=float)
    W = np.array(W66, dtype=float)

    def verify(U_mat, V_mat, W_mat, r):
        u = U_mat.reshape(n, m, r)
        v = V_mat.reshape(m, p, r)
        w = W_mat.reshape(p, n, r)
        T_reconstructed = np.einsum('ijc,jkc,kic->ijk', u, v, w)
        target = np.ones((n, m, p))
        return np.allclose(T_reconstructed, target, atol=1e-9)

    if not verify(U, V, W, 66):
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school))
        V = np.zeros((m * p, rank_school))
        W = np.zeros((p * n, rank_school))
        col2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col2] = 1
                    V[j * p + k, col2] = 1
                    W[k * n + i, col2] = 1
                    col2 += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Phase 1: Nullspace-based column elimination
    M = np.zeros((n * m * p, 66))
    for l in range(66):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    M[i * m * p + j * p + k, l] = U[i * m + j, l] * V[j * p + k, l] * W[k * n + i, l]

    eliminated = set()
    try:
        U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
        tol = 1e-10
        null_dim = np.sum(s < tol)

        if null_dim > 0:
            nullspace = Vt[66 - null_dim:, :]

            for null_vec in nullspace:
                if time.time() > _deadline - 20:
                    break
                candidates = [i for i in range(66) if i not in eliminated]
                if not candidates:
                    break
                max_idx = max(candidates, key=lambda i: abs(null_vec[i]))
                if abs(null_vec[max_idx]) < 1e-10:
                    continue

                remaining = [i for i in range(66) if i != max_idx]
                U_test = U[:, remaining]
                V_test = V[:, remaining]
                W_test = W[:, remaining]

                if verify(U_test, V_test, W_test, len(remaining)):
                    eliminated.add(max_idx)
    except:
        pass

    if eliminated:
        remaining = [i for i in range(66) if i not in eliminated]
        U = U[:, remaining]
        V = V[:, remaining]
        W = W[:, remaining]

    current_rank = U.shape[1]

    # Phase 2: Iterative drop search
    if time.time() < _deadline - 10:
        improved = True
        while improved and time.time() < _deadline - 10:
            improved = False
            r = U.shape[1]
            for l in range(r):
                if time.time() > _deadline - 10:
                    break
                U_test = np.delete(U, l, axis=1)
                V_test = np.delete(V, l, axis=1)
                W_test = np.delete(W, l, axis=1)
                if verify(U_test, V_test, W_test, r - 1):
                    U, V, W = U_test, V_test, W_test
                    improved = True
                    break

    # Phase 3: Pairwise drop search - try to drop two columns at once
    if time.time() < _deadline - 5:
        r = U.shape[1]
        for l1 in range(r):
            if time.time() > _deadline - 5:
                break
            for l2 in range(l1 + 1, r):
                if time.time() > _deadline - 5:
                    break
                remaining = [i for i in range(r) if i != l1 and i != l2]
                U_test = U[:, remaining]
                V_test = V[:, remaining]
                W_test = W[:, remaining]
                if verify(U_test, V_test, W_test, r - 2):
                    U, V, W = U_test, V_test, W_test
                    break
            if time.time() > _deadline - 5:
                break
            if U.shape[1] < r:
                break

    # Phase 4: Triple drop search
    if time.time() < _deadline - 3:
        r = U.shape[1]
        for l1 in range(r):
            if time.time() > _deadline - 3:
                break
            for l2 in range(l1 + 1, r):
                if time.time() > _deadline - 3:
                    break
                for l3 in range(l2 + 1, r):
                    if time.time() > _deadline - 3:
                        break
                    remaining = [i for i in range(r) if i != l1 and i != l2 and i != l3]
                    U_test = U[:, remaining]
                    V_test = V[:, remaining]
                    W_test = W[:, remaining]
                    if verify(U_test, V_test, W_test, r - 3):
                        U, V, W = U_test, V_test, W_test
                        break
                if time.time() > _deadline - 3:
                    break
                if U.shape[1] < r:
                    break
            if time.time() > _deadline - 3:
                break
            if U.shape[1] < r:
                break

    # Final verification and fallback
    if not verify(U, V, W, U.shape[1]):
        # Fallback to the original 66
        U = np.array(U66, dtype=float)
        V = np.array(V66, dtype=float)
        W = np.array(W66, dtype=float)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
