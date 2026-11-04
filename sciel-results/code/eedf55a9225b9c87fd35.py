"""Program for matrix multiplication tensor decomposition 3x4x7."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]

def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x4x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 3, 4, 7
    nmp = n * m * p  # 84 (schoolbook rank)

    def make_schoolbook():
        rank = nmp
        U = np.zeros((n * m, rank), dtype=np.int64)
        V = np.zeros((m * p, rank), dtype=np.int64)
        W = np.zeros((p * n, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def verify_fast2(U, V, W):
        r = U.shape[1]
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
        return np.all(T == 1)

    best_rank = nmp
    best_U, best_V, best_W = make_schoolbook()

    random.seed(12345)
    np.random.seed(12345)

    assert verify_fast2(best_U, best_V, best_W), "Schoolbook should verify"

    def find_W_given_UV(U, V, target_rank):
        W = np.zeros((p * n, target_rank), dtype=np.int64)
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, target_rank), dtype=np.float64)
                for j in range(m):
                    A[j, :] = U[i * m + j, :].astype(float) * V[j * p + k, :].astype(float)
                sol, residuals, rank, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                if len(residuals) > 0 and residuals[0] > 1e-8:
                    return None
                sol_int = np.round(sol).astype(np.int64)
                if np.allclose(A @ sol_int, np.ones(m), atol=1e-8):
                    W[k * n + i, :] = sol_int
                else:
                    found = False
                    for d0 in range(-2, 3):
                        for d1 in range(-2, 3):
                            for d2 in range(-2, 3):
                                for d3 in range(-2, 3):
                                    test = sol_int.copy()
                                    if target_rank > 0: test[0] += d0
                                    if target_rank > 1: test[1] += d1
                                    if target_rank > 2: test[2] += d2
                                    if target_rank > 3: test[3] += d3
                                    if np.allclose(A @ test, np.ones(m), atol=1e-8):
                                        W[k * n + i, :] = test
                                        found = True
                                        break
                                if found: break
                            if found: break
                        if found: break
                    if not found:
                        return None
        return W

    for target_rank in range(28, nmp):
        if time.time() > deadline - 20:
            break
        max_attempts = 200 if target_rank < 50 else 100
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            U = np.random.randint(-1, 2, size=(n * m, target_rank))
            V = np.random.randint(-1, 2, size=(m * p, target_rank))
            W = find_W_given_UV(U, V, target_rank)
            if W is not None:
                if verify_fast2(U, V, W):
                    best_rank = target_rank
                    best_U, best_V, best_W = U, V, W
                    break
        if best_rank < nmp:
            break

    if best_rank == nmp and time.time() < deadline - 30:
        for target_rank in range(28, 50):
            if time.time() > deadline - 15:
                break
            for attempt in range(100):
                if time.time() > deadline - 10:
                    break
                U = np.random.randint(-2, 3, size=(n * m, target_rank))
                V = np.random.randint(-2, 3, size=(m * p, target_rank))
                W = find_W_given_UV(U, V, target_rank)
                if W is not None:
                    if verify_fast2(U, V, W):
                        best_rank = target_rank
                        best_U, best_V, best_W = U, V, W
                        break
            if best_rank < nmp:
                break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
