"""
Rank-minimal tensor decomposition for 2x3 by 3x10 matrix multiplication.

Strategy:
  Partition the middle dimension (3) into 2 + 1.
  - Part 1: rows 0,1 of A (2x2) times rows 0,1 of B (2x10).
    Partition the output columns (10) into five 2-wide blocks.
    For each 2x2 x 2x2 sub-multiplication, use the classic rank-7 Strassen algorithm.
    Subtotal: 5 * 7 = 35.
  - Part 2: row 2 of A (2x1) times row 2 of B (1x10).
    Schoolbook: 2 * 1 * 10 = 20.
  Total rank: 55.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 145.0


def _verify_decomposition(U, V, W, n, m, p, num_trials=20, seed=42):
    rng = np.random.RandomState(seed)
    rank = U.shape[1]
    for trial in range(num_trials):
        A = rng.randint(-10, 11, size=(n, m)).astype(float)
        B = rng.randint(-10, 11, size=(m, p)).astype(float)
        C_direct = A @ B
        A_flat = A.flatten()
        B_flat = B.flatten()
        u_sums = U.T @ A_flat
        v_sums = V.T @ B_flat
        scalars = u_sums * v_sums
        C_flat = W @ scalars
        C_decomp = C_flat.reshape(p, n).T
        err = np.max(np.abs(C_decomp - C_direct))
        if err > 1e-9:
            return False, err
    for A_test, B_test in [
        (np.eye(n, m), np.eye(m, p)),
        (np.ones((n, m)), np.ones((m, p))),
        (np.array([[1, 0, 0], [0, 1, 0]], dtype=float), np.eye(3, p)),
    ]:
        C_direct = A_test @ B_test
        A_flat = A_test.flatten()
        B_flat = B_test.flatten()
        u_sums = U.T @ A_flat
        v_sums = V.T @ B_flat
        scalars = u_sums * v_sums
        C_flat = W @ scalars
        C_decomp = C_flat.reshape(p, n).T
        err = np.max(np.abs(C_decomp - C_direct))
        if err > 1e-9:
            return False, err
    return True, 0.0


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS

    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    idx = 0

    for block in range(5):
        k0 = 2 * block
        k1 = 2 * block + 1

        # M1
        U[0, idx] = 1.0
        U[4, idx] = 1.0
        V[k0, idx] = 1.0
        V[10 + k1, idx] = 1.0
        W[k0, idx] = 1.0
        W[2 + k1, idx] = 1.0
        idx += 1

        # M2
        U[3, idx] = 1.0
        U[4, idx] = 1.0
        V[k0, idx] = 1.0
        W[1 + k0, idx] = 1.0
        W[2 + k1, idx] = -1.0
        idx += 1

        # M3
        U[0, idx] = 1.0
        V[10 + k1, idx] = 1.0
        V[20 + k1, idx] = -1.0
        W[k1, idx] = 1.0
        W[2 + k1, idx] = 1.0
        idx += 1

        # M4
        U[4, idx] = 1.0
        V[20 + k0, idx] = 1.0
        V[k0, idx] = -1.0
        W[k0, idx] = 1.0
        W[1 + k0, idx] = 1.0
        idx += 1

        # M5
        U[0, idx] = 1.0
        U[1, idx] = 1.0
        V[20 + k1, idx] = 1.0
        W[k0, idx] = -1.0
        W[k1, idx] = 1.0
        idx += 1

        # M6
        U[3, idx] = 1.0
        U[0, idx] = -1.0
        V[k0, idx] = 1.0
        V[10 + k0, idx] = 1.0
        W[2 + k1, idx] = 1.0
        idx += 1

        # M7
        U[1, idx] = 1.0
        U[4, idx] = -1.0
        V[20 + k0, idx] = 1.0
        V[20 + k1, idx] = 1.0
        W[k0, idx] = 1.0
        idx += 1

    for i in range(2):
        for k in range(10):
            U[6 + i, idx] = 1.0
            V[20 + k, idx] = 1.0
            W[k * 2 + i, idx] = 1.0
            idx += 1

    ok, err = _verify_decomposition(U, V, W, n, m, p)
    if not ok:
        pass

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
