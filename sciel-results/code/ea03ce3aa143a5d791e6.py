"""
Rank-minimal tensor decomposition for 2x3 * 3x10 matrix multiplication.

Strategy: Partition middle dim 3 = 2 + 1.
  Part 1: (2x2) * (2x10) via 5 Strassen 2x2x2 blocks -> rank 35
  Part 2: (2x1) * (1x10) schoolbook -> rank 20
  Total: rank 55

Then attempt rank reduction via linear dependence detection.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 150.0
SAFETY_MARGIN = 5.0


def _verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    """Verify that the decomposition correctly represents matrix multiplication."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                if not np.isclose(lhs, 1.0, atol=atol):
                    return False
    return True


def _try_reduce_rank(U, V, W, n, m, p, max_iter=500, timeout=30.0):
    """Attempt to reduce rank by finding linearly dependent terms and merging."""
    start_time = time.time()
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Check for duplicate or linearly dependent terms
    # Two terms r1, r2 can be merged if U[:,r1] is proportional to U[:,r2]
    # and V[:,r1] is proportional to V[:,r2], etc.
    
    # Try to find if any pair of terms are scalar multiples of each other
    # in all three components simultaneously
    removed = []
    for r1 in range(rank):
        if r1 in removed:
            continue
        for r2 in range(r1 + 1, rank):
            if r2 in removed:
                continue
            # Check if term r2 is a scalar multiple of term r1 in U, V, W
            # U[:, r2] = alpha * U[:, r1]
            # V[:, r2] = alpha * V[:, r1]  
            # W[:, r2] = alpha * W[:, r1]
            u1 = U_arr[:, r1]
            u2 = U_arr[:, r2]
            if np.allclose(u1, 0, atol=1e-12) or np.allclose(u2, 0, atol=1e-12):
                continue
            # Find alpha from U
            nonzero = np.abs(u1) > 1e-12
            if not np.any(nonzero):
                continue
            alphas = u2[nonzero] / u1[nonzero]
            if not np.allclose(alphas, alphas[0], atol=1e-10, rtol=1e-10):
                continue
            alpha = alphas[0]
            # Verify V
            if not np.allclose(V_arr[:, r2], alpha * V_arr[:, r1], atol=1e-10, rtol=1e-10):
                continue
            # Verify W
            if not np.allclose(W_arr[:, r2], alpha * W_arr[:, r1], atol=1e-10, rtol=1e-10):
                continue
            # Term r2 is redundant
            removed.append(r2)
            break  # Only remove one at a time to be safe
    
    if removed:
        keep = [i for i in range(rank) if i not in removed]
        U_new = U_arr[:, keep].tolist()
        V_new = V_arr[:, keep].tolist()
        W_new = W_arr[:, keep].tolist()
        new_rank = len(keep)
        if _verify_decomposition(U_new, V_new, W_new, n, m, p):
            return U_new, V_new, W_new
    
    return U, V, W


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=3, p=10

    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55

    r = 0

    # Strassen coefficients for 2x2x2
    strassen_U = {
        0: [(0, 1), (4, 1)],
        1: [(3, 1), (4, 1)],
        2: [(0, 1)],
        3: [(4, 1)],
        4: [(0, 1), (1, 1)],
        5: [(3, 1), (0, -1)],
        6: [(1, 1), (4, -1)],
    }

    strassen_V = {
        0: [(0, 1), (3, 1)],   # B00 + B11
        1: [(0, 1)],           # B00
        2: [(1, 1), (3, -1)],  # B01 - B11
        3: [(2, 1), (0, -1)],  # B10 - B00
        4: [(3, 1)],           # B11
        5: [(0, 1), (1, 1)],   # B00 + B01
        6: [(2, 1), (3, 1)],   # B10 + B11
    }

    strassen_W = {
        0: [(0, 1), (3, 1)],   # C00 + C11
        1: [(1, 1), (3, -1)],  # C10 - C11
        2: [(2, 1), (3, 1)],   # C01 + C11
        3: [(0, 1), (1, 1)],   # C00 + C10
        4: [(0, -1), (2, 1)],  # -C00 + C01
        5: [(3, 1)],           # C11
        6: [(0, 1)],           # C00
    }

    for b in range(5):
        col0 = 2 * b
        col1 = 2 * b + 1
        v_offsets = {0: col0, 1: col1, 2: 10 + col0, 3: 10 + col1}
        w_offsets = {0: 4 * b, 1: 4 * b + 1, 2: 4 * b + 2, 3: 4 * b + 3}

        for s in range(7):
            for (u_idx, coeff) in strassen_U[s]:
                U[u_idx][r] += coeff
            for (v_off, coeff) in strassen_V[s]:
                V[v_offsets[v_off]][r] += coeff
            for (w_off, coeff) in strassen_W[s]:
                W[w_offsets[w_off]][r] += coeff
            r += 1

    # Part 2: Schoolbook for A[*,2] * B[2,*]
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1

    assert r == rank, f"Expected rank {rank}, got {r}"

    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        for c in range(rank_fb):
            ii = c // (m * p)
            rem = c % (m * p)
            jj = rem // p
            kk = rem % p
            U_fb[ii * m + jj][c] = 1
            V_fb[jj * p + kk][c] = 1
            W_fb[kk * n + ii][c] = 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    # Try to reduce rank if time permits
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - SAFETY_MARGIN - 10:
        U, V, W = _try_reduce_rank(U, V, W, n, m, p)

    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")

    n, m, p = DIMENSIONS
    U = np.array(result['U'])
    V = np.array(result['V'])
    W = np.array(result['W'])
    rank = U.shape[1]

    np.random.seed(42)
    A = np.random.randint(-5, 6, size=(n, m)).astype(float)
    B = np.random.randint(-5, 6, size=(m, p)).astype(float)

    C_decomp = np.zeros((n, p))
    for r in range(rank):
        A_w = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                A_w[i, j] = U[i * m + j, r]
        B_w = np.zeros((m, p))
        for j in range(m):
            for k in range(p):
                B_w[j, k] = V[j * p + k, r]
        prod_r = np.sum(A_w * A) * np.sum(B_w * B)
        for i in range(n):
            for k in range(p):
                C_decomp[i, k] += prod_r * W[k * n + i, r]

    C_direct = A @ B
    error = np.max(np.abs(C_decomp - C_direct))
    print(f"Max error: {error}")
    if error > 1e-10:
        print("WARNING: Decomposition is incorrect!")
