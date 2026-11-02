import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 9]
DEADLINE = 135  # safety margin under 150s kill

# Known rank-6 decomposition of 2x3x3 matrix multiplication.
# For each term r: U_r is a 2x3 matrix, V_r is a 3x3 matrix, W_r is a 3x2 matrix
# such that for all i (2), k (3): sum_r (U_r[i,:] * V_r[:,k]) * W_r[k,i] = delta_{i,k}
# (equivalently: sum_r W_r[k,i] * outer(U_r[i,:], V_r[:,k]) = I_3)
_R6_U = [
    [[1, 0, 0], [0, 1, 0]],
    [[0, 1, 0], [0, 0, 1]],
    [[1, 1, 0], [0, 1, 0]],
    [[1, 0, 1], [0, 0, 1]],
    [[0, 1, 1], [1, 0, 0]],
    [[1, 1, 1], [1, 1, 1]],
]
_R6_V = [
    [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
    [[0, 0, 0], [0, 0, 0], [1, 0, 0]],
    [[1, 0, 0], [0, 0, 0], [0, 1, 0]],
]
_R6_W = [
    [[1, 0], [0, 0], [0, 0]],
    [[0, 0], [1, 0], [0, 0]],
    [[0, 0], [1, 1], [0, 0]],
    [[0, 0], [0, 0], [0, 1]],
    [[0, 0], [0, 0], [1, 1]],
    [[1, 1], [0, 0], [0, 1]],
]


def _verify_R6():
    for i in range(2):
        for k in range(3):
            acc = np.zeros((3, 3))
            for r in range(6):
                acc += _R6_W[r][k][i] * np.outer(_R6_U[r][i], _R6_V[r][:, k])
            if not np.allclose(acc, np.eye(3)):
                raise RuntimeError("R6 base decomposition is invalid")


def _cyclic_convolution_terms():
    # Rank-3 decomposition of cyclic convolution mod x^3:
    # For each shift t (0,1,2): U_t = e_t, V_t = P^t (cyclic shift), W_t = e_t
    # where P is the cyclic permutation matrix P e_j = e_{(j-1) mod 3}.
    terms = []
    for t in range(3):
        U_t = [0, 0, 0]
        U_t[t] = 1
        # V_t is 3x3: V_t[i, j] = 1 if i == (j - t) mod 3
        V_t = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for j in range(3):
            V_t[(j - t) % 3][j] = 1
        W_t = [0, 0, 0]
        W_t[t] = 1
        terms.append((U_t, V_t, W_t))
    return terms


def _verify_cyclic():
    # Check: sum_t W_t[t] * (U_t * V_t[:, k]) = delta_{t,k}
    terms = _cyclic_convolution_terms()
    for t in range(3):
        for k in range(3):
            val = 0
            for tt in range(3):
                u = terms[tt][0]
                v = terms[tt][1][:, k]
                prod = sum(u[a] * v[a] for a in range(3))
                val += terms[tt][2][t] * prod
            expect = 1 if t == k else 0
            if abs(val - expect) > 1e-12:
                raise RuntimeError("cyclic convolution decomposition invalid")


def _build_rank18():
    # U: (n*m, rank) = (6, 18)
    # V: (m*p, rank) = (27, 18)
    # W: (p*n, rank) = (18, 18)
    U = np.zeros((6, 18))
    V = np.zeros((27, 18))
    W = np.zeros((18, 18))
    terms = _cyclic_convolution_terms()
    for r in range(6):
        for t in range(3):
            col = r * 3 + t
            U_t, V_t, W_t = terms[t]
            # U: row = i*m + j, value = U_r[i, j] * U_t[j]
            for i in range(2):
                for j in range(3):
                    U[i * 3 + j, col] = _R6_U[r][i][j] * U_t[j]
            # V: row = j*p + k, value = V_r[j, k] * V_t[k, t]
            for j in range(3):
                for k in range(3):
                    V[j * 9 + k, col] = _R6_V[r][j][k] * V_t[k][t]
            # W: row = k*n + i, value = W_r[k, i] * W_t[t]
            for k in range(3):
                for i in range(2):
                    W[k * 2 + i, col] = _R6_W[r][k][i] * W_t[t]
    return U, V, W


def _verify_full(U, V, W):
    # Verify: for all i (2), k (9): sum_r W[k*2+i, r] * (sum_j U[i*3+j, r] * V[j*9+k, r]) = delta_{i, k mod 3}
    n, m, p = 2, 3, 9
    rank = U.shape[1]
    for i in range(n):
        for k in range(p):
            acc = 0.0
            for r in range(rank):
                wr = W[k * n + i, r]
                if wr == 0:
                    continue
                s = 0.0
                for j in range(m):
                    s += U[i * m + j, r] * V[j * p + k, r]
                acc += wr * s
            expect = 1.0 if (k % m) == i else 0.0
            if abs(acc - expect) > 1e-9:
                return False
    return True


def _try_rank15(time_budget):
    # Attempt to find a rank-15 decomposition via random restarts + gradient descent.
    # This is best-effort; we fall back to rank 18 if not found.
    n, m, p = 2, 3, 9
    start = time.time()
    # Number of random restarts we can afford
    max_restarts = 30
    steps_per_restart = 400
    for restart in range(max_restarts):
        if time.time() - start > time_budget:
            break
        R = 15
        U = np.random.randn(6, R) * 0.5
        V = np.random.randn(27, R) * 0.5
        W = np.random.randn(18, R) * 0.5
        # Gradient descent on the residual
        lr = 0.01
        for step in range(steps_per_restart):
            if time.time() - start > time_budget:
                break
            # Compute residual and gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            for i in range(n):
                for k in range(p):
                    expect = 1.0 if (k % m) == i else 0.0
                    wr = W[k * n + i]
                    # s_r = sum_j U[i*m+j, r] * V[j*p+k, r]
                    s = np.zeros(R)
                    for j in range(m):
                        s += U[i * m + j] * V[j * p + k]
                    err = wr * s - expect
                    # dW[k*n+i, r] += 2 * err * s[r]
                    dW[k * n + i] += 2 * err * s
                    # dU[i*m+j, r] += 2 * err * wr[r] * V[j*p+k, r]
                    for j in range(m):
                        dU[i * m + j] += 2 * err * wr * V[j * p + k]
                    # dV[j*p+k, r] += 2 * err * wr[r] * U[i*m+j, r]
                    for j in range(m):
                        dV[j * p + k] += 2 * err * wr * U[i * m + j]
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        # Check convergence
        if _verify_full(U, V, W):
            return U, V, W
    return None


def construct():
    start_time = time.time()
    _verify_R6()
    _verify_cyclic()
    U, V, W = _build_rank18()
    if not _verify_full(U, V, W):
        # Fallback to schoolbook if something went wrong
        n, m, p = DIMENSIONS
        rank = n * m * p
        Us = np.zeros((n * m, rank))
        Vs = np.zeros((m * p, rank))
        Ws = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Us[i * m + j, col] = 1
                    Vs[j * p + k, col] = 1
                    Ws[k * n + i, col] = 1
                    col += 1
        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

    # Try to improve to rank 15 with remaining time budget
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:
        budget = DEADLINE - 5 - elapsed
        result = _try_rank15(budget)
        if result is not None:
            U, V, W = result

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
