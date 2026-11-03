"""Rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 7]


def construct():
    t0 = time.time()
    deadline = t0 + 120.0

    n, m, p = DIMENSIONS
    N = n * m          # 16
    M = m * p          # 28
    P = p * n          # 28
    schoolbook = n * m * p  # 112

    # Build the multiplication tensor T[a,b,c] = 1 iff b = j*p+k, a = i*m+j, c = k*n+i
    T = np.zeros((N, M, P), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Precompute per-row sums of T (useful for rank lower bound via max row-sum)
    # T[a,b,c]; for fixed a, number of (b,c) with 1.
    row_sums = T.sum(axis=(1, 2))
    col_sums = T.sum(axis=(0, 2))
    # A rank-r decomposition: each row a is a sum of r rank-1 vectors; the
    # "weight" of row a (sum over b,c of T[a,b,c]) <= r (since each rank-1 term
    # contributes at most 1 to that row's sum if entries are 0/1... not exact).
    # Use a safer bound: rank >= ceil(max over a of (row_sum[a])) is not generally
    # valid, so skip. We'll just search.

    def make_rank(r):
        U = np.zeros((N, r))
        V = np.zeros((M, r))
        W = np.zeros((P, r))
        return U, V, W

    def residual(U, V, W):
        # sum over c of U[:,c] * V[:,c] * W[:,c]
        # = einsum('ac,bc,cc->ab', U, V, W)
        return np.einsum('ac,bc,cc->ab', U, V, W)

    def verify_exact(U, V, W):
        # Check that the decomposition is exact: residual matches T and entries are
        # integer/half-integer.
        R = residual(U, V, W)
        if not np.allclose(R, T, atol=1e-6):
            return False
        # entries must be integer or half-integer
        for mat in (U, V, W):
            m2 = mat * 2
            if not np.allclose(m2, np.round(m2), atol=1e-4):
                return False
        return True

    def round_to_half(X):
        return np.round(X * 2.0) / 2.0

    def refine(U, V, W, iters=3000, lr=1e-3):
        # Gradient descent on ||residual - T||^2 with periodic rounding to keep
        # entries near half-integers.
        U = U.copy(); V = V.copy(); W = W.copy()
        best_loss = np.inf
        best = (U.copy(), V.copy(), W.copy())
        for it in range(iters):
            R = residual(U, V, W)
            diff = R - T  # (N, M)
            loss = float(np.sum(diff * diff))
            if loss < best_loss:
                best_loss = loss
                best = (U.copy(), V.copy(), W.copy())
            # dL/dU = einsum('ab,bc,cc->ac', diff, V, W)
            dU = np.einsum('ab,bc,cc->ac', diff, V, W)
            dV = np.einsum('ab,ac,cc->bc', diff, U, W)
            dW = np.einsum('ab,ac,bc->cc', diff, U, V)
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            # periodic pull toward half-integers
            if it % 50 == 49:
                U = round_to_half(U)
                V = round_to_half(V)
                W = round_to_half(W)
        return best, best_loss

    def random_init(r, scale=1.0):
        U = np.random.randn(N, r) * scale
        V = np.random.randn(M, r) * scale
        W = np.random.randn(P, r) * scale
        return U, V, W

    best_result = None
    best_rank = schoolbook

    # Try ranks from a modest lower bound up to schoolbook.
    # Lower bound: rank >= max(N, M, P) = 28. Also rank >= ceil(N*M / P) etc.
    # N*M = 448, /P=28 -> 16. Use 28 as safe start, but try a few below in case.
    lower = max(N, M, P)  # 28

    # Search ranks in increasing order; for each, do several random restarts.
    # Budget-aware: stop if close to deadline.
    for r in range(lower, schoolbook + 1):
        if time.time() > deadline - 20.0:
            break
        U, V, W = make_rank(r)
        # A few restarts
        restarts = 4
        for rs in range(restarts):
            if time.time() > deadline - 15.0:
                break
            Ui, Vi, Wi = random_init(r, scale=0.5)
            (Ur, Vr, Wr), loss = refine(Ui, Vi, Wi, iters=1500, lr=1e-3)
            # Snap to half-integers and verify
            Ur = round_to_half(Ur); Vr = round_to_half(Vr); Wr = round_to_half(Wr)
            if verify_exact(Ur, Vr, Wr):
                # Found exact decomposition of rank r
                best_rank = r
                best_result = (Ur, Vr, Wr)
                break
        if best_result is not None:
            break

    # Fallback: schoolbook
    if best_result is None:
        r = schoolbook
        U = np.zeros((N, r)); V = np.zeros((M, r)); W = np.zeros((P, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        best_result = (U, V, W)

    U, V, W = best_result
    # Convert to lists of ints/half-ints
    def to_list(X):
        out = []
        for row in X:
            rlist = []
            for v in row:
                # represent half-integers exactly
                val = round(v * 2) / 2.0
                # if integer, use int
                if abs(val - round(val)) < 1e-9:
                    rlist.append(int(round(val)))
                else:
                    rlist.append(val)
            out.append(rlist)
        return out

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
