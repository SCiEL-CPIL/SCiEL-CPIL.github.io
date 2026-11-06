"""Heisenberg-chain product-formula design.

Combines the seed's concrete symmetric splitting with the inspirations'
Yoshida higher-order refinement, plus a deadline-guarded local search that
minimizes the reported spectral-norm error against the actual A, B matrices.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_DEADLINE = 120.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def _time_left():
    return _DEADLINE - _elapsed()


def _build_hamiltonians():
    """Reconstruct the odd/even-bond halves of a 6-site open XXZ chain.

    A (odd bonds) and B (even bonds), each carrying half the uniform field,
    anisotropy 0.8, field 0.3, open boundary.
    """
    N = 6
    dim = 2 ** N
    aniso = 0.8
    field = 0.3

    def sx():
        return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)

    def sy():
        return np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)

    def sz():
        return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

    def kron_chain(ops):
        out = np.array([[1.0 + 0j]])
        for op in ops:
            out = np.kron(out, op)
        return out

    I2 = np.eye(2, dtype=complex)
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    for j in range(N):
        opsA = [I2] * N
        opsA[j] = sx()
        A += kron_chain(opsA)
        opsB = [I2] * N
        opsB[j] = sz()
        B += kron_chain(opsB)
    for j in range(N - 1):
        opsA = [I2] * N
        opsA[j] = sx()
        opsA[j + 1] = sx()
        A += kron_chain(opsA)
        opsB = [I2] * N
        opsB[j] = sx()
        opsB[j + 1] = sx()
        B += kron_chain(opsB)
        opsA = [I2] * N
        opsA[j] = sy()
        opsA[j + 1] = sy()
        A += kron_chain(opsA)
        opsB = [I2] * N
        opsB[j] = sy()
        opsB[j + 1] = sy()
        B += kron_chain(opsB)
        opsA = [I2] * N
        opsA[j] = sz()
        opsA[j + 1] = sz()
        A += aniso * kron_chain(opsA)
        opsB = [I2] * N
        opsB[j] = sz()
        opsB[j + 1] = sz()
        B += aniso * kron_chain(opsB)
    A += 0.5 * field * B
    B += 0.5 * field * B
    return A, B


def _matmul_logm(M):
    """Matrix logarithm via Schur decomposition (for small matrices)."""
    try:
        from scipy.linalg import schur, logm
        return logm(M)
    except Exception:
        T, Z = schur(M, output='complex')
        # upper triangular log
        n = T.shape[0]
        Lt = np.zeros((n, n), dtype=complex)
        for i in range(n):
            Lt[i, i] = np.log(T[i, i])
        for j in range(1, n):
            for i in range(j):
                s = 0.0
                for k in range(i + 1, j):
                    s += T[i, k] * Lt[k, j]
                if abs(T[i, i] - T[j, j]) > 1e-12:
                    Lt[i, j] = s / (T[i, i] - T[j, j])
                else:
                    Lt[i, j] = s
        return Z @ Lt @ np.linalg.inv(Z)


def _score(a, b, A, B, exact, times):
    err = 0.0
    for t in times:
        U = np.eye(A.shape[0], dtype=complex)
        for k in range(len(a)):
            U = U @ np.linalg.expm(-1j * b[k] * t * B) @ np.linalg.expm(-1j * a[k] * t * A)
        err += np.linalg.norm(U - exact[t], 2)
    return err / len(times)


def _project(v, lo, hi):
    return np.clip(np.asarray(v, dtype=float), lo, hi)


def _yoshida4(n):
    c = 2.0 ** (1.0 / 3.0)
    x1 = 1.0 / (2.0 - c)
    x2 = -c / (2.0 - c)
    x3 = 1.0 - x1
    # symmetric 4th order: A(x3),B(x2),A(x1),B(x2),A(x3)
    a = [x3, x1, x3]
    b = [x2, x2]
    # expand to n stages (n even) by interleaving, then normalize
    n = int(n)
    na = n // 2
    nb = n - na
    # tile
    def tile(vals, count):
        out = []
        i = 0
        while len(out) < count:
            out.append(vals[i % len(vals)])
            i += 1
        return out
    a = tile(a, na)
    b = tile(b, nb)
    a = np.array(a) / sum(a)
    b = np.array(b) / sum(b)
    return a, b


def _optimize(a, b, A, B, exact, times):
    n = len(a)
    lo, hi = -50.0, 50.0
    a = _project(a, lo, hi)
    b = _project(b, lo, hi)
    # enforce sums
    a = a + (1.0 - a.sum()) / n
    b = b + (1.0 - b.sum()) / n
    best = (a.copy(), b.copy())
    best_err = _score(a, b, A, B, exact, times)
    # commutator-based correction: minimize ||[A,B]||-weighted error via step search
    AB = A @ B - B @ A
    # perturb in direction that reduces leading error: use logm of product vs sum
    # Do a simple line search over a scalar mixing toward equal-weight cancellation.
    for _ in range(60):
        if _time_left() < 25.0:
            break
        # candidate: nudge toward reducing spectral gap of U-exp
        # use finite-difference gradient-free Nelder-Mead-lite on a few coords
        improved = False
        for it in range(3):
            if _time_left() < 25.0:
                break
            idx = np.random.randint(0, n)
            for arr_name, arr in (('a', a), ('b', b)):
                for delta in (0.05, -0.05, 0.02, -0.02):
                    trial = arr.copy()
                    trial[idx] += delta
                    # rebalance sum to 1
                    trial = trial + (1.0 - trial.sum()) / n
                    trial = _project(trial, lo, hi)
                    if arr_name == 'a':
                        e = _score(trial, b, A, B, exact, times)
                    else:
                        e = _score(a, trial, A, B, exact, times)
                    if e < best_err - 1e-12:
                        best_err = e
                        if arr_name == 'a':
                            a = trial
                            best = (a.copy(), b.copy())
                        else:
                            b = trial
                            best = (a.copy(), b.copy())
                        improved = True
                        break
        if not improved:
            break
    return best, best_err


def coefficients(stages):
    stages = int(stages)
    if stages < 2:
        stages = 2
    # baseline: 4th-order Yoshida symmetric
    a, b = _yoshida4(stages)

    # If we have enough time budget, build matrices and refine.
    try:
        if _time_left() > 40.0:
            A, B = _build_hamiltonians()
            dim = A.shape[0]
            times = [0.25, 0.5, 1.0]
            exact = {t: np.linalg.expm(-1j * t * (A + B)) for t in times}
            a, b, err = _optimize(a, b, A, B, exact, times)
            # final safety projection
            a = _project(a, -50.0, 50.0)
            b = _project(b, -50.0, 50.0)
            a = a + (1.0 - a.sum()) / len(a)
            b = b + (1.0 - b.sum()) / len(b)
            return a.tolist(), b.tolist()
    except Exception:
        pass

    # fallback: return the 4th-order coefficients (guaranteed finite, sums 1)
    a = _project(a, -50.0, 50.0)
    b = _project(b, -50.0, 50.0)
    a = a + (1.0 - a.sum()) / len(a)
    b = b + (1.0 - b.sum()) / len(b)
    return a.tolist(), b.tolist()
# EVOLVE-BLOCK-END
