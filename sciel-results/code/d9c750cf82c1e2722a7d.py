"""Product-formula design for a six-site open XXZ Heisenberg chain.

Strategy: the 12-stage product formula U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)
has a leading effective Hamiltonian H_eff that is linear in the coefficients.
We (1) compute H_eff via the matrix logarithm of the product at t=1,
(2) solve a least-squares problem to match A+B, and (3) polish with a short
Nelder-Mead on the true multi-time spectral-norm error.
"""
import time
import numpy as np
from scipy.linalg import expm, logm


class _Timeout(Exception):
    pass


# Deadline guard: evaluator kills at 150 s wall-clock; keep a safety margin.
_T0 = time.time()
_DEADLINE = _T0 + 128.0  # 22 s safety margin

_N = 6
_DIM = 2 ** _N
_LAM = 0.8
_H = 0.3
_TIMES = (0.25, 0.5, 1.0)
_STAGES = 12

_B_global = None


def _site_op(op, site):
    ops = [np.eye(2, dtype=complex)] * _N
    ops[site] = op
    out = ops[0]
    for i in range(1, _N):
        out = np.kron(out, ops[i])
    return out


def _build_hamiltonians():
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    half_h = 0.5 * _H
    A = np.zeros((_DIM, _DIM), dtype=complex)
    B = np.zeros((_DIM, _DIM), dtype=complex)
    for i in range(_N):
        A += half_h * _site_op(sz, i)
        B += half_h * _site_op(sz, i)
    for i in range(_N - 1):
        bond = (_LAM * (
            _site_op(sx, i) @ _site_op(sx, i + 1)
            + _site_op(sy, i) @ _site_op(sy, i + 1)
        ) + _site_op(sz, i) @ _site_op(sz, i + 1))
        if i % 2 == 0:
            A += bond
        else:
            B += bond
    return A, B


def _product_unitary(a, b, t, A, B):
    U = np.eye(_DIM, dtype=complex)
    for k in range(_STAGES):
        if b[k] != 0.0:
            U = expm(-1j * b[k] * t * B) @ U
        if a[k] != 0.0:
            U = expm(-1j * a[k] * t * A) @ U
    return U


def _error(a, b, A, B):
    total = 0.0
    for t in _TIMES:
        U = _product_unitary(a, b, t, A, B)
        exact = expm(-1j * t * (A + B))
        err = np.linalg.norm(U - exact, 2)
        if np.isnan(err):
            return 1e6
        total += err
    return total / len(_TIMES)


def _normalize(a, b):
    sa = sum(a)
    sb = sum(b)
    if abs(sa) > 1e-14:
        a = [x / sa for x in a]
    if abs(sb) > 1e-14:
        b = [x / sb for x in b]
    return a, b


def _clip(a, b):
    a = [min(50.0, max(-50.0, float(x))) for x in a]
    b = [min(50.0, max(-50.0, float(x))) for x in b]
    return a, b


def _build_design_matrix(b, B, BA):
    D2 = _DIM * _DIM
    M = np.zeros((2 * D2, _STAGES), dtype=float)
    for k in range(_STAGES):
        M_k = b[k] * B
        comm = np.zeros((_DIM, _DIM), dtype=complex)
        for j in range(k):
            comm += b[j] * BA
        M_k = M_k + comm
        vec = M_k.reshape(-1)
        M[:D2, k] = vec.real
        M[D2:2 * D2, k] = vec.imag
    return M


def _rhs(target):
    D2 = _DIM * _DIM
    r = np.zeros(2 * D2, dtype=float)
    tv = target.reshape(-1)
    r[:D2] = tv.real
    r[D2:2 * D2] = tv.imag
    return r


def coefficients(stages):
    stages = 12  # contract fixes twelve stages

    if time.time() > _DEADLINE - 5.0:
        return [1.0 / 12] * 12, [1.0 / 12] * 12

    A, B = _build_hamiltonians()
    target = A + B
    BA = B @ A - A @ B

    def solve_for_a(b):
        M = _build_design_matrix(b, B, BA)
        r = _rhs(target)
        sol, *_ = np.linalg.lstsq(M, r, rcond=None)
        a = sol.tolist()
        a, b = _normalize(a, b)
        a, b = _clip(a, b)
        return a, b

    best_a = [1.0 / 12] * 12
    best_b = [1.0 / 12] * 12
    best_err = _error(best_a, best_b, A, B)

    # Try several b profiles
    b_profiles = [[1.0 / 12] * 12]
    for w in [0.3, 0.5, 0.7, 1.0]:
        bp = [max(0.05, 1.0 - w * abs(k - 5.5) / 5.5) for k in range(12)]
        s = sum(bp)
        b_profiles.append([x / s for x in bp])
    bp = [1.0 - 0.5 * (k / 11.0) for k in range(12)]
    b_profiles.append([x / sum(bp) for x in bp])
    bp = [0.5 + 0.5 * (k / 11.0) for k in range(12)]
    b_profiles.append([x / sum(bp) for x in bp])
    bp = [0.3 + 0.7 * (1.0 - abs(k - 5.5) / 5.5) for k in range(12)]
    b_profiles.append([x / sum(bp) for x in bp])

    for bp in b_profiles:
        if time.time() > _DEADLINE - 20.0:
            break
        a, bb = solve_for_a(bp)
        err = _error(a, bb, A, B)
        if err < best_err:
            best_err = err
            best_a, best_b = a, bb

    # Polish with Nelder-Mead on the true error, with a hard time stop.
    if time.time() < _DEADLINE - 30.0:
        from scipy.optimize import minimize

        x0 = np.array(best_a + best_b, dtype=float)
        _stop = {"flag": False}

        def obj(x):
            if _stop["flag"] or time.time() > _DEADLINE - 12.0:
                _stop["flag"] = True
                raise _Timeout
            a = list(x[:12])
            b = list(x[12:24])
            a, b = _normalize(a, b)
            a, b = _clip(a, b)
            return _error(a, b, A, B)

        try:
            res = minimize(
                obj, x0, method="Nelder-Mead",
                options={"maxiter": 300, "xatol": 1e-7, "fatol": 1e-9, "adaptive": True},
            )
            if res.x is not None and not _stop["flag"]:
                a = list(res.x[:12])
                b = list(res.x[12:24])
                a, b = _normalize(a, b)
                a, b = _clip(a, b)
                f = _error(a, b, A, B)
                if f < best_err:
                    best_err = f
                    best_a, best_b = a, b
        except _Timeout:
            pass
        except Exception:
            pass

    return best_a, best_b
