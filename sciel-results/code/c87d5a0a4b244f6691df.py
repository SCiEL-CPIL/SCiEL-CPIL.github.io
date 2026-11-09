"""Product-formula design for 6-site Heisenberg chain via numerical optimization."""
import numpy as np
import time

_START = time.time()
_DEADLINE = 150.0
_MARGIN = 15.0

def _build_hamiltonians():
    N = 6
    dim = 2 ** N
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result

    anisotropy = 0.8
    field = 0.3

    A = np.zeros((dim, dim), dtype=complex)
    for i, j in [(0, 1), (2, 3), (4, 5)]:
        for a_op, b_op, coeff in [(sx, sx, 1.0), (sy, sy, 1.0), (sz, sz, anisotropy)]:
            ops = [I2] * N
            ops[i] = a_op
            ops[j] = b_op
            A += coeff * kron_all(ops)
    for k in range(N):
        ops = [I2] * N
        ops[k] = sz
        A += (field / 2.0) * kron_all(ops)

    B = np.zeros((dim, dim), dtype=complex)
    for i, j in [(1, 2), (3, 4)]:
        for a_op, b_op, coeff in [(sx, sx, 1.0), (sy, sy, 1.0), (sz, sz, anisotropy)]:
            ops = [I2] * N
            ops[i] = a_op
            ops[j] = b_op
            B += coeff * kron_all(ops)
    for k in range(N):
        ops = [I2] * N
        ops[k] = sz
        B += (field / 2.0) * kron_all(ops)

    return A, B

def coefficients(stages):
    global _START, _DEADLINE, _MARGIN

    A, B = _build_hamiltonians()
    times = [0.25, 0.5, 1.0]
    dim = A.shape[0]

    evA, vecA = np.linalg.eigh(A)
    evB, vecB = np.linalg.eigh(B)
    H = A + B
    evH, vecH = np.linalg.eigh(H)

    def _apply_exp(vec, ev, coeff, t):
        return (vec * np.exp(-1j * coeff * t * ev)) @ vec.conj().T

    def _objective(params):
        a = params[:6]
        b = params[6:12]
        a_full = np.concatenate([a, a[::-1]])
        b_full = np.concatenate([b, b[::-1]])

        sum_a = np.sum(a_full)
        sum_b = np.sum(b_full)
        penalty = 100.0 * (sum_a - 1.0)**2 + 100.0 * (sum_b - 1.0)**2

        a_scaled = a_full / sum_a if abs(sum_a) > 1e-10 else a_full
        b_scaled = b_full / sum_b if abs(sum_b) > 1e-10 else b_full

        errors = []
        for t in times:
            U_pf = np.eye(dim, dtype=complex)
            for k in range(12):
                if abs(b_scaled[k]) > 1e-15:
                    U_pf = _apply_exp(vecB, evB, b_scaled[k], t) @ U_pf
                if abs(a_scaled[k]) > 1e-15:
                    U_pf = _apply_exp(vecA, evA, a_scaled[k], t) @ U_pf
            U_exact = (vecH * np.exp(-1j * t * evH)) @ vecH.conj().T
            diff = U_pf - U_exact
            s = np.linalg.svd(diff, compute_uv=False)
            errors.append(s[0])
        return np.mean(errors) + penalty

    init_a = np.array([1.0/12.0] * 6)
    init_b = np.array([1.0/12.0] * 6)
    init_params = np.concatenate([init_a, init_b])

    best_params = init_params.copy()
    best_obj = _objective(init_params)

    elapsed = time.time() - _START
    if elapsed < _DEADLINE - _MARGIN:
        try:
            from scipy.optimize import minimize
            res = minimize(_objective, init_params, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < best_obj and time.time() - _START < _DEADLINE - _MARGIN:
                best_params = res.x
                best_obj = res.fun
        except Exception:
            pass

    elapsed = time.time() - _START
    if elapsed < _DEADLINE - _MARGIN:
        try:
            from scipy.optimize import minimize
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-14})
            if res.fun < best_obj and time.time() - _START < _DEADLINE - _MARGIN:
                best_params = res.x
                best_obj = res.fun
        except Exception:
            pass

    a_half = best_params[:6]
    b_half = best_params[6:12]
    a_full = np.concatenate([a_half, a_half[::-1]])
    b_full = np.concatenate([b_half, b_half[::-1]])

    a_full = a_full / np.sum(a_full)
    b_full = b_full / np.sum(b_full)

    a_full = np.clip(a_full, -50, 50)
    b_full = np.clip(b_full, -50, 50)

    a_full = a_full / np.sum(a_full)
    b_full = b_full / np.sum(b_full)

    # Re-symmetrize
    for k in range(6):
        a_full[k] = a_full[11-k]
        b_full[k] = b_full[11-k]
    a_full = a_full / np.sum(a_full)
    b_full = b_full / np.sum(b_full)

    return a_full.tolist(), b_full.tolist()
