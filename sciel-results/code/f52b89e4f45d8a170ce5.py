"""Analytic uncertainty bound via precomputed 3x3 quadratic forms + multi-start L-BFGS-B + root-finding."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, fsolve

_DEADLINE = time.time() + 200.0

# ---------------------------------------------------------------------------
# Precomputation: I[n] = integral x^{2n} exp(-x^2) dx
# ---------------------------------------------------------------------------
def _precompute_I(N):
    I = np.zeros(N)
    I[0] = np.sqrt(np.pi)
    for n in range(1, N):
        I[n] = I[n - 1] * (2.0 * n - 1.0) / (2.0 * n)
    return I

# We need I[0] .. I[13] (max index i+j+1 with i,j in 0..6 -> 13)
_I = _precompute_I(14)

# ---------------------------------------------------------------------------
# Hermite power-basis coefficients (even degrees only): H_{2k} in x^{2j}
# H0  = 1
# H4  = x^4 - 6 x^2 + 3
# H8  = x^8 - 28 x^6 + 210 x^4 - 420 x^2 + 105
# H12 = x^12 - 66 x^10 + 1155 x^8 - 6930 x^6 + 13860 x^4 - 7560 x^2 + 945
# ---------------------------------------------------------------------------
H0_c  = np.array([1.0, 0, 0, 0, 0, 0, 0])
H4_c  = np.array([3.0, -6, 1, 0, 0, 0, 0])
H8_c  = np.array([105.0, -420, 210, -28, 1, 0, 0])
H12_c = np.array([945.0, -7560, 13860, -6930, 1155, -66, 1])

# a12 = -(a0 + 3 a4 + 105 a8) / 945   (enforces g(0)=0)
c12 = np.array([-1.0 / 945.0, -3.0 / 945.0, -105.0 / 945.0])

col0 = H0_c + c12[0] * H12_c
col1 = H4_c + c12[1] * H12_c
col2 = H8_c + c12[2] * H12_c
A_mat = np.column_stack((col0, col1, col2))   # shape (7, 3)

sign_vec = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

# ---------------------------------------------------------------------------
# Build 7x7 moment matrices
#   M0[i,j] = I[i+j]
#   M1[i,j] = I[i+j+1]
# ---------------------------------------------------------------------------
_idx0 = np.add.outer(np.arange(7), np.arange(7))
_idx1 = _idx0 + 1
M0 = _I[_idx0]
M1 = _I[_idx1]

D = np.diag(sign_vec)

# 3x3 quadratic-form matrices
B0 = A_mat.T @ M0 @ A_mat
B1 = A_mat.T @ M1 @ A_mat
C0 = A_mat.T @ (D @ M0 @ D) @ A_mat
C1 = A_mat.T @ (D @ M1 @ D) @ A_mat

_BIG = 1e18


def _bound_and_grad(a):
    """Return (bound, grad) for the uncertainty product sqrt(Vx*Vxi)."""
    a = np.asarray(a, dtype=float)
    b0v = float(a @ B0 @ a)
    b1v = float(a @ B1 @ a)
    c0v = float(a @ C0 @ a)
    c1v = float(a @ C1 @ a)

    if (not np.isfinite(b0v)) or (not np.isfinite(c0v)) or \
       (not np.isfinite(b1v)) or (not np.isfinite(c1v)) or \
       b0v <= 1e-14 or c0v <= 1e-14:
        return _BIG, np.array([_BIG, _BIG, _BIG])

    vx = b1v / b0v
    vxi = c1v / c0v

    if vx <= 1e-14 or vxi <= 1e-14 or not np.isfinite(vx) or not np.isfinite(vxi):
        return _BIG, np.array([_BIG, _BIG, _BIG])

    bound = np.sqrt(vx * vxi)
    if not np.isfinite(bound):
        return _BIG, np.array([_BIG, _BIG, _BIG])

    # Gradients of the four quadratic forms
    B0a = B0 @ a
    B1a = B1 @ a
    C0a = C0 @ a
    C1a = C1 @ a

    # dvx = (2 (B1 a) b0v - 2 (B0 a) b1v) / b0v^2
    dvx = (2.0 * B1a * b0v - 2.0 * B0a * b1v) / (b0v * b0v)
    dvxi = (2.0 * C1a * c0v - 2.0 * C0a * c1v) / (c0v * c0v)

    grad = (vxi * dvx + vx * dvxi) / (2.0 * bound)
    if not np.all(np.isfinite(grad)):
        return _BIG, np.array([_BIG, _BIG, _BIG])
    return bound, grad


def _objective(a):
    v, _ = _bound_and_grad(a)
    return v


def _objective_grad(a):
    return _bound_and_grad(a)


def _grad_only(a):
    _, g = _bound_and_grad(a)
    return g


def construct():
    # Curated diverse starting points + random spread
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([-1.0, 0.05, 0.0005]),
        np.array([-1.5, 0.06, 0.0006]),
        np.array([-2.0, 0.08, 0.001]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-2.5, 0.10, 0.0012]),
        np.array([-3.0, 0.12, 0.0015]),
        np.array([-0.8, 0.03, 0.0003]),
        np.array([-1.8, 0.07, 0.0008]),
        np.array([-1.3, 0.045, 0.0004]),
        np.array([-1.2, 0.042, 0.00035]),
        np.array([-1.15, 0.038, 0.00032]),
        np.array([-1.1, 0.045, 0.00028]),
        np.array([-1.4, 0.05, 0.0005]),
        np.array([-0.9, 0.035, 0.0004]),
    ]
    rng = np.random.default_rng(42)
    for i in range(20):
        starts.append(rng.normal([-1.5, 0.05, 0.0005], [0.5, 0.03, 0.0003], 3))

    best_val = np.inf
    best_params = None
    candidates = []

    for start in starts:
        if time.time() > _DEADLINE - 40:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                           jac=True,
                           options={'maxiter': 4000, 'ftol': 1e-18,
                                    'gtol': 1e-16, 'maxls': 30})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
            if np.isfinite(res.fun):
                candidates.append((res.fun, res.x.copy()))
        except Exception:
            pass

    # Root-finding polish on the best few candidates
    candidates.sort(key=lambda x: x[0])
    for val, params in candidates[:10]:
        if time.time() > _DEADLINE - 15:
            break
        try:
            sol, info, ier, msg = fsolve(_grad_only, params, full_output=True)
            if ier == 1 and np.all(np.isfinite(sol)):
                sol_val, _ = _bound_and_grad(sol)
                if np.isfinite(sol_val) and sol_val < best_val:
                    best_val = sol_val
                    best_params = sol.copy()
        except Exception:
            pass

    # Final high-precision polish
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                           jac=True,
                           options={'maxiter': 20000, 'ftol': 1e-20,
                                    'gtol': 1e-18, 'maxls': 50})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None or not np.all(np.isfinite(best_params)):
        return np.array([-1.13, 0.04, 0.000308642])

    return np.array(best_params)
# EVOLVE-BLOCK-END
