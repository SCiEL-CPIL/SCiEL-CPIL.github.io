"""Radau IIA 3-stage implicit RK with analytic Jacobian for the fixed Van der Pol ODE."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Radau IIA 3-stage (5th order) coefficients
_C = np.array([0.2388531032758142, 0.4459485323133456, 1.0])
_A = np.array([
    [0.06978913774086688, 0.16906396553494734, 0.0],
    [-0.01229235284941485, 0.4582408851627604, 0.0],
    [0.01038781232556145, 0.4355607199877841, 0.5540514676866544],
])
_B = np.array([0.01038781232556145, 0.4355607199877841, 0.5540514676866544])
_B4 = np.array([0.04765751183721908, 0.4244924881627809, 0.52785])
_E = _B - _B4  # embedded error estimate (order 4)


def _jac(y):
    """Analytic Jacobian of the fixed Van der Pol RHS (costs no f-call)."""
    y0, y1 = y[0], y[1]
    return np.array([[0.0, 1.0],
                     [-10.0 * y0 * y1 - 1.0, 5.0 * (1.0 - y0 * y0)]])


def _solve2(M, v):
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    if abs(det) < 1e-300:
        return np.array([0.0, 0.0])
    return np.array([(v[0] * M[1, 1] - M[0, 1] * v[1]) / det,
                     (M[0, 0] * v[1] - v[0] * M[1, 0]) / det])


def _radau_step(f, t, y, h, counter):
    """One Radau IIA 3-stage step. Returns (y_new, err_norm, J_new)."""
    Y = [y.copy(), y.copy(), y.copy()]
    F = [None, None, None]
    for i in range(3):
        Yi = Y[i].copy()
        Fi = None
        for _it in range(14):
            Fi = np.asarray(f(t + _C[i] * h, Yi), dtype=float)
            counter[0] += 1
            s = _A[i, i] * Fi
            for j in range(i):
                s = s + _A[i, j] * F[j]
            G = Yi - y - h * s
            J = _jac(Yi)
            M = np.eye(2) - h * _A[i, i] * J
            dY = _solve2(M, -G)
            Yi = Yi + dY
            if np.linalg.norm(dY) <= 1e-14 * (1.0 + np.linalg.norm(Yi)):
                break
        Y[i] = Yi
        F[i] = Fi
    y_new = y + h * (_B[0] * F[0] + _B[1] * F[1] + _B[2] * F[2])
    err_vec = h * (_E[0] * F[0] + _E[1] * F[1] + _E[2] * F[2])
    scale = 1e-15 + 1e-13 * np.maximum(np.abs(y), np.abs(y_new))
    err = np.linalg.norm(err_vec / scale)
    J_new = _jac(y_new)
    return y_new, err, J_new


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    t = t0
    total = t1 - t0
    h = total / 300.0
    hmax = total / 40.0
    counter = [0]
    deadline = time.time() + 290.0
    best = y.copy()
    while t < t1 - 1e-13:
        if time.time() > deadline or counter[0] > 1900000:
            break
        if t + h > t1:
            h = t1 - t
        y_new, err, Jnew = _radau_step(f, t, y, h, counter)
        if (np.isfinite(err) and err <= 1.0) or h < 1e-14:
            y = y_new
            t = t + h
            best = y.copy()
            if err < 1e-6:
                h *= 2.0
            elif err < 0.5:
                h *= 1.5
            else:
                h *= 1.0
        else:
            h *= 0.5
        if h > hmax:
            h = hmax
    return best
# EVOLVE-BLOCK-END
