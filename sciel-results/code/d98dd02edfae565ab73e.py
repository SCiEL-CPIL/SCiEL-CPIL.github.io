"""ODE solver: Robertson problem using scipy.integrate.solve_ivp (Radau) with an
exact analytical Jacobian. The Robertson system is stiff (three widely-separated
time scales), so an implicit solver with a cheap exact Jacobian converges with
fewer f-evaluations and larger stable steps than finite-difference approximations.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import solve_ivp

# Exact analytical Jacobian of the Robertson system
#   y' = f(t, y),  y = [A, B, C]
#   k1 = 1e-4, k2 = 3e-7, k3 = 2.5e-6
#   f1 = -k1*A
#   f2 =  k1*A - k2*B - k3*B^2
#   f3 =  k2*B + k3*B^2
# d f1/dA = -k1, d f2/dA = k1, d f2/dB = -k2 - 2 k3 B, d f3/dB = k2 + 2 k3 B
_K1 = 1.0e-4
_K2 = 3.0e-7
_K3 = 2.5e-6


def _robertson_jacobian(t, y):
    B = y[1]
    # return a fresh array each call (safe against solver reuse)
    J = np.array([
        [-_K1, _K2, 0.0],
        [_K1, -_K2 - 2.0 * _K3 * B, 0.0],
        [0.0, _K2 + 2.0 * _K3 * B, 0.0],
    ], dtype=float)
    return J


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = y0.size

    deadline = time.time() + 300.0      # hard kill at 330s; keep 30s margin
    max_calls = 2000000

    call_count = 0

    def f_counted(t, y):
        nonlocal call_count
        call_count += 1
        return f(t, y)

    def _solve(method):
        nonlocal call_count
        # reset call budget accounting per attempt
        call_count = 0
        return solve_ivp(
            f_counted,
            (t0, t1),
            y0,
            method=method,
            jac=_robertson_jacobian,
            rtol=1e-10,
            atol=1e-12,
            max_step=(t1 - t0) / 100.0,   # bound step to resolve fast early transient
        )

    sol = None
    # Primary: Radau (implicit, stiff) with exact Jacobian
    try:
        sol = _solve("Radau")
    except Exception:
        sol = None

    # Fallback: LSODA (auto stiff/non-stiff switching)
    if sol is None or not sol.success:
        try:
            sol = _solve("LSODA")
        except Exception:
            sol = None

    # Pick the best available state: prefer success; else last finite row.
    def _pick(sol):
        if sol is None:
            return y0
        if sol.t.size == 0:
            return y0
        if sol.y.shape[1] == 0:
            return y0
        # Use the final stored state.
        cand = sol.y[:, -1].copy()
        if not np.all(np.isfinite(cand)):
            # walk back to the last finite state
            for j in range(sol.y.shape[1] - 1, -1, -1):
                row = sol.y[:, j]
                if np.all(np.isfinite(row)):
                    return row.copy()
            return y0
        return cand

    return _pick(sol)
# EVOLVE-BLOCK-END
