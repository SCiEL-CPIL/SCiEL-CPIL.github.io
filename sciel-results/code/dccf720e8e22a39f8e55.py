import numpy as np
import time
from scipy.optimize import minimize, root

_T0 = time.time()
_DEADLINE = _T0 + 195.0

# Precompute I[n] = integral x^{2n} e^{-x^2} dx
I = np.zeros(25)
I[0] = np.sqrt(np.pi)
for n in range(1, 25):
    I[n] = I[n - 1] * (2.0 * n - 1.0) / (2.0 * n)

# Hermite polynomial coefficients in power basis x^{2k}
H0_c = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
H4_c = np.array([3.0, -6.0, 1.0, 0.0, 0.0, 0.0, 0.0])
H8_c = np.array([105.0, -420.0, 210.0, -28.0, 1.0, 0.0, 0.0])
H12_c = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])

c12 = np.array([-1.0 / 945.0, -3.0 / 945.0, -105.0 / 945.0])

col0 = H0_c + c12[0] * H12_c
col1 = H4_c + c12[1] * H12_c
col2 = H8_c + c12[2] * H12_c
A_mat = np.column_stack((col0, col1, col2))

sgn = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
K7 = np.arange(7)
# Precompute I[i+j+n] for all i,j in [0,7), n in {0,1}
I0 = I[K7[:, None] + K7[None, :]]
I1 = I[K7[:, None] + K7[None, :] + 1]


def _get_p(a):
    return A_mat @ a


def _compute_bound_and_grad(a):
    p = A_mat @ a
    q = p * sgn

    Mx0 = float(p @ I0 @ p)
    Mx1 = float(p @ I1 @ p)
    Mxi0 = float(q @ I0 @ q)
    Mxi1 = float(q @ I1 @ q)

    if Mx0 <= 1e-14 or Mxi0 <= 1e-14 or Mx1 <= 1e-14 or Mxi1 <= 1e-14:
        return 1e18, np.zeros(3)

    vx = Mx1 / Mx0
    vxi = Mxi1 / Mxi0
    prod = vx * vxi
    if prod <= 0:
        return 1e18, np.zeros(3)

    bound = np.sqrt(prod)
    if not np.isfinite(bound):
        return 1e18, np.zeros(3)

    # Gradient
    Sx0 = I0 @ p
    Sx1 = I1 @ p
    Sxi0 = I0 @ q
    Sxi1 = I1 @ q

    dMx0 = 2.0 * (A_mat.T @ Sx0)
    dMx1 = 2.0 * (A_mat.T @ Sx1)
    dMxi0 = 2.0 * (A_mat.T @ Sxi0)
    dMxi1 = 2.0 * (A_mat.T @ Sxi1)

    inv_Mx0_sq = 1.0 / (Mx0 * Mx0)
    inv_Mxi0_sq = 1.0 / (Mxi0 * Mxi0)

    dvx = (dMx1 * Mx0 - Mx1 * dMx0) * inv_Mx0_sq
    dvxi = (dMxi1 * Mxi0 - Mxi1 * dMxi0) * inv_Mxi0_sq

    factor = 0.5 / bound
    grad = factor * (vxi * dvx + vx * dvxi)

    if not np.all(np.isfinite(grad)):
        return bound, np.zeros(3)

    return bound, grad


def _stationary_equations(a):
    p = A_mat @ a
    q = p * sgn

    Mx0 = float(p @ I0 @ p)
    Mx1 = float(p @ I1 @ p)
    Mxi0 = float(q @ I0 @ q)
    Mxi1 = float(q @ I1 @ q)

    if Mx0 <= 1e-15 or Mxi0 <= 1e-15 or Mx1 <= 1e-15 or Mxi1 <= 1e-15:
        return np.array([1e18, 1e18, 1e18])

    vx = Mx1 / Mx0
    vxi = Mxi1 / Mxi0

    Sx0 = I0 @ p
    Sx1 = I1 @ p
    Sxi0 = I0 @ q
    Sxi1 = I1 @ q

    dMx0 = 2.0 * (A_mat.T @ Sx0)
    dMx1 = 2.0 * (A_mat.T @ Sx1)
    dMxi0 = 2.0 * (A_mat.T @ Sxi0)
    dMxi1 = 2.0 * (A_mat.T @ Sxi1)

    inv_Mx0_sq = 1.0 / (Mx0 * Mx0)
    inv_Mxi0_sq = 1.0 / (Mxi0 * Mxi0)

    dvx = (dMx1 * Mx0 - Mx1 * dMx0) * inv_Mx0_sq
    dvxi = (dMxi1 * Mxi0 - Mxi1 * dMxi0) * inv_Mxi0_sq

    grad_F = vxi * dvx + vx * dvxi

    return grad_F


def construct():
    seed_params = np.array([-1.13, 0.04, 0.000308642])
    seed_val, _ = _compute_bound_and_grad(seed_params)

    best_val = seed_val
    best_params = seed_params.copy()

    def _try_improve(a_sol):
        nonlocal best_val, best_params
        if not np.all(np.isfinite(a_sol)):
            return
        val, _ = _compute_bound_and_grad(a_sol)
        if not np.isfinite(val) or val >= best_val:
            return
        # Polish
        try:
            res = minimize(_compute_bound_and_grad, a_sol, method='L-BFGS-B',
                           jac=True, options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < val:
                val = res.fun
                a_sol = res.x
        except Exception:
            pass
        if val < best_val:
            best_val = val
            best_params = a_sol.copy()

    # Build a rich set of starting points
    starts = [seed_params.copy()]

    # Dense grid around the seed
    for da0 in np.linspace(-0.5, 0.5, 7):
        for da4 in np.linspace(-0.1, 0.1, 7):
            for da8 in np.linspace(-0.005, 0.005, 5):
                starts.append(seed_params + np.array([da0, da4, da8]))

    # Broader grid
    for a0 in np.linspace(-3.0, 1.5, 8):
        for a4 in np.linspace(-0.3, 0.3, 8):
            for a8 in np.linspace(-0.02, 0.02, 5):
                starts.append(np.array([a0, a4, a8]))

    # Random starts
    rng = np.random.default_rng(42)
    for _ in range(200):
        a0 = rng.uniform(-3.5, 2.0)
        a4 = rng.uniform(-0.6, 0.6)
        a8 = rng.uniform(-0.06, 0.06)
        starts.append(np.array([a0, a4, a8]))

    # Log-scale random starts (to explore different magnitudes)
    for _ in range(100):
        a0 = rng.uniform(-3.5, 2.0)
        a4 = np.sign(rng.uniform(-1, 1)) * 10 ** rng.uniform(-2, 0.5)
        a8 = np.sign(rng.uniform(-1, 1)) * 10 ** rng.uniform(-4, -1)
        starts.append(np.array([a0, a4, a8]))

    # Run root-finding from each start
    for start in starts:
        if time.time() > _DEADLINE - 8.0:
            break
        try:
            res = root(_stationary_equations, start, method='hybr', tol=1e-12, maxfev=200)
            if res.success and np.all(np.isfinite(res.x)):
                _try_improve(res.x)
        except Exception:
            pass

    # Also try direct L-BFGS-B from some starts (for non-stationary minima)
    for start in starts[:100]:
        if time.time() > _DEADLINE - 8.0:
            break
        try:
            res = minimize(_compute_bound_and_grad, start, method='L-BFGS-B',
                           jac=True, options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-11})
            if res.success and res.fun < best_val:
                _try_improve(res.x)
        except Exception:
            pass

    # Final high-precision polish
    if time.time() < _DEADLINE - 5.0:
        try:
            res = minimize(_compute_bound_and_grad, best_params, method='L-BFGS-B',
                           jac=True, options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass

    # Even tighter polish
    if time.time() < _DEADLINE - 3.0:
        try:
            res2 = minimize(_compute_bound_and_grad, best_params, method='L-BFGS-B',
                            jac=True, options={'maxiter': 2000, 'ftol': 1e-17, 'gtol': 1e-15})
            if res2.fun < best_val:
                best_val = res2.fun
                best_params = res2.x
        except Exception:
            pass

    # Perturbation escape
    if time.time() < _DEADLINE - 3.0:
        rng_polish = np.random.default_rng(123)
        for _ in range(10):
            if time.time() > _DEADLINE - 3.0:
                break
            try:
                pert = rng_polish.normal(0, 1e-7, size=3)
                start_p = best_params + pert
                res_p = minimize(_compute_bound_and_grad, start_p, method='L-BFGS-B',
                                 jac=True, options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13})
                if res_p.success and res_p.fun < best_val:
                    best_val = res_p.fun
                    best_params = res_p.x
            except Exception:
                pass

    if not np.all(np.isfinite(best_params)):
        return seed_params

    return np.array(best_params)
