"""Growth-index fit: solve GR ODE for reference, fit 6-coefficient gamma ansatz."""
import time
import numpy as np
from scipy.integrate import solve_ivp

_DEADLINE = 80.0
_T0 = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _T0)

# --- Cosmology helpers ---
def _Om(a, Om0, w0, wa):
    Om0 = float(Om0)
    x = np.asarray(a, dtype=float)
    x = np.where(x > 1e-12, x, 1e-12)
    y = (1.0 - x) / x
    w_eff = w0 + wa * y
    y1 = y / 3.0
    y2 = y * y / 2.0
    y3 = y * y * y / 6.0
    num = 1.0 + w_eff * y1 + w_eff * (1.0 + w_eff) * y2 + w_eff * (1.0 + w_eff) * (2.0 + 3.0 * w_eff) * y3
    return Om0 * x ** (-3.0) * num

def _Om_scalar(a, Om0, w0, wa):
    return float(_Om(np.array([a]), Om0, w0, wa)[0])

# --- Reference: solve the GR second-order ODE for D(a) ---
def _D_ref(a, Om0, w0, wa, n=2000):
    a = float(a)
    if a >= 1.0:
        return 1.0
    if a < 1e-3:
        a = 1e-3
    a_arr = np.linspace(a, 1.0, n)
    Om_arr = _Om(a_arr, Om0, w0, wa)
    H2 = Om_arr * a_arr ** (-3.0)
    f_ref = Om_arr ** 0.55
    # ODE: D'' + (1 + 1.5*Om + f) D' - 1.5*f D = 0, with D'(1)=f(1)=Om0^0.55, D(1)=1
    f1 = Om0 ** 0.55
    def rhs(x, y):
        i = int(x)
        if i >= n - 1:
            i = n - 2
        O = Om_arr[i]
        F = f_ref[i]
        return np.array([y[1], -(1.0 + 1.5 * O + F) * y[1] + 1.5 * F * y[0]])
    sol = solve_ivp(rhs, [0.0, float(n - 1)], [1.0, f1], t_eval=np.arange(n),
                    method='RK45', rtol=1e-10, atol=1e-12)
    D = sol.y[0]
    idx = int(np.argmin(np.abs(a_arr - a)))
    return float(D[idx])

def _f_ref(a, Om0, w0, wa):
    return float(_Om_scalar(a, Om0, w0, wa) ** 0.55)

# --- Approximation from coefficients ---
def _gamma(c, a, Om0, w0, wa):
    Om_a = _Om(a, Om0, w0, wa)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    return (c[0] + c[1] * q + c[2] * q ** 2 + c[3] * u + c[4] * wa * a + c[5] * u * q)

def _f_approx(c, a, Om0, w0, wa):
    Om_a = _Om(a, Om0, w0, wa)
    g = _gamma(c, a, Om0, w0, wa)
    return Om_a ** g

def _D_approx(c, a, Om0, w0, wa, n=800):
    a = float(a)
    if a >= 1.0:
        return 1.0
    a_arr = np.linspace(a, 1.0, n)
    Om_arr = _Om(a_arr, Om0, w0, wa)
    g_arr = _gamma(c, a_arr, Om0, w0, wa)
    f_arr = Om_arr ** g_arr
    ln_a = np.log(a_arr)
    # D(a)/D(1) = exp(- integral_{ln a}^{0} f exp(x) dx )
    integrand = f_arr * np.exp(ln_a)
    integral = np.trapz(integrand, ln_a)
    return float(np.exp(-integral))

# --- Objective ---
def _error(c, Om0, w0, wa, a_vals):
    best = 0.0
    for a in a_vals:
        fr = _f_ref(a, Om0, w0, wa)
        fa = _f_approx(c, a, Om0, w0, wa)
        if abs(fr) > 1e-12:
            best = max(best, abs(fa - fr) / abs(fr))
        Dr = _D_ref(a, Om0, w0, wa)
        Da = _D_approx(c, a, Om0, w0, wa)
        if abs(Dr) > 1e-12:
            best = max(best, abs(Da - Dr) / abs(Dr))
    return best

def _total_error(c, cosmo_grid, a_vals):
    best = 0.0
    for (Om0, w0, wa) in cosmo_grid:
        best = max(best, _error(c, Om0, w0, wa, a_vals))
    return best

def _check_gamma_range(c, cosmo_grid, a_vals):
    for (Om0, w0, wa) in cosmo_grid:
        for a in a_vals:
            g = _gamma(c, np.array([a]), Om0, w0, wa)[0]
            if g < 0.1 or g > 1.2:
                return False
    return True

# --- Optimization: Nelder-Mead with constraint penalty ---
def _penalized(c, cosmo_grid, a_vals):
    if not _check_gamma_range(c, cosmo_grid, a_vals):
        return 100.0
    for ci in c:
        if ci < -10.0 or ci > 10.0:
            return 100.0
    return _total_error(c, cosmo_grid, a_vals)

def _nelder_mead(f, x0, cosmo_grid, a_vals, max_iter=300):
    n = len(x0)
    # initial simplex
    simplex = [list(x0)]
    for i in range(n):
        pt = list(x0)
        step = 0.1 if x0[i] != 0.0 else 0.1
        pt[i] += step
        simplex.append(pt)
    vals = [f(pt, cosmo_grid, a_vals) for pt in simplex]
    best_c = simplex[int(np.argmin(vals))]
    best_val = min(vals)

    for it in range(max_iter):
        if _remaining() < 3.0:
            break
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        if vals[0] < best_val:
            best_val = vals[0]
            best_c = list(simplex[0])
        # centroid of all but worst
        worst = simplex[-1]
        centroid = np.mean(simplex[:-1], axis=0)
        # reflection
        sigma = 1.0
        xr = centroid + sigma * (centroid - np.array(worst))
        fr = f(xr, cosmo_grid, a_vals)
        if fr < vals[0]:
            # expansion
            xe = centroid + 2.0 * (centroid - np.array(worst))
            fe = f(xe, cosmo_grid, a_vals)
            if fe < fr:
                simplex[-1] = list(xe); vals[-1] = fe
            else:
                simplex[-1] = list(xr); vals[-1] = fr
        elif fr < vals[-2]:
            simplex[-1] = list(xr); vals[-1] = fr
        else:
            # contraction
            if fr < vals[-1]:
                xc = centroid + 0.5 * (np.array(worst) - centroid)
                fc = f(xc, cosmo_grid, a_vals)
                if fc <= fr:
                    simplex[-1] = list(xc); vals[-1] = fc
                else:
                    # shrink
                    for i in range(1, n + 1):
                        simplex[i] = [simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                        vals[i] = f(simplex[i], cosmo_grid, a_vals)
            else:
                for i in range(1, n + 1):
                    simplex[i] = [simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                    vals[i] = f(simplex[i], cosmo_grid, a_vals)
    return best_c, best_val

def construct():
    # Grid of cosmologies (Om0, w0, wa)
    cosmo_grid = [
        (0.30, -1.0, 0.0),
        (0.27, -1.0, 0.0),
        (0.30, -1.0, 0.5),
        (0.30, -1.0, -0.5),
        (0.27, -1.2, 0.8),
        (0.30, -0.9, 0.0),
        (0.25, -1.1, 0.3),
        (0.30, -1.0, 1.0),
        (0.30, -0.8, -0.3),
        (0.27, -1.3, 1.2),
    ]
    a_vals = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    # Initial guess: constant gamma
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_c = list(x0)
    best_val = _penalized(x0, cosmo_grid, a_vals)

    # Local Nelder-Mead
    c_opt, val_opt = _nelder_mead(_penalized, x0, cosmo_grid, a_vals)
    if val_opt < best_val:
        best_c = c_opt
        best_val = val_opt

    # Try a couple of random restarts if time permits
    rng = np.random.RandomState(42)
    for _ in range(3):
        if _remaining() < 5.0:
            break
        x0r = [0.55 + 0.2 * (rng.rand() - 0.5), 0.5 * (rng.rand() - 0.5),
               0.5 * (rng.rand() - 0.5), 0.3 * (rng.rand() - 0.5),
               0.3 * (rng.rand() - 0.5), 0.3 * (rng.rand() - 0.5)]
        c_opt, val_opt = _nelder_mead(_penalized, x0r, cosmo_grid, a_vals)
        if val_opt < best_val:
            best_c = c_opt
            best_val = val_opt

    # Clamp to [-10,10]
    best_c = [max(-10.0, min(10.0, float(c))) for c in best_c]
    return best_c
