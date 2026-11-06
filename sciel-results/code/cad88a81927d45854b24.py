"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _Om_a(a, Om0, w0, wa):
    """Om(a) for CPL dark energy."""
    a = np.asarray(a, dtype=float)
    # E^2 = Om_m a^-3 + Om_r a^-4 + Om_L a^(-3(1+w0+wa))
    # Use Om_m = Om0, Om_r = 0 (matter only, no radiation for simplicity)
    # Actually, let's use standard: E^2 = Om_m a^-3 + Om_L a^(-3(1+w0+wa))
    # Om_L = 1 - Om_m
    Om_m = Om0
    Om_L = 1.0 - Om_m
    u = 1.0 + w0 + wa * (1.0 - a)
    E2 = Om_m * a**(-3) + Om_L * a**(-3*u)
    return Om_m * a**(-3) / E2

def _gamma_func(c, a, Om0, w0, wa):
    """Compute gamma from coefficients."""
    Om_a = _Om_a(a, Om0, w0, wa)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    c0, c1, c2, c3, c4, c5 = c
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
    return gamma

def _f_of_a(c, a, Om0, w0, wa):
    """f = Om(a)^gamma."""
    gamma = _gamma_func(c, a, Om0, w0, wa)
    Om_a = _Om_a(a, Om0, w0, wa)
    return Om_a ** gamma

def _D_normalized(c, a, Om0, w0, wa, n=500):
    """D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)."""
    # Integrate f(exp(x)) from x=ln(a) to x=0
    # If a < 1, ln(a) < 0, so integral from ln(a) to 0
    # D(a)/D(1) = exp(-int_{ln(a)}^0 f(exp(x)) dx)
    # For a=1, this is 1.
    # Let's compute the integral numerically
    a = np.asarray(a, dtype=float)
    D = np.ones_like(a)
    for i, ai in enumerate(a):
        if ai >= 1.0:
            D[i] = 1.0
            continue
        x = np.linspace(np.log(ai), 0.0, n)
        f_vals = _f_of_a(c, np.exp(x), Om0, w0, wa)
        integral = np.trapz(f_vals, x)
        D[i] = np.exp(-integral)
    return D

def _reference_f(a, Om0, w0, wa, n=1000):
    """Reference f from solving the ODE: f' + f^2 + 1.5*Om(a) = 0, f(1) = Om(1)^0.55."""
    # Actually, the standard growth equation in terms of f = dlnD/dlna:
    # f' + f^2 + 1.5*Om(a) = 0 (in matter-dominated, but with DE it's more complex)
    # The correct equation: f' + f^2 + 1.5*Om(a) = 0 is approximate.
    # More accurately: f' = -f^2 - 1.5*Om(a) + ... 
    # Let's use: f' = -f^2 - 1.5*Om(a) with f(1) = Om(1)^0.55
    # Actually, the standard is: f' + f^2 + 1.5*Om = 0
    a = np.asarray(a, dtype=float)
    f_ref = np.ones_like(a)
    # Integrate backward from a=1
    a_max = 1.0
    a_min = min(a)
    n_steps = 2000
    a_grid = np.linspace(a_max, a_min, n_steps)
    f_grid = np.zeros(n_steps)
    f_grid[0] = Om0 ** 0.55  # f(1) = Om(1)^gamma, Om(1)=Om0
    # f' = df/dlna = -f^2 - 1.5*Om(a)
    # dlna = da/a, so df/da = (1/a)*(-f^2 - 1.5*Om(a))
    for i in range(n_steps - 1):
        ai = a_grid[i]
        fi = f_grid[i]
        Om_i = _Om_a(ai, Om0, w0, wa)
        df_da = (1.0/ai) * (-fi**2 - 1.5*Om_i)
        da = a_grid[i+1] - a_grid[i]
        f_grid[i+1] = fi + df_da * da
    # Interpolate
    for i, ai in enumerate(a):
        if ai > a_max:
            f_ref[i] = f_grid[0]
        else:
            f_ref[i] = np.interp(ai, a_grid, f_grid)
    return f_ref

def _reference_D(a, Om0, w0, wa):
    """Reference D from integrating f_ref."""
    a = np.asarray(a, dtype=float)
    D = np.ones_like(a)
    a_min = min(a)
    n_steps = 2000
    a_grid = np.linspace(a_min, 1.0, n_steps)
    f_grid = np.zeros(n_steps)
    f_grid[-1] = Om0 ** 0.55
    for i in range(n_steps - 1, 0, -1):
        ai = a_grid[i]
        fi = f_grid[i]
        Om_i = _Om_a(ai, Om0, w0, wa)
        df_da = (1.0/ai) * (-fi**2 - 1.5*Om_i)
        da = a_grid[i-1] - a_grid[i]
        f_grid[i-1] = fi + df_da * da
    # D(a)/D(1) = exp(int_{a}^{1} f(a')/a' da')
    # = exp(int_{ln(a)}^{0} f(exp(x)) dx)
    # So D(a)/D(1) = exp(int_{ln(a)}^0 f(exp(x)) dx)
    for i, ai in enumerate(a):
        if ai >= 1.0:
            D[i] = 1.0
            continue
        x = np.linspace(np.log(ai), 0.0, 500)
        f_vals = np.interp(np.exp(x), a_grid, f_grid)
        integral = np.trapz(f_vals, x)
        D[i] = np.exp(integral)
    return D

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Validation cosmologies (typical values)
    # Om0 in [0.2, 0.35], w0 in [-1.2, 0.0], wa in [-0.5, 0.5]
    models = [
        (0.27, -1.0, 0.0),    # LCDM
        (0.27, -1.0, 0.3),    # CPL positive wa
        (0.27, -1.0, -0.3),   # CPL negative wa
        (0.30, -0.9, 0.2),    # varying Om0
        (0.25, -1.1, 0.1),    # another
        (0.35, -0.8, -0.2),   # high Om0
        (0.20, -1.0, 0.4),    # low Om0, high wa
        (0.30, -1.2, 0.0),    # low w0
    ]
    
    a_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    # Precompute reference values
    ref_data = []
    for (Om0, w0, wa) in models:
        f_ref = _reference_f(a_vals, Om0, w0, wa)
        D_ref = _reference_D(a_vals, Om0, w0, wa)
        ref_data.append((Om0, w0, wa, f_ref, D_ref))
    
    def objective(c):
        if time.time() > deadline:
            return 1e10
        c = np.clip(c, -10, 10)
        # Check gamma constraint
        max_err = 0.0
        for (Om0, w0, wa, f_ref, D_ref) in ref_data:
            gamma = _gamma_func(c, a_vals, Om0, w0, wa)
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return 1e10
            f_app = _f_of_a(c, a_vals, Om0, w0, wa)
            D_app = _D_normalized(c, a_vals, Om0, w0, wa)
            # Relative errors
            err_f = np.abs(f_app - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
            err_D = np.abs(D_app - D_ref) / np.maximum(np.abs(D_ref), 1e-10)
            max_err = max(max_err, np.max(err_f), np.max(err_D))
        return max_err
    
    # Initial guess: constant gamma ~ 0.55
    c0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Try Nelder-Mead first
    best_c = c0.copy()
    best_obj = objective(c0)
    
    try:
        res = minimize(objective, c0, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
        if res.fun < best_obj:
            best_c = res.x
            best_obj = res.fun
    except:
        pass
    
    if time.time() < deadline - 5:
        # Try L-BFGS-B with bounds
        try:
            bounds = [(-10, 10)] * 6
            res2 = minimize(objective, best_c, method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 100, 'gtol': 1e-6})
            if res2.fun < best_obj:
                best_c = res2.x
                best_obj = res2.fun
        except:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 2:
        try:
            res3 = minimize(objective, best_c, method='Nelder-Mead',
                           options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8})
            if res3.fun < best_obj:
                best_c = res3.x
                best_obj = res3.fun
        except:
            pass
    
    # Ensure coefficients are in [-10, 10]
    best_c = np.clip(best_c, -10, 10)
    
    return list(best_c)
# EVOLVE-BLOCK-END
