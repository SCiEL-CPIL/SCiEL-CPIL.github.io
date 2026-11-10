"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.integrate import solve_ivp
# EVOLVE-BLOCK-START

def _ode_rhs(x, y, Om0, w0, wa):
    a = np.exp(x)
    Om = Om0 * a**(-3)
    Ode = 1.0 - Om
    w = w0 + wa * (1.0 - a)
    coeff = 1.5 * (Om + Ode * (1.0 + 3.0 * w))
    return [y[1], -coeff * y[1] + 1.5 * Om * y[0]]

def _growing_mode(x_grid, Om0, w0, wa):
    """Solve the ODE and extract the pure growing mode via eigenvalue at early time."""
    x0 = np.log(0.01)
    sol1 = solve_ivp(_ode_rhs, [x0, 0.0], [1.0, 0.0], args=(Om0, w0, wa),
                     method='DOP853', rtol=1e-10, atol=1e-12, dense_output=True)
    sol2 = solve_ivp(_ode_rhs, [x0, 0.0], [0.0, 1.0], args=(Om0, w0, wa),
                     method='DOP853', rtol=1e-10, atol=1e-12, dense_output=True)
    
    D1_0 = sol1.sol(x0)[0]
    D1p_0 = sol1.sol(x0)[1]
    D2_0 = sol2.sol(x0)[0]
    D2p_0 = sol2.sol(x0)[1]
    
    # For the pure growing mode: D'/D = 1 at early times (matter-dominated)
    denom = D2p_0 - D2_0
    if abs(denom) < 1e-15:
        D = sol2.sol(x_grid)[0]
        Dp = sol2.sol(x_grid)[1]
    else:
        A = 1.0
        B = -A * (D1p_0 - D1_0) / denom
        D = A * sol1.sol(x_grid)[0] + B * sol2.sol(x_grid)[0]
        Dp = A * sol1.sol(x_grid)[1] + B * sol2.sol(x_grid)[1]
    
    Dn = D / D[-1]
    fn = Dp / D
    return fn, Dn

def _gamma_of(a, Om0, w0, wa, coeffs):
    c0, c1, c2, c3, c4, c5 = coeffs
    Om = Om0 * a**(-3)
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    return c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q

def _fit_coefficients():
    # Representative CPL cosmologies covering the parameter space
    params = [
        (0.30, -1.0, 0.0),
        (0.30, -0.9, 0.3),
        (0.30, -1.1, 0.5),
        (0.25, -1.0, 0.2),
        (0.35, -0.8, 0.4),
        (0.30, -1.0, -0.5),
        (0.28, -1.2, 0.6),
        (0.32, -0.7, 0.1),
        (0.20, -1.0, 0.0),
        (0.40, -1.0, 0.0),
        (0.30, -1.5, 0.5),
        (0.30, -0.5, 0.5),
    ]
    
    a_grid = np.logspace(-2, 0, 300)
    x_grid = np.log(a_grid)
    
    rows = []
    targets = []
    weights = []
    
    for (Om0, w0, wa) in params:
        fn, Dn = _growing_mode(x_grid, Om0, w0, wa)
        Om = Om0 * a_grid**(-3)
        q = 1.0 - Om
        u = 1.0 + w0 + wa * (1.0 - a_grid)
        
        valid = (Om > 1e-4) & (Om < 0.999) & (fn > 1e-10) & (np.isfinite(fn))
        if np.sum(valid) < 10:
            continue
            
        gamma_target = np.log(fn[valid]) / np.log(Om[valid])
        valid2 = (gamma_target > 0.05) & (gamma_target < 1.5) & np.isfinite(gamma_target)
        if np.sum(valid2) < 10:
            continue
            
        a_v = a_grid[valid][valid2]
        Om_v = Om[valid][valid2]
        q_v = q[valid][valid2]
        u_v = u[valid][valid2]
        gamma_v = gamma_target[valid2]
        
        w = 1.0 / (1.0 + (np.abs(np.log(Om_v)) - 1.0)**2)
        
        for i in range(len(a_v)):
            ai = a_v[i]
            ui = u_v[i]
            qi = q_v[i]
            rows.append([1.0, qi, qi**2, ui, wa * ai, ui * qi])
            targets.append(gamma_v[i])
            weights.append(w[i])
    
    if len(rows) < 100:
        return [0.55, 0.0, 0.0, 0.05, 0.1, 0.0]
    
    A = np.array(rows)
    b = np.array(targets)
    w = np.array(weights)
    
    Aw = A * np.sqrt(w)[:, None]
    bw = b * np.sqrt(w)
    
    sol, res, rank, sv = np.linalg.lstsq(Aw, bw, rcond=None)
    coeffs = sol.tolist()
    
    coeffs = [float(np.clip(c, -10.0, 10.0)) for c in coeffs]
    
    # Local refinement: minimize max relative error of f and D
    def _score(coeffs):
        max_err = 0.0
        for (Om0, w0, wa) in params:
            fn_num, Dn_num = _growing_mode(x_grid, Om0, w0, wa)
            Om = Om0 * a_grid**(-3)
            gamma = _gamma_of(a_grid, Om0, w0, wa, coeffs)
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return 10.0
            f_approx = Om**gamma
            err_f = np.abs(f_approx - fn_num) / np.abs(fn_num)
            x_vals = x_grid
            f_vals = f_approx
            cum_int = np.zeros_like(x_vals)
            for i in range(len(x_vals) - 2, -1, -1):
                cum_int[i] = cum_int[i+1] + 0.5 * (f_vals[i] + f_vals[i+1]) * (x_vals[i+1] - x_vals[i])
            D_approx = np.exp(-cum_int)
            err_D = np.abs(D_approx - Dn_num) / np.abs(Dn_num)
            mask = (Dn_num > 1e-10) & (fn_num > 1e-10)
            if np.any(mask):
                max_err = max(max_err, np.max(err_f[mask]), np.max(err_D[mask]))
        return max_err * 100.0
    
    best_coeffs = coeffs[:]
    best_score = _score(best_coeffs)
    
    for _ in range(20):
        improved = False
        for i in range(6):
            for delta in [0.01, -0.01, 0.05, -0.05, 0.1, -0.1]:
                test = best_coeffs[:]
                test[i] = np.clip(test[i] + delta, -10.0, 10.0)
                s = _score(test)
                if s < best_score:
                    best_score = s
                    best_coeffs = test
                    improved = True
        if not improved:
            break
    
    return best_coeffs

def construct():
    # Physics-constrained affine: optimize all 6 coefficients against ODE reference.
    try:
        coeffs = _fit_coefficients()
        # Clip to [-10,10] for safety
        coeffs = [float(np.clip(c, -10.0, 10.0)) for c in coeffs]
        # Sanity: ensure finite
        if not all(np.isfinite(c) for c in coeffs):
            raise ValueError
        return coeffs
    except Exception:
        # Fallback to a reasonable constant-gamma with mild CPL correction
        return [0.55, 0.0, 0.0, 0.05, 0.1, 0.0]
# EVOLVE-BLOCK-END
