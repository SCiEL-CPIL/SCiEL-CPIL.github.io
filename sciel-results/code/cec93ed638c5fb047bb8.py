"""Optimized six-coefficient growth-index fit for CPL dark-energy cosmologies."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_DEADLINE = 82.0  # safety margin under 90s wall-clock kill
_T0 = time.time()

# --- Reference cosmology models (disjoint validation-like set) ---
# (Om0, w0, wa)
_MODELS = [
    (0.315, -1.0, 0.0),
    (0.30, -1.0, 0.0),
    (0.32, -1.0, 0.0),
    (0.315, -1.0, 0.5),
    (0.315, -1.0, -0.5),
    (0.315, -0.9, 0.0),
    (0.315, -1.1, 0.0),
    (0.28, -1.0, 0.5),
    (0.35, -0.95, 0.3),
    (0.30, -1.05, 0.4),
]

# Precompute a grid of a values (log-spaced) for integration/evaluation
_A_GRID = np.logspace(-1.5, 0.0, 400)  # a in [~0.03, 1]
_A_GRID = _A_GRID[_A_GRID < 1.0]  # exclude a=1 for the integral endpoint
_N_A = len(_A_GRID)


def _w_of_a(w0, wa, a):
    return w0 + wa * (1.0 - a)


def _Om_a(Om0, w0, wa, a):
    """Omega_m(a) for CPL dark energy."""
    # E^2 = Om0 a^-3 + (1-Om0) a^(-3(1+w0+wa)) * a^(-wa)
    # = Om0 a^-3 + Ode0 * a^(-3-3w0-3wa) * a^(-wa)
    # Actually: Ode(a) = Ode0 * a^(-3(1+w0+wa)) * a^(-wa) ... let me be careful.
    # rho_de(a) = rho_de0 * a^{-3(1+w(a))} where w(a)=w0+wa(1-a)
    # = rho_de0 * a^{-3(1+w0+wa(1-a))} = rho_de0 * a^{-3(1+w0+wa)} * a^{3wa*a}
    # So E^2(a) = Om0 a^{-3} + Ode0 * a^{-3(1+w0+wa)} * a^{3*wa*a}
    # For numerical integration, we work in x = ln(a):
    # a = e^x, a^{-3} = e^{-3x}, a^{-3(1+w0+wa)} = e^{-3(1+w0+wa)x}
    # a^{3*wa*a} = exp(3*wa*e^x)
    # E^2 = Om0 e^{-3x} + Ode0 e^{-3(1+w0+wa)x} * exp(3*wa*e^x)
    # Om(a) = Om0 e^{-3x} / E^2
    # But for the ODE we need Om(a) directly. Let's compute in a-space.
    Ode0 = 1.0 - Om0
    # rho_de(a)/rho_crit(a) = Ode0 * a^{-3(1+w0+wa)} * exp(3*wa*a)
    # This is expensive. Alternative: use the standard form
    # E^2(a) = Om0/a^3 + Ode0 * a^{-3(1+w0+wa)} * a^{-wa}  -- NO
    # Let me use: E^2(a) = Om0 a^{-3} + Ode0 a^{-3(1+w0+wa)} exp(3 wa a)
    # Actually the cleanest: d(ln rho_de)/d(ln a) = -3(1+w(a))
    # rho_de(a) = rho_de0 * exp(-3 integral_1^a (1+w(a'))/a' da')
    # = rho_de0 * exp(-3 integral_1^a (1+w0+wa(1-a'))/a' da')
    # = rho_de0 * exp(-3 integral_1^a (1+w0+wa)/a' da' - 3wa integral_1^a da')
    # = rho_de0 * exp(-3(1+w0+wa) ln(a) - 3wa(a-1))
    # = rho_de0 * a^{-3(1+w0+wa)} * exp(-3wa(a-1))
    # = rho_de0 * a^{-3(1+w0+wa)} * exp(3wa(1-a))
    # So E^2(a) = Om0 a^{-3} + Ode0 a^{-3(1+w0+wa)} * exp(3*wa*(1-a))
    # Om(a) = Om0 a^{-3} / E^2(a)
    # For a in [0.03, 1], this is fine.
    a_safe = np.maximum(a, 1e-10)
    term_m = Om0 * a_safe ** (-3.0)
    exponent = -3.0 * (1.0 + w0 + wa) * np.log(a_safe) + 3.0 * wa * (1.0 - a_safe)
    term_de = Ode0 * np.exp(exponent)
    E2 = term_m + term_de
    Om_a = term_m / E2
    return Om_a, E2


def _true_f_and_D(Om0, w0, wa, a_grid):
    """Solve the GR second-order ODE for D(a) and compute f(a) = d ln D / d ln a."""
    # ODE: D'' + [3/a + H'/H] D' - (4/3) Om(a) D = 0  (in a-space, prime = d/da)
    # Or in x=ln a: D_xx + (3/2 + (1/2)w(a)) D_x - (2/3) Om(a) D = 0
    # where D_x = dD/dx = a dD/da, D_xx = a^2 d2D/da2 + a dD/da
    # Standard form: D''(x) + (3/2 + (1/2)w(x)) D'(x) - (2/3) Om(x) D(x) = 0
    # Wait, let me re-derive. The growth equation in x=ln a:
    # D'' + (3/2 + H'/H) D' - (4 pi G rho_m / 3) D = 0  ... in units where c=1
    # Actually: D'' + (3/2 + (1/2)w_eff) D' - (2/3) Om(a) D = 0
    # where H'/H = dH/dx / H = (1/2)(w(a) + 1) ... no.
    # H(x) = H0 E(x), dH/dx = H0 E'(x), H'/H = E'/E
    # E^2 = Om0 e^{-3x} + Ode0 e^{-3(1+w0+wa)x} e^{3wa(e^x)}
    # Let me just use the standard form:
    # D''(x) + (3/2 + (1/2) w(a)) D'(x) - (2/3) Om(a) D(x) = 0
    # This is the commonly used form for CPL.
    # Actually the correct coefficient: the term is (3/2 + (1/2)(1+w) * something)...
    # Let me use: D'' + [3/2 + (1/2)w(a)] D' - (2/3)Om(a)D = 0
    # This is standard for flat CPL.
    
    x_grid = np.log(a_grid)
    n = len(x_grid)
    dx = x_grid[1] - x_grid[0]
    
    # Compute Om(a) and w(a) on grid
    Om_arr, E2_arr = _Om_a(Om0, w0, wa, a_grid)
    w_arr = w0 + wa * (1.0 - a_grid)
    
    # Solve ODE: y'' + p(x) y' - q(x) y = 0
    # p(x) = 3/2 + w(x)/2
    # q(x) = (2/3) Om(x)
    # Initial conditions at a=1 (x=0): D(1)=1, D'(1) = f(1) ~ 0.55 (but we want the true value)
    # Better: use D(1)=1, D'(1)=f_1 where f_1 is the true f at a=1.
    # We can compute f_1 from the ODE at a=1: f_1 = (2/3)Om(1) / (3/2 + w(1)/2)
    # Since at a=1: D''(0) + p(0)D'(0) - q(0)D(0) = 0, and for a linearized solution
    # near a=1, D ~ a^f, so D' = f, D'' = f(f-1). 
    # f(f-1) + p(0)f - q(0) = 0 => f^2 + (p(0)-1)f - q(0) = 0
    # f = [-(p(0)-1) + sqrt((p(0)-1)^2 + 4q(0))]/2
    p0 = 1.5 + w0 / 2.0  # w(1) = w0
    q0 = (2.0/3.0) * Om0
    disc = (p0 - 1.0)**2 + 4.0 * q0
    f1 = (-(p0 - 1.0) + math.sqrt(disc)) / 2.0
    
    # Integrate from x=0 backward to x=x_grid[0]
    # Use RK4 in x-space
    # State: [D, D']
    def rhs(x, state):
        a_val = math.exp(x)
        # Compute Om(a) and w(a) at this a
        Ode0 = 1.0 - Om0
        a_s = max(a_val, 1e-10)
        term_m = Om0 * a_s ** (-3.0)
        exponent = -3.0 * (1.0 + w0 + wa) * math.log(a_s) + 3.0 * wa * (1.0 - a_s)
        term_de = Ode0 * math.exp(exponent)
        E2 = term_m + term_de
        Om_val = term_m / E2
        w_val = w0 + wa * (1.0 - a_val)
        p_val = 1.5 + w_val / 2.0
        q_val = (2.0/3.0) * Om_val
        D, Dp = state
        Dpp = -p_val * Dp + q_val * D
        return [Dp, Dpp]
    
    # Integrate from x=0 to x_grid[0] (negative direction)
    x_start = 0.0
    x_end = x_grid[0]
    n_steps = max(int(abs(x_end - x_start) / (dx * 0.5)), 2000)
    h = (x_end - x_start) / n_steps
    
    state = [1.0, f1]
    D_arr = np.zeros(n)
    Dp_arr = np.zeros(n)
    
    # We need values at x_grid points. Let's do a simple adaptive or fixed-step.
    # For efficiency, use a fixed number of steps and interpolate.
    # Actually, let's just integrate step by step and store at grid points.
    # Simpler: use scipy's solve_ivp if available, but let's do RK4 manually for control.
    
    # Pre-compute for speed: we'll do a single RK4 pass with fine steps
    # and record at grid points.
    # Grid in x: x_grid from x_grid[0] to 0
    # Let's integrate from x=0 down to x_grid[0]
    
    # For speed, precompute Om and w on a fine x grid
    # Actually, let's just do it straightforwardly with a fine step.
    
    # To be fast, use a moderate number of steps
    n_fine = 5000
    h_fine = (x_end - x_start) / n_fine
    
    # We need D and D' at each x_grid point
    # Map x_grid to fine grid indices
    # x_grid[0] = x_end, x_grid[-1] ~ close to 0
    
    D_vals = np.zeros(n)
    Dp_vals = np.zeros(n)
    
    # State at x=0
    state = [1.0, f1]
    # We integrate from x=0 to x_end (negative h)
    # Record at grid points
    
    # Create a fine x array
    x_fine = np.linspace(x_start, x_end, n_fine + 1)
    
    # RK4
    D_fine = np.zeros(n_fine + 1)
    Dp_fine = np.zeros(n_fine + 1)
    D_fine[0] = 1.0
    Dp_fine[0] = f1
    
    for i in range(n_fine):
        x_i = x_fine[i]
        s = [D_fine[i], Dp_fine[i]]
        k1 = rhs(x_i, s)
        s2 = [s[0] + 0.5*h_fine*k1[0], s[1] + 0.5*h_fine*k1[1]]
        k2 = rhs(x_i + 0.5*h_fine, s2)
        s3 = [s[0] + 0.5*h_fine*k2[0], s[1] + 0.5*h_fine*k2[1]]
        k3 = rhs(x_i + 0.5*h_fine, s3)
        s4 = [s[0] + h_fine*k3[0], s[1] + h_fine*k3[1]]
        k4 = rhs(x_i + h_fine, s4)
        D_fine[i+1] = s[0] + (h_fine/6.0)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        Dp_fine[i+1] = s[1] + (h_fine/6.0)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    
    # Interpolate to x_grid
    # x_grid is decreasing (from x_end to ~0), x_fine is also decreasing
    # x_fine goes from x_start=0 to x_end=x_grid[0]
    # x_grid goes from x_grid[0] to ~0
    # So x_grid is reversed relative to x_fine
    # Let's just interpolate using np.interp
    # x_fine is monotonically decreasing, x_grid is monotonically decreasing
    # np.interp expects increasing x, so flip
    x_fine_inc = x_fine[::-1]
    D_fine_inc = D_fine[::-1]
    Dp_fine_inc = Dp_fine[::-1]
    x_grid_inc = x_grid[::-1]
    
    D_vals_inc = np.interp(x_grid_inc, x_fine_inc, D_fine_inc)
    Dp_vals_inc = np.interp(x_grid_inc, x_fine_inc, Dp_fine_inc)
    
    D_vals = D_vals_inc[::-1]
    Dp_vals = Dp_vals_inc[::-1]
    
    # f(a) = D'(x) = a D'(a) ... wait, D'(x) = dD/dx = a dD/da
    # The growth rate f = d ln D / d ln a = D'(x)/D(x)
    f_vals = Dp_vals / D_vals
    
    # Normalized D: D(a)/D(1) = D_vals / D_vals at a=1
    # D(1) = D at x=0 = 1 (by our IC)
    D_norm = D_vals / 1.0  # D(1)=1
    
    return f_vals, D_norm


def _gamma_from_coeffs(coeffs, a, Om_a, w0, wa):
    """Compute gamma from the six coefficients."""
    c0, c1, c2, c3, c4, c5 = coeffs
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
    return gamma


def _f_from_coeffs(coeffs, a, Om_a, w0, wa):
    """Compute f = Om(a)^gamma."""
    gamma = _gamma_from_coeffs(coeffs, a, Om_a, w0, wa)
    # f = Om(a)^gamma
    # Clip Om_a to avoid log(0)
    Om_safe = np.maximum(Om_a, 1e-10)
    f = np.power(Om_safe, gamma)
    return f, gamma


def _D_norm_from_coeffs(coeffs, a_grid, Om_a_grid, w0, wa, x_grid):
    """Compute normalized D(a)/D(1) = exp(-integral from ln(a) to 0 of f(e^x) dx)."""
    f_vals, gamma = _f_from_coeffs(coeffs, a_grid, Om_a_grid, w0, wa)
    # integral from ln(a) to 0 of f(e^x) dx
    # = - integral from 0 to ln(a) of f(e^x) dx
    # For a < 1, ln(a) < 0, so integral from ln(a) to 0 is positive
    # D(a)/D(1) = exp(-integral_{ln a}^{0} f(e^x) dx)
    # = exp(-[F(0) - F(ln a)]) where F' = f
    # = exp(-F(0) + F(ln a))
    # We need F(0) = integral_{-inf}^{0} ... no, just from ln a to 0.
    # Let's compute G(a) = integral_{ln a}^{0} f(e^x) dx
    # Using cumulative integration from a=1 (x=0) backward.
    # G(a) = integral_{x_a}^{0} f(e^x) dx where x_a = ln(a)
    # = - integral_{0}^{x_a} f(e^x) dx
    # For a<1, x_a<0, so this is positive.
    
    # Use trapezoidal rule on the x_grid
    # x_grid is from x_grid[0] (most negative) to ~0
    # f_vals corresponds to a_grid which corresponds to x_grid
    # We want integral from x_grid[i] to 0
    # Let's compute cumulative integral from x=0 downward.
    
    # x_grid[0] is the smallest a (most negative x), x_grid[-1] is closest to 0
    # Let's reverse to go from 0 to negative
    x_rev = x_grid[::-1]  # from ~0 to x_grid[0]
    f_rev = f_vals[::-1]
    
    # Cumulative trapezoidal integral from x=0
    # integral from x_rev[0] to x_rev[i]
    cum_int = np.zeros(len(x_rev))
    for i in range(1, len(x_rev)):
        cum_int[i] = cum_int[i-1] + 0.5 * (f_rev[i-1] + f_rev[i]) * (x_rev[i] - x_rev[i-1])
    
    # G(a) for a corresponding to x_rev[i] = integral from x_rev[i] to x_rev[0]
    # = cum_int[0] - cum_int[i] ... wait
    # cum_int[i] = integral from x_rev[0] to x_rev[i]
    # We want integral from x_rev[i] to 0 = integral from x_rev[i] to x_rev[0]
    # = - integral from x_rev[0] to x_rev[i] = -cum_int[i]
    # But x_rev[0] ~ 0, x_rev[i] < 0, so integral from x_rev[i] to 0 is positive
    # = cum_int[0] - cum_int[i] if cum_int[0]=0 ... 
    # Actually cum_int[0] = 0 (at x_rev[0] ~ 0)
    # cum_int[i] = integral from x_rev[0] to x_rev[i] (negative direction, so negative)
    # integral from x_rev[i] to x_rev[0] = -cum_int[i]
    G_rev = -cum_int  # G at each x_rev point
    
    # D_norm(a) = exp(-G(a))
    D_norm_rev = np.exp(-G_rev)
    
    # Reverse back to original order
    D_norm = D_norm_rev[::-1]
    
    return D_norm


def _objective(coeffs, models, a_grid, x_grid):
    """Compute the max relative error over all models and a values."""
    max_err = 0.0
    max_gamma = 0.0
    min_gamma = 10.0
    
    for (Om0, w0, wa) in models:
        # Compute true f and D
        f_true, D_true = _true_f_and_D(Om0, w0, wa, a_grid)
        
        # Compute Om(a)
        Om_a, E2 = _Om_a(Om0, w0, wa, a_grid)
        
        # Compute model f and D
        f_model, gamma = _f_from_coeffs(coeffs, a_grid, Om_a, w0, wa)
        D_model = _D_norm_from_coeffs(coeffs, a_grid, Om_a, w0, wa, x_grid)
        
        # Track gamma range
        max_gamma = max(max_gamma, np.max(gamma))
        min_gamma = min(min_gamma, np.min(gamma))
        
        # Relative errors
        # f error
        denom_f = np.maximum(np.abs(f_true), 1e-10)
        rel_f = np.abs(f_model - f_true) / denom_f
        max_err = max(max_err, np.max(rel_f))
        
        # D error
        denom_D = np.maximum(np.abs(D_true), 1e-10)
        rel_D = np.abs(D_model - D_true) / denom_D
        max_err = max(max_err, np.max(rel_D))
    
    # Gamma constraint penalty
    gamma_violation = 0.0
    if max_gamma > 1.2:
        gamma_violation = max(gamma_violation, max_gamma - 1.2)
    if min_gamma < 0.1:
        gamma_violation = max(gamma_violation, 0.1 - min_gamma)
    
    # Soft constraint: add penalty
    penalty = 10.0 * gamma_violation**2
    
    return max_err + penalty, max_err


def _clamp_coeffs(coeffs):
    """Clamp coefficients to [-10, 10]."""
    return np.clip(coeffs, -10.0, 10.0)


def construct():
    global _T0, _DEADLINE
    
    start_time = time.time()
    
    # Initial guess: constant gamma
    best_coeffs = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_err, _ = _objective(best_coeffs, _MODELS, _A_GRID, np.log(_A_GRID))
    
    # Nelder-Mead style optimization
    n = 6
    # Initial simplex
    simplex = [best_coeffs.copy()]
    for i in range(n):
        pt = best_coeffs.copy()
        pt[i] += 0.05
        simplex.append(_clamp_coeffs(pt))
    
    # Evaluate simplex
    simplex_errs = []
    for pt in simplex:
        err, _ = _objective(pt, _MODELS, _A_GRID, np.log(_A_GRID))
        simplex_errs.append(err)
    
    simplex = np.array(simplex)
    simplex_errs = np.array(simplex_errs)
    
    # Nelder-Mead iterations
    alpha = 1.0
    gamma_exp = 2.0
    rho = 0.5
    sigma = 0.5
    
    max_iter = 200
    for it in range(max_iter):
        if time.time() - start_time > _DEADLINE - 5:
            break
        
        # Sort by error
        order = np.argsort(simplex_errs)
        simplex = simplex[order]
        simplex_errs = simplex_errs[order]
        
        # Check convergence
        if abs(simplex_errs[0] - simplex_errs[-1]) < 1e-8:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        worst_err = simplex_errs[-1]
        
        # Reflection
        x_r = centroid + alpha * (centroid - worst)
        x_r = _clamp_coeffs(x_r)
        err_r, _ = _objective(x_r, _MODELS, _A_GRID, np.log(_A_GRID))
        
        if err_r < simplex_errs[0]:
            # Expansion
            x_e = centroid + gamma_exp * (centroid - worst)
            x_e = _clamp_coeffs(x_e)
            err_e, _ = _objective(x_e, _MODELS, _A_GRID, np.log(_A_GRID))
            if err_e < err_r:
                simplex[-1] = x_e
                simplex_errs[-1] = err_e
            else:
                simplex[-1] = x_r
                simplex_errs[-1] = err_r
        elif err_r < simplex_errs[-2]:
            simplex[-1] = x_r
            simplex_errs[-1] = err_r
        else:
            # Contraction
            x_c = centroid + rho * (worst - centroid)
            x_c = _clamp_coeffs(x_c)
            err_c, _ = _objective(x_c, _MODELS, _A_GRID, np.log(_A_GRID))
            if err_c < worst_err:
                simplex[-1] = x_c
                simplex_errs[-1] = err_c
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = _clamp_coeffs(simplex[i])
                    simplex_errs[i], _ = _objective(simplex[i], _MODELS, _A_GRID, np.log(_A_GRID))
        
        # Track best
        best_idx = np.argmin(simplex_errs)
        if simplex_errs[best_idx] < best_err:
            best_err = simplex_errs[best_idx]
            best_coeffs = simplex[best_idx].copy()
    
    # Final polish with smaller steps if time permits
    if time.time() - start_time < _DEADLINE - 10:
        # Try a few random perturbations around the best
        for _ in range(50):
            if time.time() - start_time > _DEADLINE - 5:
                break
            perturbation = np.random.normal(0, 0.01, n)
            candidate = _clamp_coeffs(best_coeffs + perturbation)
            err, _ = _objective(candidate, _MODELS, _A_GRID, np.log(_A_GRID))
            if err < best_err:
                best_err = err
                best_coeffs = candidate
    
    return [float(x) for x in best_coeffs]
# EVOLVE-BLOCK-END
