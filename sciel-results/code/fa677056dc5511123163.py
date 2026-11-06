"""Cosmology: compact dark-energy growth approximation.
Uses differential evolution + local refinement to fit six coefficients.
"""
import time
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import differential_evolution, minimize

_START = time.time()
_DEADLINE = 85.0  # safety margin before 90s kill

def _elapsed():
    return time.time() - _START

def _check_deadline():
    if _elapsed() > _DEADLINE:
        raise TimeoutError()

def _om_matter(a, om0):
    """Matter density parameter at scale factor a."""
    return om0 / (a * a * a)

def _om_de(a, om0, w0, wa):
    """Dark energy density parameter at scale factor a for CPL model."""
    # H(a)^2/H0^2 = om0/a^3 + (1-om0)*exp(3*(1+w0+wa)*(1-a))
    # om_de = (1-om0)*exp(3*(1+w0+wa)*(1-a)) / E^2
    e2 = om0 / (a ** 3) + (1.0 - om0) * np.exp(3.0 * (1.0 + w0 + wa) * (1.0 - a))
    de_part = (1.0 - om0) * np.exp(3.0 * (1.0 + w0 + wa) * (1.0 - a))
    return de_part / e2

def _gamma_of_a(a, om0, w0, wa, c):
    """Compute gamma from the six-coefficient ansatz."""
    c0, c1, c2, c3, c4, c5 = c
    om_a = _om_matter(a, om0)
    q = 1.0 - om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
    return gamma

def _f_of_a(a, om0, w0, wa, c):
    """Compute f = Om(a)^gamma."""
    om_a = _om_matter(a, om0)
    gamma = _gamma_of_a(a, om0, w0, wa, c)
    return om_a ** gamma

def _solve_growth_ode(om0, w0, wa, a_min=0.01):
    """Solve the GR growth ODE to get reference f and D(a)/D(1).
    
    The growth equation for D(a):
    D'' + [3/E * (1 + 0.5*Om_de*(1+3w)/Om_m) + 2/a] * D' - 3*Om_m/(2*a*E^2) * D = 0
    
    where E^2 = H^2/H0^2, Om_m = Om0/a^3, Om_de = (1-Om0)*exp(3(1+w0+wa)(1-a))
    
    We solve in terms of ln(a) for stability.
    """
    def ode(x, y):
        a = np.exp(x)
        E2 = om0 / (a**3) + (1.0 - om0) * np.exp(3.0 * (1.0 + w0 + wa) * (1.0 - a))
        om_m = om0 / (a**3)
        om_de = (1.0 - om0) * np.exp(3.0 * (1.0 + w0 + wa) * (1.0 - a))
        w = w0 + wa * (1.0 - a)
        # f = Om_m^gamma_ref where gamma_ref is the "true" growth index from ODE
        # Actually we need to solve for D directly, then f = D'/D (in ln a)
        # D''(x) + A(x)*D'(x) + B(x)*D(x) = 0 where x = ln(a)
        # A(x) = 3/E * (1 + 0.5*Om_de*(1+3w)/Om_m) + 2
        # B(x) = -3*Om_m/(2*E^2)
        E = np.sqrt(E2)
        A = 3.0 / E * (1.0 + 0.5 * om_de * (1.0 + 3.0 * w) / om_m) + 2.0
        B = -3.0 * om_m / (2.0 * E2)
        dydx = np.array([y[1], -A * y[1] + B * y[0]])
        return dydx

    x_min = np.log(a_min)
    x_max = np.log(1.0)
    # Initial conditions at a_min: D ~ a^(6/5) approximately, D' ~ (6/5)*a^(1/5)
    a_init = a_min
    D_init = a_init ** (6.0 / 5.0)
    Dp_init = (6.0 / 5.0) * a_init ** (1.0 / 5.0)
    
    sol = solve_ivp(ode, [x_min, x_max], [D_init, Dp_init], 
                    method='DOP853', rtol=1e-10, atol=1e-12,
                    dense_output=True, max_step=0.01)
    
    if not sol.success:
        # Fallback: use power-law approximation
        a_grid = np.exp(np.linspace(x_min, x_max, 200))
        D_ref = (a_grid / 1.0) ** 1.1  # approximate
        f_ref = 1.1 * (a_grid ** 0.2)  # approximate
        return a_grid, D_ref, f_ref
    
    return sol

def _get_reference(om0, w0, wa, a_grid):
    """Get reference f and D(a)/D(1) on a_grid for a given cosmology."""
    sol = _solve_growth_ode(om0, w0, wa)
    
    if isinstance(sol, tuple):
        # Fallback was used
        a_fb, D_fb, f_fb = sol
        # Interpolate to our grid
        D_ref = np.interp(a_grid, a_fb, D_fb)
        f_ref = np.interp(a_grid, a_fb, f_fb)
    else:
        x_grid = np.log(a_grid)
        y = sol.sol(x_grid)
        D_raw = y[0]
        Dp_raw = y[1]
        # Normalize so D(1) = 1
        D_ref = D_raw / D_raw[-1]
        # f = dD/dln(a) / D = D'/D (where D' is dD/dln(a))
        f_ref = Dp_raw / D_raw
    
    # Ensure positivity
    D_ref = np.maximum(D_ref, 1e-10)
    f_ref = np.maximum(f_ref, 1e-10)
    
    return f_ref, D_ref

def _build_validation_set():
    """Build a diverse set of validation cosmologies and a-grid."""
    # Various matter densities
    om0_vals = [0.2, 0.25, 0.27, 0.3, 0.35, 0.4]
    # Various CPL parameters
    w0_vals = [-1.0, -0.9, -0.8, -1.1]
    wa_vals = [0.0, 0.3, 0.5, -0.3, -0.5, 1.0]
    
    # Create a subset to keep computation manageable
    # Use a structured grid
    cosmologies = []
    for om0 in om0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                # Ensure physical: Om_de(1) = 1-Om0 >= 0
                if 0 < om0 < 1:
                    cosmologies.append((om0, w0, wa))
    
    # Sample a subset if too many
    if len(cosmologies) > 30:
        np.random.seed(42)
        idx = np.random.choice(len(cosmologies), 30, replace=False)
        cosmologies = [cosmologies[i] for i in idx]
    
    # Add some specific well-known cases
    cosmologies.append((0.27, -1.0, 0.0))   # LambdaCDM
    cosmologies.append((0.3, -0.9, 0.3))    # Quintessence-like
    cosmologies.append((0.25, -1.1, 0.5))   # Phantom-like
    
    # a-grid: logarithmically spaced
    a_grid = np.logspace(np.log10(0.05), 0.0, 50)
    a_grid = np.unique(a_grid)  # ensure sorted unique
    
    return cosmologies, a_grid

def _compute_errors(c, cosmologies, a_grid, f_refs, D_refs):
    """Compute relative errors for f and D across all validation points."""
    max_err_f = 0.0
    max_err_D = 0.0
    
    for i, (om0, w0, wa) in enumerate(cosmologies):
        f_ref = f_refs[i]
        D_ref = D_refs[i]
        
        for j, a in enumerate(a_grid):
            # Compute model f
            om_a = om0 / (a ** 3)
            if om_a <= 0:
                continue
            gamma = _gamma_of_a(a, om0, w0, wa, c)
            f_model = om_a ** gamma
            
            # Relative error in f
            if abs(f_ref[j]) > 1e-10:
                err_f = abs(f_model - f_ref[j]) / abs(f_ref[j])
                if err_f > max_err_f:
                    max_err_f = err_f
            
            # Compute model D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)
            # Use numerical integration
            def integrand(x):
                aa = np.exp(x)
                om_aa = om0 / (aa ** 3)
                if om_aa <= 0:
                    return 0.0
                gam = _gamma_of_a(aa, om0, w0, wa, c)
                return om_aa ** gam
            
            try:
                integral, _ = quad(integrand, np.log(a), 0.0, limit=100)
                D_model = np.exp(-integral)
            except Exception:
                D_model = D_ref[j]  # fallback
            
            if abs(D_ref[j]) > 1e-10:
                err_D = abs(D_model - D_ref[j]) / abs(D_ref[j])
                if err_D > max_err_D:
                    max_err_D = err_D
    
    return max_err_f, max_err_D

def _objective_true(c, cosmologies, a_grid, f_refs, D_refs):
    """True objective: 100 * max relative error across f and D."""
    max_err_f, max_err_D = _compute_errors(c, cosmologies, a_grid, f_refs, D_refs)
    return 100.0 * max(max_err_f, max_err_D)

def _objective_smooth(c, cosmologies, a_grid, f_refs, D_refs, beta=50.0):
    """Smooth surrogate using log-sum-exp for the max."""
    errs = []
    
    for i, (om0, w0, wa) in enumerate(cosmologies):
        f_ref = f_refs[i]
        D_ref = D_refs[i]
        
        for j, a in enumerate(a_grid):
            om_a = om0 / (a ** 3)
            if om_a <= 0:
                continue
            gamma = _gamma_of_a(a, om0, w0, wa, c)
            f_model = om_a ** gamma
            
            if abs(f_ref[j]) > 1e-10:
                err_f = abs(f_model - f_ref[j]) / abs(f_ref[j])
                errs.append(err_f)
            
            # For D, use a fast approximation during optimization
            # D_model ≈ D_ref * exp(-integral of (f_model/f_ref - 1) * f_ref dx)
            # This is an approximation; we'll use exact in polish
            # For speed, approximate D error as proportional to f error integrated
            if abs(f_ref[j]) > 1e-10:
                err_D_approx = err_f * 0.5  # rough scaling
                errs.append(err_D_approx)
    
    if not errs:
        return 1e6
    
    errs = np.array(errs)
    # Log-sum-exp for soft max
    m = np.max(errs)
    if m < 1e-15:
        return 0.0
    lse = m + np.log(np.sum(np.exp((errs - m) * beta)) / beta)
    return 100.0 * lse

def construct():
    """Construct the six coefficients using differential evolution + local polish."""
    _check_deadline()
    
    # Build validation set
    cosmologies, a_grid = _build_validation_set()
    n_cosmo = len(cosmologies)
    
    # Pre-compute references
    f_refs = []
    D_refs = []
    for i, (om0, w0, wa) in enumerate(cosmologies):
        _check_deadline()
        f_ref, D_ref = _get_reference(om0, w0, wa, a_grid)
        f_refs.append(f_ref)
        D_refs.append(D_ref)
    
    _check_deadline()
    
    # Bounds for coefficients
    bounds = [(-10, 10)] * 6
    
    # Initial best: constant gamma = 0.55
    c_best = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Evaluate initial
    score_init = _objective_true(c_best, cosmologies, a_grid, f_refs, D_refs)
    
    # Phase 1: Differential Evolution with smooth objective
    if _elapsed() < _DEADLINE - 30:
        try:
            de_result = differential_evolution(
                _objective_smooth,
                bounds,
                args=(cosmologies, a_grid, f_refs, D_refs, 50.0),
                maxiter=50,
                popsize=15,
                tol=1e-4,
                seed=42,
                polish=False,
                updating='immediate',
                callback=None
            )
            if de_result.success and _elapsed() < _DEADLINE - 10:
                c_de = de_result.x
                score_de = _objective_true(c_de, cosmologies, a_grid, f_refs, D_refs)
                if score_de < score_init:
                    c_best = c_de
                    score_init = score_de
        except Exception:
            pass
    
    _check_deadline()
    
    # Phase 2: Local refinement with true objective
    if _elapsed() < _DEADLINE - 5:
        try:
            res = minimize(
                _objective_true,
                c_best,
                args=(cosmologies, a_grid, f_refs, D_refs),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.success:
                c_polished = res.x
                score_polished = _objective_true(c_polished, cosmologies, a_grid, f_refs, D_refs)
                if score_polished < score_init:
                    c_best = c_polished
                    score_init = score_polished
        except Exception:
            pass
    
    _check_deadline()
    
    # Phase 3: Additional local polish with tighter tolerance
    if _elapsed() < _DEADLINE - 2:
        try:
            res2 = minimize(
                _objective_true,
                c_best,
                args=(cosmologies, a_grid, f_refs, D_refs),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res2.success:
                c_final = res2.x
                score_final = _objective_true(c_final, cosmologies, a_grid, f_refs, D_refs)
                if score_final < score_init:
                    c_best = c_final
                    score_init = score_final
        except Exception:
            pass
    
    # Ensure coefficients are in [-10, 10] and finite
    c_best = np.clip(c_best, -10, 10)
    if not np.all(np.isfinite(c_best)):
        c_best = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Verify gamma constraint: gamma in [0.1, 1.2] at all checked locations
    # If violated, fall back to constant gamma
    gamma_ok = True
    for om0, w0, wa in cosmologies:
        for a in a_grid:
            g = _gamma_of_a(a, om0, w0, wa, c_best)
            if g < 0.1 or g > 1.2:
                gamma_ok = False
                break
        if not gamma_ok:
            break
    
    if not gamma_ok:
        # Try to fix by scaling down non-constant terms
        for scale in [0.5, 0.3, 0.1, 0.0]:
            c_try = np.array([c_best[0] + scale * (c_best[1] - c_best[1]),
                              scale * c_best[1], scale * c_best[2],
                              scale * c_best[3], scale * c_best[4],
                              scale * c_best[5]])
            # Actually just scale the non-constant parts
            c_try = np.array([c_best[0], scale*c_best[1], scale*c_best[2],
                              scale*c_best[3], scale*c_best[4], scale*c_best[5]])
            gamma_ok = True
            for om0, w0, wa in cosmologies:
                for a in a_grid:
                    g = _gamma_of_a(a, om0, w0, wa, c_try)
                    if g < 0.1 or g > 1.2:
                        gamma_ok = False
                        break
                if not gamma_ok:
                    break
            if gamma_ok:
                c_best = c_try
                break
    
    if not gamma_ok:
        c_best = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    return [float(x) for x in c_best]
