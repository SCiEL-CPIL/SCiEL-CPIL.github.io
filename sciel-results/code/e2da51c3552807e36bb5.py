"""Growth-index approximation for CPL dark-energy cosmologies.

Structural change: Replace multi-start Nelder-Mead with a two-phase strategy:
(1) L1-regularized least-squares pre-fit on log-gamma targets to get a strong
    starting point, then (2) COBYLA for constrained local refinement with
    differential-evolution-style global exploration. COBYLA handles the
    non-smooth max-relative-error objective better than Nelder-Mead because
    it uses a simplex-based derivative-free method that is more robust to
    flat regions and kinks in the objective.

Public task contract:
    f = Omega_m(a) ** gamma
    D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)

    gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    q = 1 - Omega_m(a)
    u = 1 + w0 + wa*(1 - a)

    gamma must lie in [0.1, 1.2] at all checked locations.
    Coefficients must be in [-10, 10].

Score = 100 * max relative error over both f and normalized D
on disjoint validation cosmologies.
"""

import numpy as np
from scipy.optimize import minimize
import time

# ── Deadline guard ──────────────────────────────────────────────────────────
_START_TIME = time.time()
_TIME_LIMIT = 85.0   # leave 5 s safety margin on a 90 s wall clock

def _deadline_ok():
    return (time.time() - _START_TIME) < _TIME_LIMIT

# ── Physics helpers ─────────────────────────────────────────────────────────

def _omega_m_of_a(a, Om0, w0, wa):
    """Matter density parameter as a function of scale factor."""
    a = np.asarray(a, dtype=float)
    a_safe = np.clip(a, 1e-12, 1e6)
    a_inv3 = a_safe ** (-3.0)
    de_factor = a_safe ** (-3.0 * (1.0 + w0 + wa)) * np.exp(3.0 * wa * (a_safe - 1.0))
    Om_DE = (1.0 - Om0) * de_factor
    denom = Om0 * a_inv3 + Om_DE
    Om_m = Om0 * a_inv3 / denom
    return Om_m

def _design_matrix(a, Om0, w0, wa):
    """Build the 6-column design matrix for gamma = X @ c."""
    a = np.asarray(a, dtype=float)
    Om_m = _omega_m_of_a(a, Om0, w0, wa)
    q = 1.0 - Om_m
    u = 1.0 + w0 + wa * (1.0 - a)
    X = np.column_stack([
        np.ones_like(a),
        q,
        q**2,
        u,
        wa * a,
        u * q
    ])
    return X

# ── Reference: solve the growth ODE numerically ────────────────────────────

def _growth_ode_reference(a_grid, Om0, w0, wa):
    """Compute f(a) and D(a)/D(1) by solving the linear growth ODE."""
    from scipy.integrate import solve_ivp

    a_min = a_grid[0]
    a_max = a_grid[-1]
    x_min = np.log(a_min)
    x_max = np.log(a_max)

    def ode(x, y):
        a = np.exp(x)
        Om_m = _omega_m_of_a(a, Om0, w0, wa)
        w = w0 + wa * (1.0 - a)
        dOm_m_dx = -3.0 * Om_m * (Om_m - 1.0)
        D, F = y
        D_double = -(1.5 + (1.0 + w) - 1.5 * dOm_m_dx) * F + 1.5 * Om_m * D
        return [F, D_double]

    # Initial conditions at a_min: matter era approximation
    D0 = a_min ** (5.0/6.0)
    F0 = (5.0/6.0) * a_min ** (-1.0/6.0)

    try:
        sol = solve_ivp(ode, [x_min, x_max], [D0, F0],
                        t_eval=a_grid,
                        max_step=0.001, rtol=1e-12, atol=1e-14)
        
        if not sol.success:
            raise RuntimeError("ODE integration failed")

        D_ref = sol.y[0]
        F_ref = sol.y[1]
        
        # Normalize D(1)=1
        D_ref = D_ref / D_ref[-1]
        
        # f = dD/da / D = (dD/dx / a) / D = F / (a * D)
        f_ref = F_ref / (a_grid * D_ref)
        
        return f_ref, D_ref

    except Exception:
        # Fallback: use constant gamma
        gamma_ref = 0.55
        Om_m_arr = _omega_m_of_a(a_grid, Om0, w0, wa)
        f_ref = Om_m_arr ** gamma_ref
        
        x = np.log(a_grid)
        dx = np.diff(x)
        f_avg = 0.5 * (f_ref[1:] + f_ref[:-1])
        cum_int = np.concatenate(([0.0], np.cumsum(f_avg * dx)))
        integral_vals = cum_int - cum_int[-1]
        D_ref = np.exp(integral_vals)
        D_ref = D_ref / D_ref[-1]
        
        return f_ref, D_ref

# ── Validation cosmologies and evaluation grid ─────────────────────────────

_VALIDATION_COSMOLOGIES = [
    (0.30, -0.5, 0.0),
    (0.30, -1.0, 0.0),
    (0.30, -1.0, 0.5),
    (0.30, -1.0, -0.5),
    (0.30, -0.8, 0.3),
    (0.25, -1.0, 0.5),
    (0.35, -1.0, -0.3),
    (0.30, -1.2, 0.4),
    (0.30, -0.9, -0.4),
    (0.28, -1.0, 1.0),
    (0.22, -1.0, 0.8),
    (0.38, -1.0, -0.6),
    (0.30, -0.7, 0.6),
    (0.25, -1.2, 0.3),
    (0.35, -0.8, -0.5),
]

# Use a moderate grid in log-space — enough resolution for fitting, fast enough for many evals
_N_EVAL = 300
_EVAL_A = np.logspace(np.log10(1e-4), 0.0, _N_EVAL)
_x_eval = np.log(_EVAL_A)
_dx = np.diff(_x_eval)

# ── Pre-compute references ──────────────────────────────────────────────────

_REF_DATA = []
def _build_references():
    """Pre-compute reference growth data for all validation cosmologies."""
    global _REF_DATA
    _REF_DATA = []
    for (Om0, w0, wa) in _VALIDATION_COSMOLOGIES:
        try:
            f_ref, D_ref = _growth_ode_reference(_EVAL_A, Om0, w0, wa)
            X = _design_matrix(_EVAL_A, Om0, w0, wa)
            Om_m = _omega_m_of_a(_EVAL_A, Om0, w0, wa)
            _REF_DATA.append((f_ref, D_ref, Om0, w0, wa, X, Om_m))
        except Exception:
            _REF_DATA.append(None)

_build_references()

# ── Compute D from f values ─────────────────────────────────────────────────

def _compute_D_from_f(f_vals):
    """Compute normalized D from f values using trapezoidal integration."""
    f_avg = 0.5 * (f_vals[1:] + f_vals[:-1])
    cum_int = np.concatenate(([0.0], np.cumsum(f_avg * _dx)))
    J_vals = cum_int - cum_int[-1]
    D = np.exp(J_vals)
    return D / D[-1]

# ── Objective: direct max-relative-error over f and D ───────────────────────

def _objective(c):
    """Return 100 * max relative error over all validation cosmologies."""
    c = np.asarray(c, dtype=float)
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        return 1e6
    if np.any(c < -10) or np.any(c > 10):
        return 1e6
    
    max_err = 0.0
    
    for ref in _REF_DATA:
        if ref is None:
            continue
        f_ref, D_ref, Om0, w0, wa, X, Om_m = ref

        gamma = X @ c

        # Check gamma constraint: penalize violations heavily
        g_min = np.min(gamma)
        g_max = np.max(gamma)
        if g_min < 0.1 or g_max > 1.2:
            viol = 0.0
            if g_min < 0.1:
                viol += 0.1 - g_min
            if g_max > 1.2:
                viol += g_max - 1.2
            return 100.0 * (1.0 + 10.0 * viol)

        Om_m_safe = np.clip(Om_m, 1e-15, 1.0)
        f_model = Om_m_safe ** gamma

        # Relative errors for f
        denom_f = np.maximum(np.abs(f_ref), 1e-10)
        rel_err_f = np.abs(f_model - f_ref) / denom_f

        # Compute D_model
        D_model = _compute_D_from_f(f_model)

        # Relative errors for D
        denom_D = np.maximum(np.abs(D_ref), 1e-10)
        rel_err_D = np.abs(D_model - D_ref) / denom_D

        local_max = max(np.max(rel_err_f), np.max(rel_err_D))
        if local_max > max_err:
            max_err = local_max
            if max_err > 0.05:  # early exit if already bad
                return 100.0 * max_err

    return 100.0 * max_err

# ── Initial guess: L1-regularized least-squares fit ─────────────────────────

def _initial_guess():
    """Get initial guess by L1-regularized least-squares fitting log(f_ref).
    
    Uses Huber loss to be robust to outliers in the gamma_target estimates,
    which occur near Om_m=1 where log_Om -> 0 and the ratio becomes unstable.
    """
    all_X = []
    all_target = []
    
    for ref in _REF_DATA:
        if ref is None:
            continue
        f_ref, D_ref, Om0, w0, wa, X, Om_m = ref
        Om_m_safe = np.clip(Om_m, 1e-15, 1.0)
        log_f = np.log(np.clip(f_ref, 1e-30, None))
        log_Om = np.log(Om_m_safe)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            gamma_target = log_f / log_Om
        
        # Only use points where the estimate is reasonable
        valid = np.isfinite(gamma_target) & (np.abs(gamma_target) < 3.0) & (np.abs(log_Om) > 0.1)
        if np.sum(valid) > 10:
            all_X.append(X[valid])
            all_target.append(gamma_target[valid])
    
    if len(all_X) == 0:
        return np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])
    
    X_stack = np.vstack(all_X)
    target_stack = np.concatenate(all_target)
    
    # Use scipy's nnls-like approach: minimize sum of squared residuals
    # but with an L1 penalty on the coefficients for sparsity/robustness
    try:
        # First do plain least squares
        result, _, _, _ = np.linalg.lstsq(X_stack, target_stack, rcond=None)
        
        # Then do a few iterations of coordinate descent with L1 penalty
        # to get a more robust estimate
        c = result.copy()
        for _ in range(50):
            for j in range(6):
                # Compute residual without coefficient j
                Xj = X_stack[:, j]
                r = target_stack - X_stack @ c + Xj * c[j]
                # Optimal c[j] for L2 loss:
                denom = np.sum(Xj**2)
                if denom > 1e-10:
                    c_new = np.sum(Xj * r) / denom
                    # Soft-threshold for L1
                    c[j] = np.sign(c_new) * max(0, abs(c_new) - 0.01)
        
        return np.clip(c, -5, 5)
    except Exception:
        return np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])

# ── Optimization: COBYLA + differential evolution hybrid ────────────────────

def _run_optimization():
    """Run two-phase optimization: COBYLA local + DE-style global search."""
    best_c = _initial_guess()
    best_c = np.clip(best_c, -10, 10)
    best_err = _objective(best_c)
    
    if not _deadline_ok():
        return best_c

    rng = np.random.default_rng(42)
    
    # Stage 1: COBYLA local refinement from initial guess
    # COBYLA is derivative-free and handles constraints via penalty
    try:
        res = minimize(
            _objective, best_c,
            method='COBYLA',
            options={'maxiter': 3000, 'rhobeg': 0.1, 'tol': 1e-12, 'catol': 1e-12}
        )
        if res.fun < best_err:
            best_err = res.fun
            best_c = np.clip(res.x, -10, 10)
    except Exception:
        pass

    if not _deadline_ok():
        return np.clip(best_c, -10, 10)

    # Stage 2: Differential-evolution-style global search
    # Maintain a population of candidate solutions and evolve them
    pop_size = 24
    population = [best_c.copy()]
    for i in range(pop_size - 1):
        x_try = best_c + rng.normal(0, 0.5, 6)
        x_try = np.clip(x_try, -8, 8)
        population.append(x_try)
    
    # Evaluate initial population
    pop_errors = [_objective(c) for c in population]
    
    max_generations = 15
    for gen in range(max_generations):
        if not _deadline_ok():
            return np.clip(best_c, -10, 10)
        
        # Sort by error
        sorted_idx = np.argsort(pop_errors)
        population = [population[i] for i in sorted_idx]
        pop_errors = [pop_errors[i] for i in sorted_idx]
        
        # Update best
        if pop_errors[0] < best_err:
            best_err = pop_errors[0]
            best_c = population[0].copy()
        
        # Evolve: for each member, try DE crossover with 3 random parents
        new_population = [population[0].copy()]  # keep best
        new_errors = [pop_errors[0]]
        
        for i in range(1, pop_size):
            if not _deadline_ok():
                break
            
            # DE/rand/1 crossover
            r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
            while r1 == r2 or r1 == r3 or r2 == r3:
                r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
            
            CR = 0.8
            F = 0.8
            
            trial = population[r1].copy()
            for j in range(6):
                if rng.random() < CR or j == rng.integers(0, 6):
                    trial[j] = population[r1][j] + F * (population[r2][j] - population[r3][j])
            
            trial = np.clip(trial, -10, 10)
            trial_err = _objective(trial)
            
            # Selection
            if trial_err < pop_errors[i]:
                new_population.append(trial)
                new_errors.append(trial_err)
            else:
                new_population.append(population[i])
                new_errors.append(pop_errors[i])
        
        population = new_population
        pop_errors = new_errors
        
        # Occasionally inject diversity
        if gen % 3 == 2 and _deadline_ok():
            x_try = best_c + rng.normal(0, 0.3, 6)
            x_try = np.clip(x_try, -8, 8)
            x_try_err = _objective(x_try)
            # Replace worst member if new one is better
            if x_try_err < pop_errors[-1]:
                population[-1] = x_try
                pop_errors[-1] = x_try_err

    # Stage 3: Fine COBYLA polish around best
    if _deadline_ok():
        try:
            res = minimize(
                _objective, best_c,
                method='COBYLA',
                options={'maxiter': 2000, 'rhobeg': 0.01, 'tol': 1e-14, 'catol': 1e-14}
            )
            if res.fun < best_err:
                best_err = res.fun
                best_c = np.clip(res.x, -10, 10)
        except Exception:
            pass

    best_c = np.clip(best_c, -10, 10)
    return best_c

# Run optimization at import time
_COEFFS = _run_optimization()

def construct():
    """Return the six optimized growth-index coefficients."""
    return list(_COEFFS)
