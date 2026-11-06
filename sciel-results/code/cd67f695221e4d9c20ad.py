"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution
import time

_DEADLINE = 85.0  # safety margin before 90s kill
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _check_deadline():
    return _elapsed() > _DEADLINE

def _Om_a(Om0, w0, wa, a):
    """Matter density parameter at scale factor a for CPL dark energy."""
    # Dark energy density: rho_de(a) = rho_de0 * a^(-3(1+w0+wa)) * a^(3*wa)
    # = a^(-3-3w0-3wa+3wa) = a^(-3(1+w0)) ... wait, let me be careful.
    # For CPL: w(a) = w0 + wa*(1-a)
    # rho_de(a) = rho_de0 * exp(-3 * integral_1^a (1+w(a'))/a' da')
    # integral_1^a (1+w0+wa(1-a'))/a' da' = integral_1^a [(1+w0)/a' + wa(1-a')/a'] da'
    # = (1+w0)*ln(a) + wa * integral_1^a (1/a' - 1) da'
    # = (1+w0)*ln(a) + wa*(ln(a) - (a-1))
    # So rho_de(a)/rho_de0 = exp(-3[(1+w0+wa)*ln(a) - wa*(a-1)])
    # = a^(-3(1+w0+wa)) * exp(3*wa*(a-1))
    # E(a)^2 = Om0*a^-3 + (1-Om0)*a^(-3(1+w0+wa))*exp(3*wa*(a-1))
    # Om(a) = Om0*a^-3 / E(a)^2
    if wa == 0:
        rho_de_ratio = a ** (-3*(1+w0))
    else:
        rho_de_ratio = a ** (-3*(1+w0+wa)) * np.exp(3*wa*(a-1))
    rho_m_ratio = a ** (-3)
    E2 = Om0 * rho_m_ratio + (1 - Om0) * rho_de_ratio
    return Om0 * rho_m_ratio / E2

def _compute_Om_grid(Om0, w0, wa, a_vals):
    """Vectorized Om(a) computation."""
    a = np.asarray(a_vals)
    if wa == 0:
        rho_de_ratio = a ** (-3*(1+w0))
    else:
        rho_de_ratio = a ** (-3*(1+w0+wa)) * np.exp(3*wa*(a-1))
    rho_m_ratio = a ** (-3)
    E2 = Om0 * rho_m_ratio + (1 - Om0) * rho_de_ratio
    return Om0 * rho_m_ratio / E2

def _solve_growth_ode(Om0, w0, wa, a_grid, D_at_1=1.0, f_at_1=None):
    """Solve the growth ODE: D'' + 2H D' - 4piG rho_m D = 0
    In terms of x=ln(a): d2D/dx2 + (3/2 + (1+w(a))/2) dD/dx - (3/2) Om(a) D = 0
    Actually the standard form is:
    D'' + (3/2 + (1+w)/2) D' - (3/2) Om(a) D = 0  where ' = d/d ln(a)
    
    We want f = D'/D and D/D(1).
    """
    # Use x = ln(a), a = exp(x)
    # x ranges from ln(a_min) to 0 (a=1)
    a_min = a_grid[0]
    x_min = np.log(a_min)
    x_max = 0.0  # a=1
    
    def ode(x, y):
        # y = [D, D']
        a = np.exp(x)
        Om_a = _Om_a(Om0, w0, wa, a)
        w_a = w0 + wa*(1-a)
        # Coefficient of D': 3/2 + (1+w_a)/2
        c1 = 1.5 + 0.5*(1+w_a)
        # Coefficient of D: -3/2 * Om_a
        c0 = -1.5 * Om_a
        return [y[1], -c1*y[1] + c0*y[0]]
    
    # We need D(1) and D'(1). Use the known asymptotic at a=1 (late times):
    # f(1) ≈ (10 Om0/27)^(4/7) * (1 - 0.5 wa) ... or use the standard approximation
    # Better: solve backward from a=1 with initial conditions.
    # At a=1: D(1)=1, f(1) = D'(1)/D(1) = f(1)
    # Standard approximation for f(1): (10*Om0/27)**(4/7)
    f1 = (10.0*Om0/27.0)**(4.0/7.0)
    # D'(1) = f1 * D(1) = f1
    y0 = [1.0, f1]
    
    # Solve from x=0 backward to x=x_min
    sol = solve_ivp(ode, [x_max, x_min], y0, method='DOP853', 
                    rtol=1e-10, atol=1e-12, dense_output=True)
    
    if not sol.success:
        return None, None
    
    # Evaluate on the grid
    x_eval = np.log(a_grid)
    # solve_ivp with decreasing x interval: sol.t goes from x_max to x_min
    # We need to evaluate at x_eval which are <= 0
    # Use dense output
    Y = sol.sol(x_eval)
    D_vals = Y[0]
    Dp_vals = Y[1]
    f_vals = Dp_vals / D_vals
    
    # Normalize D so D(1)=1
    D_norm = D_vals / D_vals[-1]  # D at a=1 (last point)
    
    return f_vals, D_norm

def _gamma_from_coeffs(c, q, u, wa, a):
    c0, c1, c2, c3, c4, c5 = c
    return c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q

def _objective_smooth(c, a_grid, Om0s, w0s, was, f_refs, D_refs, logit_scale=50.0):
    """Smooth surrogate for max relative error using log-sum-exp."""
    max_err = 0.0
    for i, (Om0, w0, wa) in enumerate(zip(Om0s, w0s, was)):
        a = a_grid
        Om_a = _compute_Om_grid(Om0, w0, wa, a)
        q = 1 - Om_a
        u = 1 + w0 + wa*(1-a)
        gamma = _gamma_from_coeffs(c, q, u, wa, a)
        
        # Check gamma bounds
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e6
        
        f_approx = Om_a ** gamma
        D_approx = np.exp(-np.cumsum(f_approx)[::-1])  # This is wrong, need proper integration
        
        # Actually, D(a)/D(1) = exp(-integral_ln(a)^0 f(exp(x)) dx)
        # where x = ln(a). So D(a)/D(1) = exp(-int_{ln(a)}^0 f(e^x) dx)
        # We need to compute this integral properly.
        
        # For the smooth objective, let's just use f error and approximate D error
        # Compute D more carefully using trapezoidal rule on the grid
        x = np.log(a)
        # f as function of x: f(x) = f_approx
        # D(a)/D(1) = exp(-int_x^0 f(x') dx')
        # int_x^0 f(x') dx' = -int_0^x f(x') dx'
        # So D(a)/D(1) = exp(int_0^x f(x') dx')
        # But we need to be careful with direction.
        
        # Let's compute cumulative integral from x to 0
        # int_x^0 f(x') dx' 
        # On our grid, x goes from x_min (most negative) to 0
        # int_{x_i}^0 f(x') dx' = -int_0^{x_i} f(x') dx'
        
        # Use cumulative trapezoidal integration
        from scipy.integrate import cumulative_trapezoid
        # f as function of x, we want int from x to 0
        # int_x^0 f = -int_0^x f
        # int_0^x f for x<0: this is negative of int_x^0 f
        
        # Let's compute int_0^{x_i} f(x') dx' for each i
        # Since x_i < 0, int_0^{x_i} = -int_{x_i}^0
        # cumulative_trapezoid(f, x, initial=0) gives int_{x_0}^{x_i} f dx'
        # We want int_{x_i}^0 = int_{x_0}^0 - int_{x_0}^{x_i}
        
        int_x0 = cumulative_trapezoid(f_approx, x, initial=0.0)
        # int_x0[i] = int_{x[0]}^{x[i]} f dx'
        # int_{x[i]}^0 = int_{x[0]}^0 - int_{x[0]}^{x[i]}
        # int_{x[0]}^0 = int_x0[-1]
        int_to_0 = int_x0[-1] - int_x0
        # D(a)/D(1) = exp(-int_{x[i]}^0 f dx') = exp(-int_to_0)
        # Wait: D(a)/D(1) = exp(-integral_ln(a)^0 f(exp(x)) dx)
        # ln(a) = x, so integral from x to 0 of f(e^x) dx = int_{x}^0 f(x') dx'
        # So D/D(1) = exp(-int_{x}^0 f(x') dx') = exp(-int_to_0)
        
        D_approx = np.exp(-int_to_0)
        
        # Relative errors
        f_err = np.abs(f_approx - f_refs[i]) / np.abs(f_refs[i])
        D_err = np.abs(D_approx - D_refs[i]) / np.abs(D_refs[i])
        
        all_err = np.concatenate([f_err, D_err])
        all_err = np.clip(all_err, 1e-15, None)
        
        # Soft max via log-sum-exp
        max_e = np.max(all_err)
        if max_e > 0:
            soft_max = max_e + np.log(np.sum(np.exp(all_err - max_e))) / len(all_err)
        else:
            soft_max = 0.0
        
        if soft_max > max_err:
            max_err = soft_max
    
    return max_err

def _objective_exact(c, a_grid, Om0s, w0s, was, f_refs, D_refs):
    """Exact max relative error."""
    max_err = 0.0
    for i, (Om0, w0, wa) in enumerate(zip(Om0s, w0s, was)):
        a = a_grid
        Om_a = _compute_Om_grid(Om0, w0, wa, a)
        q = 1 - Om_a
        u = 1 + w0 + wa*(1-a)
        gamma = _gamma_from_coeffs(c, q, u, wa, a)
        
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e6
        
        f_approx = Om_a ** gamma
        
        from scipy.integrate import cumulative_trapezoid
        x = np.log(a)
        int_x0 = cumulative_trapezoid(f_approx, x, initial=0.0)
        int_to_0 = int_x0[-1] - int_x0
        D_approx = np.exp(-int_to_0)
        
        f_err = np.max(np.abs(f_approx - f_refs[i]) / np.abs(f_refs[i]))
        D_err = np.max(np.abs(D_approx - D_refs[i]) / np.abs(D_refs[i]))
        
        if f_err > max_err:
            max_err = f_err
        if D_err > max_err:
            max_err = D_err
    
    return max_err

# EVOLVE-BLOCK-END
def construct():
    global _start_time
    _start_time = time.time()
    
    # Define a representative set of cosmologies for optimization
    # The evaluator uses "disjoint public validation cosmologies"
    # We'll use a broad grid to ensure robustness
    Om0_vals = [0.25, 0.30, 0.35]
    w0_vals = [-1.0, -0.9, -0.8]
    wa_vals = [0.0, 0.3, 0.5]
    
    # Scale factor grid
    a_grid = np.logspace(np.log10(0.01), 0, 200)
    
    # Pre-compute references
    f_refs = []
    D_refs = []
    valid_models = []
    
    for Om0 in Om0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                if _check_deadline():
                    break
                f_ref, D_ref = _solve_growth_ode(Om0, w0, wa, a_grid)
                if f_ref is not None:
                    f_refs.append(f_ref)
                    D_refs.append(D_ref)
                    valid_models.append((Om0, w0, wa))
    
    if not valid_models:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    Om0s = [m[0] for m in valid_models]
    w0s = [m[1] for m in valid_models]
    was = [m[2] for m in valid_models]
    
    # Start from the seed
    c0_init = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Try to improve using differential evolution (global) then polish with BFGS
    best_c = c0_init
    best_score = _objective_exact(c0_init, a_grid, Om0s, w0s, was, f_refs, D_refs)
    
    if not _check_deadline():
        # Global search with differential evolution
        bounds = [(-10, 10)] * 6
        de_result = differential_evolution(
            _objective_smooth, bounds, 
            args=(a_grid, Om0s, w0s, was, f_refs, D_refs, 50.0),
            maxiter=50, tol=1e-6, seed=42, polish=False,
            updating='immediate'
        )
        if de_result.fun < best_score:
            best_c = de_result.x
            best_score = de_result.fun
    
    if not _check_deadline():
        # Polish with BFGS
        def obj_for_minimize(c):
            return _objective_smooth(c, a_grid, Om0s, w0s, was, f_refs, D_refs, 100.0)
        
        res = minimize(obj_for_minimize, best_c, method='BFGS', 
                       options={'maxiter': 200, 'gtol': 1e-10})
        if res.fun < best_score:
            best_c = res.x
            best_score = res.fun
    
    # Final exact evaluation
    final_score = _objective_exact(best_c, a_grid, Om0s, w0s, was, f_refs, D_refs)
    if final_score < best_score:
        best_score = final_score
    
    # Ensure coefficients are in [-10, 10]
    best_c = np.clip(best_c, -10, 10)
    
    return list(best_c)
# EVOLVE-BLOCK-END
