"""Improved growth-index ansatz with runtime optimization."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
import time

# Constants for the optimization
DEADLINE = 80.0  # seconds, leaving 10s safety margin

def _Om_of_a(a, Om0, wa):
    """Matter density parameter at scale factor a for CPL dark energy."""
    # E^2(a) = Om0 * a^-3 + (1-Om0) * a^(-3*(1+w0+wa*(1-a)))
    # We need w0, but the task says u = 1+w0+wa*(1-a)
    # Actually, for CPL: w(a) = w0 + wa*(1-a)
    # The integral for E(a)^2 involves exp(-3 * integral w(a) da / a)
    # Let's use the standard approximation or exact solution
    # E(a)^2 = Om_m * a^-3 + Om_de * exp(-3 * ∫ w(a) da / a)
    # ∫ w(a) da / a = ∫ (w0 + wa*(1-a)) da / a = w0*ln(a) + wa*(a - ln(a))
    # So Om_de(a) = Om_de0 * exp(-3*(w0*ln(a) + wa*(a - ln(a))))
    # = Om_de0 * a^(-3*w0) * exp(-3*wa*(a - ln(a)))
    # This is getting complex. Let's use a simpler reference.
    # Actually, the task says "smooth CPL dark-energy cosmologies" and gives q=1-Om(a).
    # For the reference, I'll use the standard growth equation.
    pass

def _build_reference_models():
    """Build a set of reference cosmologies for fitting."""
    models = []
    # Standard ΛCDM
    models.append((0.3, 0.0, 0.0))  # Om0, w0, wa
    # Quintessence
    models.append((0.3, -0.5, 0.0))
    models.append((0.3, -1.0, 0.5))
    models.append((0.25, -1.0, 1.0))
    models.append((0.35, -0.9, 0.3))
    return models

def _compute_reference(Om0, w0, wa, a_values):
    """Compute reference f and D for a given cosmology."""
    a_min = a_values.min()
    a_max = a_values.max()
    
    # Compute E(a)^2 for CPL dark energy
    # w(a) = w0 + wa*(1-a)
    # H(a)^2/H0^2 = Om0*a^-3 + (1-Om0)*exp(-3*∫_1^a w(a')/a' da')
    # ∫_1^a w(a')/a' da' = w0*ln(a) + wa*(a - 1 - ln(a))
    
    def E2(a):
        integral = w0 * np.log(a) + wa * (a - 1 - np.log(a))
        de_term = (1 - Om0) * np.exp(-3 * integral)
        return Om0 * a**(-3) + de_term
    
    # Growth equation: D'' + 2*H*H_dot/H^2 * D' - 3*Om_m(a)*D = 0
    # Or in terms of ln(a): d2D/dln(a)^2 + (3/2 + dlnH/dlna) * dD/dlna - 3*Om_m(a)/2 * D = 0
    
    def dH2_dlna(a):
        # d/dlna [E2(a)] = dE2/da * a
        # E2 = Om0*a^-3 + (1-Om0)*exp(-3*(w0*ln(a) + wa*(a-1-ln(a))))
        # Let g(a) = w0*ln(a) + wa*(a-1-ln(a))
        # dg/da = w0/a + wa*(1 - 1/a) = w0/a + wa*(a-1)/a
        # dE2/da = -3*Om0*a^-4 + (1-Om0)*exp(-3*g) * (-3*dg/da)
        dg_da = w0/a + wa*(a-1)/a
        de_term = (1 - Om0) * np.exp(-3 * (w0*np.log(a) + wa*(a-1-np.log(a))))
        dE2_da = -3*Om0*a**(-4) + de_term * (-3*dg_da)
        return a * dE2_da
    
    def growth_ode(x, y):
        a = np.exp(x)
        dD = y[1]
        d2D = - (3/2 + dH2_dlna(a)/(2*E2(a))) * dD + 3*Om0*a**(-3)/E2(a) * y[0]
        return [dD, d2D]
    
    # Integrate from a=0.1 to a=1
    x_vals = np.log(a_values)
    x_min = x_vals.min()
    x_max = x_vals.max()
    
    # Initial conditions at a_min
    a0 = np.exp(x_min)
    E2_0 = E2(a0)
    Om_m_0 = Om0 * a0**(-3) / E2_0
    # For early times, D ~ a^(3/2) * (Om_m0)^(2/5)
    D0 = a0
    dD0 = 1.0  # approximate
    
    sol = solve_ivp(growth_ode, [x_min, x_max], [D0, dD0], 
                    t_eval=x_vals, rtol=1e-10, atol=1e-12, max_step=0.01)
    
    if not sol.success:
        return None, None
    
    D = sol.y[0]
    dD = sol.y[1]
    D = D / D[-1]  # Normalize to D(1)=1
    dD = dD / D[-1]
    
    # f = dD/dlna / D = dD/D
    f = dD / D
    
    return f, D

def _evaluate_ansatz(coeffs, models, a_values):
    """Evaluate the ansatz and compute max relative error."""
    c0, c1, c2, c3, c4, c5 = coeffs
    
    max_err = 0.0
    
    for (Om0, w0, wa) in models:
        f_ref, D_ref = _compute_reference(Om0, w0, wa, a_values)
        if f_ref is None:
            continue
        
        # Compute ansatz values
        Om_a = Om0 * a_values**(-3)
        # Need E2 to get Om(a) properly
        # Actually, for the ansatz we need Om(a) = Om0*a^-3/E2(a)
        # Let's compute E2
        integral = w0 * np.log(a_values) + wa * (a_values - 1 - np.log(a_values))
        de_term = (1 - Om0) * np.exp(-3 * integral)
        E2 = Om0 * a_values**(-3) + de_term
        Om_a = Om0 * a_values**(-3) / E2
        
        q = 1 - Om_a
        u = 1 + w0 + wa * (1 - a_values)
        
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_values + c5*u*q
        
        # Check gamma range
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1000.0
        
        f_ans = Om_a**gamma
        
        # Compute D from f
        # D(a)/D(1) = exp(-∫_a^1 f(exp(x)) dx) where x=ln(a)
        # Let's compute the integral numerically
        x = np.log(a_values)
        f_x = f_ans  # f as function of x
        
        # Integrate from each x_i to x_max (which is ln(1)=0)
        # Using cumulative trapezoidal integration
        from scipy.integrate import cumulative_trapezoid
        # We need ∫_x_i^0 f(x) dx = ∫_0^x_i f(x) dx with sign change
        # Let's compute ∫_0^x f(x) dx for all x
        # Sort x values
        idx = np.argsort(x)
        x_sorted = x[idx]
        f_sorted = f_x[idx]
        
        # Integrate from x[0] to each point
        int_from_min = cumulative_trapezoid(f_sorted, x_sorted, initial=0)
        # ∫_x_i^0 f dx = ∫_x_min^0 f dx - ∫_x_min^x_i f dx
        # = int_from_min[-1] - int_from_min[i]
        int_to_0 = int_from_min[-1] - int_from_min
        
        # Unsort
        int_to_0_unsorted = np.empty_like(int_to_0)
        int_to_0_unsorted[idx] = int_to_0
        
        D_ans = np.exp(-int_to_0_unsorted)
        
        # Compute relative errors
        err_f = np.abs(f_ans - f_ref) / np.abs(f_ref)
        err_D = np.abs(D_ans - D_ref) / np.abs(D_ref)
        
        max_err = max(max_err, np.max(err_f), np.max(err_D))
    
    return 100 * max_err

def construct():
    """Construct the six coefficients."""
    start_time = time.time()
    
    # Set up the fitting
    models = _build_reference_models()
    a_values = np.linspace(0.1, 1.0, 50)
    
    # Initial guess
    best_coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_score = _evaluate_ansatz(best_coeffs, models, a_values)
    
    # Use differential evolution for global optimization
    bounds = [(-10, 10)] * 6
    
    def objective(coeffs):
        return _evaluate_ansatz(coeffs, models, a_values)
    
    try:
        result = differential_evolution(
            objective, bounds, 
            maxiter=50, popsize=15, 
            tol=1e-6, seed=42,
            polish=True
        )
        
        if result.fun < best_score:
            best_coeffs = result.x.tolist()
            best_score = result.fun
        
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5:
        return best_coeffs
    
    # Try local refinement
    from scipy.optimize import minimize
    
    try:
        res = minimize(objective, best_coeffs, method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.fun < best_score:
            best_coeffs = res.x.tolist()
            best_score = res.fun
    except Exception:
        pass
    
    return best_coeffs
