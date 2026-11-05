"""Improved growth-index approximation using local linear regression of ln(f) derivative."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp, quad

# Deadline guard
_DEADLINE = time.time() + 85.0  # 85s safety margin within 90s limit
_TIME_PER_EVAL = 0.5  # rough estimate per eval for budget

def _om_a(a, om0, w0, wa):
    """Matter density parameter at scale factor a for CPL dark energy."""
    # EdS: Om_m(a) = Om0 * a^-3
    # DE: Om_de(a) = Om_de0 * a^-3(1+3w0+3wa) / (1+wa) * (1/(1+wa))... 
    # Actually: rho_de(a) = rho_de0 * (1/a)^(3(1+w0+wa*(1-1/a)))
    # More carefully: rho_de(a) = rho_de0 * exp(-3*int_1^a (1+w(a'))/a' da')
    # For CPL: w(a) = w0 + wa*(1-a)
    # int_1^a (1+w(a'))/a' da' = int_1^a (1+w0+wa-wa*a')/a' da'
    # = int_1^a (1+w0+wa)/a' da' - wa*int_1^a da'
    # = (1+w0+wa)*ln(a) - wa*(a-1)
    # So rho_de(a) = rho_de0 * a^(-(1+w0+wa)) * exp(wa*(a-1))
    
    # Let's use the standard form:
    # Om_de(a) = Om_de0 * a^{-3(1+w0+wa)} * exp(wa*(1-a)) / [1 + (Om_de0/Om0) * a^{-3(1+w0+wa)} * exp(wa*(1-a))]
    
    de_factor = np.exp(wa * (1 - a)) * np.power(a, -3.0*(1.0+w0+wa))
    om_de0 = 1.0 - om0
    num = om_de0 * de_factor
    den = om0 + num
    return om0 / den

def _gamma_from_coeffs(c, a, om0, w0, wa):
    """Compute gamma from coefficients and cosmological parameters."""
    om = _om_a(a, om0, w0, wa)
    q = 1.0 - om
    u = 1.0 + w0 + wa * (1.0 - a)
    c0, c1, c2, c3, c4, c5 = c
    return c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q

def _f_from_gamma(gamma, om):
    """Growth rate f = Om^gamma."""
    return np.power(om, gamma)

def _normalized_D(a, c, om0, w0, wa, a_min=1e-6):
    """Compute D(a)/D(1) = exp(-int_ln(a)^0 f(exp(x)) dx) using numerical integration."""
    # D(a)/D(1) = exp(-int_{ln(a)}^0 f(e^x) dx) = exp(int_0^{ln(a)} f(e^x) dx) for a<1
    # Wait: D(a)/D(1) = exp(-∫_{ln a}^0 f(e^x) dx)
    # For a < 1, ln(a) < 0, so -∫_{ln a}^0 = ∫_0^{ln a} (negative) ... let me re-check
    # D(a)/D(1) = exp(-∫_{ln(a)}^0 f(e^x) dx)
    # If a<1, ln(a)<0, so the integral is from negative to 0, and we negate it.
    # This gives a positive exponent for a<1? No, f>0, integral from ln(a) to 0 of positive = positive, negate = negative.
    # So D(a)/D(1) < 1 for a<1. That makes sense.
    
    if a >= 1.0:
        return 1.0
    
    def integrand(x):
        ai = np.exp(x)
        om = _om_a(ai, om0, w0, wa)
        g = _gamma_from_coeffs(c, ai, om0, w0, wa)
        return np.power(om, g)
    
    try:
        val, err = quad(integrand, np.log(a), 0.0, limit=100)
        return np.exp(-val)
    except:
        return np.nan

def _compute_errors(c, models, a_values):
    """Compute max relative error for f and D across models and a values."""
    max_err = 0.0
    
    for om0, w0, wa in models:
        for a in a_values:
            om = _om_a(a, om0, w0, wa)
            if om <= 0 or om >= 1:
                continue
            
            # Our f
            g = _gamma_from_coeffs(c, a, om0, w0, wa)
            if g < 0.1 or g > 1.2:
                return 1e10  # Invalid gamma
            
            f_ours = np.power(om, g)
            
            # Reference: use ODE to compute f_ref and D_ref
            # For efficiency, we'll compute f_ref using the ODE solution
            # But for the objective, we need a fast reference. Let's use a precomputed approach.
            # Actually, let's compute f_ref via the ODE relation: f = Om^gamma_ref where gamma_ref comes from ODE.
            # This is circular. Instead, let's use the known approximate formula as reference for the ODE.
            
            # For a proper reference, we need to solve the ODE. Let's do it efficiently.
            f_ref, D_ref = _solve_reference(om0, w0, wa, a)
            
            if np.isnan(f_ref) or np.isnan(D_ref):
                continue
            
            # Relative error in f
            if abs(f_ref) > 1e-10:
                err_f = abs(f_ours - f_ref) / abs(f_ref)
                max_err = max(max_err, err_f)
            
            # Relative error in D
            D_ours = _normalized_D(a, c, om0, w0, wa)
            if not np.isnan(D_ours) and abs(D_ref) > 1e-10:
                err_D = abs(D_ours - D_ref) / abs(D_ref)
                max_err = max(max_err, err_D)
    
    return max_err

def _solve_reference(om0, w0, wa, a_eval, n_points=50):
    """Solve the growth ODE to get f and D at a_eval."""
    # ODE: D'' + 2H D' + (3/2)Om H^2 D = 0
    # Let y = D, y' = v
    # D''/D = -2H D'/D - (3/2)Om H^2 = -2H f - (3/2)Om H^2
    # f = D'/D
    # f' + 2H f = -f^2 - (3/2)Om H^2
    # In terms of x = ln a: df/dx + (1+q)f = -f^2 - (3/2)Om
    # where q = 1 - Om
    
    # We solve from a=1 (where f(1) ≈ Om0^0.55 or similar) backward to a_eval.
    
    def ode(x, y):
        ai = np.exp(x)
        om = _om_a(ai, om0, w0, wa)
        q = 1.0 - om
        f = y[0]
        dfdx = -f**2 - 1.5*om - (1.0+q)*f
        return [dfdx]
    
    # Initial condition at x=0 (a=1)
    f0 = np.power(om0, 0.55)
    
    x_end = np.log(a_eval)
    
    try:
        sol = solve_ivp(ode, [0, x_end], [f0], method='DOP853', rtol=1e-10, atol=1e-12, dense_output=True)
        if not sol.success:
            return np.nan, np.nan
        
        f_ref = sol.y[0, -1]
        
        # Compute D from f: D(a)/D(1) = exp(-∫_0^{ln(a)} f(e^x) dx)
        # Wait, D(a)/D(1) = exp(∫_a^1 f(a')/a' da') = exp(∫_{ln a}^0 f(e^x) dx)
        # Actually: d ln D / d ln a = f, so ln(D(a)/D(1)) = ∫_1^a f(a')/a' da' = ∫_0^{ln a} f(e^x) dx
        # For a<1, this integral is negative, so D(a)/D(1) < 1. Correct.
        
        def f_integrand(x):
            return sol.sol(x)[0]
        
        val, err = quad(f_integrand, 0.0, x_end, limit=100)
        D_ref = np.exp(val)
        
        return f_ref, D_ref
    except:
        return np.nan, np.nan

# Precompute reference values for speed
_reference_cache = {}

def _get_reference(om0, w0, wa, a):
    key = (round(om0,6), round(w0,6), round(wa,6), round(a,6))
    if key not in _reference_cache:
        _reference_cache[key] = _solve_reference(om0, w0, wa, a)
    return _reference_cache[key]

def _objective(c):
    """Objective function: 100 * max relative error."""
    global _TIME_PER_EVAL
    
    if time.time() > _DEADLINE:
        return 1e10
    
    # Validation models (typical CPL dark energy cosmologies)
    models = [
        (0.3, 0.0, 0.0),      # wCDM
        (0.3, -0.5, 0.0),     # Quintessence
        (0.3, -1.0, 0.0),     # Cosmological constant
        (0.3, -0.5, 0.5),     # CPL with positive wa
        (0.3, -1.0, 0.5),     # CPL with wa
        (0.25, -0.5, 0.3),    # Different Om0
        (0.3, -0.7, 0.2),     # Another CPL
    ]
    
    a_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    
    max_err = 0.0
    count = 0
    total = len(models) * len(a_values)
    
    for om0, w0, wa in models:
        for a in a_values:
            count += 1
            if time.time() > _DEADLINE:
                return 1e10
            
            om = _om_a(a, om0, w0, wa)
            if om <= 0.01 or om >= 0.99:
                continue
            
            f_ref, D_ref = _get_reference(om0, w0, wa, a)
            if np.isnan(f_ref) or np.isnan(D_ref):
                continue
            
            g = _gamma_from_coeffs(c, a, om0, w0, wa)
            if g < 0.1 or g > 1.2:
                return 1e10
            
            f_ours = np.power(om, g)
            
            err_f = abs(f_ours - f_ref) / abs(f_ref) if abs(f_ref) > 1e-10 else 0
            max_err = max(max_err, err_f)
            
            D_ours = _normalized_D(a, c, om0, w0, wa)
            if not np.isnan(D_ours) and abs(D_ref) > 1e-10:
                err_D = abs(D_ours - D_ref) / abs(D_ref)
                max_err = max(max_err, err_D)
    
    return 100.0 * max_err

def construct():
    """Return six coefficients for the growth-index approximation."""
    start_time = time.time()
    
    # Initial guess: constant gamma
    c0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Try multiple optimization strategies
    best_c = c0.copy()
    best_err = _objective(c0)
    
    # Nelder-Mead optimization
    if time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective, c0, method='Nelder-Mead', 
                         options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            if res.fun < best_err:
                best_err = res.fun
                best_c = res.x
        except:
            pass
    
    # Powell optimization from best so far
    if time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective, best_c, method='Powell',
                         options={'maxiter': 100, 'xtol': 1e-6, 'ftol': 1e-8})
            if res.fun < best_err:
                best_err = res.fun
                best_c = res.x
        except:
            pass
    
    # Local refinement with Nelder-Mead
    if time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective, best_c, method='Nelder-Mead',
                         options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < best_err:
                best_err = res.fun
                best_c = res.x
        except:
            pass
    
    # Clamp to valid range
    best_c = np.clip(best_c, -10.0, 10.0)
    
    return list(best_c)

# Ensure the function is available
if __name__ == "__main__":
    print(construct())
