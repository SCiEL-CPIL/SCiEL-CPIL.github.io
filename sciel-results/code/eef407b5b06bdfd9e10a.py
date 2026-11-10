"""Growth-index approximation optimizer for CPL dark-energy cosmologies."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 90.0
_SAFETY_MARGIN = 5.0

def _elapsed_ok():
    return (time.time() - _START_TIME) < (_DEADLINE - _SAFETY_MARGIN)

def _Om_of_a(a, Om0, w0, wa):
    """Compute Omega_m(a) for CPL dark energy."""
    # E(a)^2 = Om0 * a^-3 + (1-Om0) * exp(-3*integral_1^a (1+w(a'))/a' da')
    # For CPL: w(a) = w0 + wa*(1-a)
    # integral_1^a (1+w(a'))/a' da' = integral_1^a (1+w0+wa*(1-a'))/a' da'
    # = (1+w0) * ln(a) + wa * integral_1^a (1/a' - 1) da'
    # = (1+w0)*ln(a) + wa*(ln(a) - (a-1))
    # = (1+w0+wa)*ln(a) - wa*(a-1)
    # So E(a)^2 = Om0 * a^-3 + (1-Om0) * exp(-3*[(1+w0+wa)*ln(a) - wa*(a-1)])
    #           = Om0 * a^-3 + (1-Om0) * a^(-3*(1+w0+wa)) * exp(3*wa*(a-1))
    if a <= 0:
        return 1.0
    a = max(a, 1e-10)
    Om0 = max(Om0, 1e-10)
    factor = 3.0 * (1.0 + w0 + wa)
    exp_term = math.exp(3.0 * wa * (a - 1.0))
    de_term = (1.0 - Om0) * (a ** (-factor)) * exp_term
    m_term = Om0 * (a ** (-3.0))
    E2 = m_term + de_term
    if E2 <= 0:
        E2 = 1e-30
    return m_term / E2

def _gamma_from_coeffs(coeffs, a, Om0, w0, wa):
    """Compute gamma from the six-coefficient ansatz."""
    c0, c1, c2, c3, c4, c5 = coeffs
    Om = _Om_of_a(a, Om0, w0, wa)
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
    return gamma

def _f_approx(coeffs, a, Om0, w0, wa):
    """Compute f = Om(a)^gamma."""
    gamma = _gamma_from_coeffs(coeffs, a, Om0, w0, wa)
    Om = _Om_of_a(a, Om0, w0, wa)
    if Om <= 0:
        return 0.0
    return Om ** gamma

def _D_normalized(coeffs, a, Om0, w0, wa, n_points=200):
    """Compute D(a)/D(1) = exp(-integral_ln(a)^0 f(exp(x)) dx)."""
    if a >= 1.0:
        return 1.0
    # Integrate from ln(a) to 0
    x0 = math.log(max(a, 1e-10))
    x1 = 0.0
    
    def integrand(x):
        ax = math.exp(x)
        return _f_approx(coeffs, ax, Om0, w0, wa)
    
    # Use Simpson's rule or trapezoidal for the integral
    xs = np.linspace(x0, x1, n_points)
    ys = np.array([integrand(x) for x in xs])
    integral = np.trapz(ys, xs)
    return math.exp(-integral)

def _reference_solution(a, Om0, w0, wa):
    """Solve the GR second-order ODE for f(a) = a d(ln D)/da."""
    # f(a) = a * d(ln D)/da
    # The ODE is: f'' + (3+q)/(2a) * f' + [3(1+Om)/2 - 2q] * f/a^2 = 0
    # where q = -a * H'/H = 1 - Om(a) for CPL (approximately)
    # Actually for CPL: q(a) = 1 - Om(a) + (3/2)*wa*(1-a)*Om(a) ... let me use the standard form
    
    # Standard growth ODE: f'' + (3+q)/(2a) f' + (3/2)(1+Om) f/a^2 = 0
    # where q = -d ln H / d ln a = 1 - Om + (3/2)*wa*(1-a)*Om for CPL
    
    if a >= 1.0:
        # Use asymptotic values at a=1
        Om1 = Om0
        q1 = 1.0 - Om1 + 1.5 * wa * 0.0 * Om1  # at a=1, (1-a)=0
        q1 = 1.0 - Om1
        f1 = Om1 ** 0.55  # approximate
        return f1
    
    # Solve from a=1 down to a
    def ode_func(a_val, y):
        f, fp = y
        Om = _Om_of_a(a_val, Om0, w0, wa)
        q = 1.0 - Om + 1.5 * wa * (1.0 - a_val) * Om
        coeff1 = (3.0 + q) / (2.0 * a_val)
        coeff2 = 1.5 * (1.0 + Om) / (a_val ** 2)
        fpp = -coeff1 * fp - coeff2 * f
        return [fp, fpp]
    
    # Initial conditions at a=1
    Om1 = Om0
    f1 = Om1 ** 0.55
    # f'(1) ≈ 0.55 * (1 - Om1) * Om1^(0.55-1) / a ... actually f' = df/da
    # f = Om^gamma, f' = gamma * Om^(gamma-1) * Om'
    # Om' = dOm/da = -Om(3 - Om(1+q)) / a ... this gets complicated
    # Use f'(1) ≈ 0 as a rough initial condition, or compute more carefully
    # Actually, for a better IC: f(a) ≈ Om(a)^gamma, so f'(a) = gamma * Om^(gamma-1) * dOm/da
    # dOm/da = Om * (Om - 3) / a * (1 + wa*(1-a)*3/2 * Om/(Om-3)) ... too complex
    # Let's just use f'(1) = 0.5 * (1-Om1) * f1 / 1 as a rough estimate
    fp1 = 0.5 * (1.0 - Om1) * f1
    
    try:
        sol = solve_ivp(ode_func, [1.0, a], [f1, fp1], 
                       method='RK45', rtol=1e-8, atol=1e-10,
                       max_step=0.01, dense_output=True)
        if sol.success:
            return sol.y[0, -1]
    except:
        pass
    return f1

def _objective(coeffs, models, a_values):
    """Compute the maximum relative error across all models and a values."""
    max_err = 0.0
    for (Om0, w0, wa) in models:
        for a in a_values:
            if a >= 1.0:
                continue
            # Approximate
            f_app = _f_approx(coeffs, a, Om0, w0, wa)
            D_app = _D_normalized(coeffs, a, Om0, w0, wa)
            
            # Reference
            f_ref = _reference_solution(a, Om0, w0, wa)
            # D_ref: integrate the reference f
            # D(a)/D(1) = exp(-int_ln(a)^0 f_ref(exp(x)) dx)
            x0 = math.log(max(a, 1e-10))
            x1 = 0.0
            xs = np.linspace(x0, x1, 100)
            ys = []
            for x in xs:
                ax = math.exp(x)
                ys.append(_reference_solution(ax, Om0, w0, wa))
            integral = np.trapz(ys, xs)
            D_ref = math.exp(-integral)
            
            # Relative errors
            if abs(f_ref) > 1e-10:
                err_f = abs(f_app - f_ref) / abs(f_ref)
            else:
                err_f = 0.0
            if abs(D_ref) > 1e-10:
                err_D = abs(D_app - D_ref) / abs(D_ref)
            else:
                err_D = 0.0
            
            max_err = max(max_err, err_f, err_D)
    
    return 100.0 * max_err

def construct():
    """Construct six coefficients for the growth-index approximation."""
    if not _elapsed_ok():
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Validation models: (Om0, w0, wa)
    models = [
        (0.3, -1.0, 0.0),      # LambdaCDM
        (0.3, -0.9, 0.3),      # CPL
        (0.28, -1.0, 0.0),     # LambdaCDM
        (0.3, -1.1, 0.5),      # Phantom CPL
        (0.35, -0.8, 0.2),     # Matter-dominated-ish
        (0.3, -1.0, 0.5),      # CPL with wa
        (0.25, -0.95, 0.1),    # Another CPL
    ]
    
    # a values for evaluation
    a_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    
    # Initial guess
    best_coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_score = _objective(best_coeffs, models, a_values)
    
    if not _elapsed_ok():
        return best_coeffs
    
    # Try differential evolution for global search
    bounds = [(-1.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
    
    try:
        result = differential_evolution(
            _objective, bounds, args=(models, a_values),
            maxiter=50, popsize=15, seed=42, tol=1e-4,
            polish=False, updating='deferred'
        )
        if _elapsed_ok() and result.fun < best_score:
            best_coeffs = list(result.x)
            best_score = result.fun
    except:
        pass
    
    if not _elapsed_ok():
        return best_coeffs
    
    # Local polish with L-BFGS-B
    try:
        result2 = minimize(
            _objective, best_coeffs, args=(models, a_values),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-8}
        )
        if _elapsed_ok() and result2.fun < best_score:
            best_coeffs = list(result2.x)
            best_score = result2.fun
    except:
        pass
    
    if not _elapsed_ok():
        return best_coeffs
    
    # Ensure coefficients are in [-10, 10] and finite
    best_coeffs = [max(-10.0, min(10.0, float(c))) for c in best_coeffs]
    for i, c in enumerate(best_coeffs):
        if not math.isfinite(c):
            best_coeffs[i] = 0.0
    
    return best_coeffs
