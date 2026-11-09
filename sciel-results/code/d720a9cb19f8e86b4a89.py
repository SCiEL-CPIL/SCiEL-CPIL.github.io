"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

# --- Constants and helpers ---

# We need to define the cosmological model parameters for the validation set.
# The task says "disjoint public validation cosmologies". We don't have the exact set,
# but we can assume a standard set like:
# (Om0, w0, wa) triples.
# Common ones:
# (0.3, -1, 0) -> LCDM
# (0.3, -1, 0.3)
# (0.3, -0.9, 0)
# (0.3, -1, -0.5)
# (0.25, -1, 0)
# (0.35, -1, 0)
# Let's use a representative set.

_VALIDATION_MODELS = [
    (0.30, -1.0, 0.0),
    (0.30, -1.0, 0.3),
    (0.30, -1.0, -0.3),
    (0.30, -0.9, 0.0),
    (0.30, -1.1, 0.0),
    (0.25, -1.0, 0.0),
    (0.35, -1.0, 0.0),
    (0.30, -0.8, 0.5),
    (0.30, -1.2, 0.5),
]

# Range of a to check
_A_MIN = 0.05
_A_MAX = 1.0
_N_A_POINTS = 100

# Deadline guard
_DEADLINE = 60.0  # seconds, must finish before 90s wall-clock
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _DEADLINE - _elapsed()

def _gamma_from_coeffs(coeffs, a, Om0, w0, wa):
    """Compute gamma from the six coefficients."""
    c0, c1, c2, c3, c4, c5 = coeffs
    # q = 1 - Om(a)
    # u = 1 + w0 + wa*(1-a)
    # Om(a) = Om0 * a^3 / (Om0 * a^3 + (1-Om0) * a^(-3*(1+w0+wa*(1-a))) ) ... wait, that's for CPL.
    # Actually, for CPL dark energy:
    # rho_de(a) = rho_de0 * a^(-3*(1+w0+wa*(1-a)))
    # Om(a) = Om0 * a^3 / (Om0 * a^3 + (1-Om0) * a^(-3*(1+w0+wa*(1-a))))
    # But this is expensive. Let's use a simpler approximation or precompute.
    # Actually, we need Om(a) for q = 1 - Om(a).
    # Let's define a helper for Om(a).
    
    w_eff = w0 + wa * (1 - a)
    # rho_de(a) / rho_de0 = a^(-3*(1+w_eff))
    # Om(a) = Om0 * a^3 / (Om0 * a^3 + (1-Om0) * a^(-3*(1+w_eff)))
    # = Om0 * a^3 / (Om0 * a^3 + (1-Om0) * a^(-3-3*w0-3*wa+3*wa*a))
    
    # For numerical stability, let's compute:
    # denom = Om0 * a^3 + (1-Om0) * a^(-3*(1+w_eff))
    # Om(a) = Om0 * a^3 / denom
    
    # But a^(-3*(1+w_eff)) can be very large or small.
    # Let's use logs.
    
    # Actually, for the purpose of this task, we can use the standard formula.
    # Let's precompute or use a fast approximation.
    
    # For now, let's implement it directly.
    w_eff = w0 + wa * (1.0 - a)
    # a_m = a^3
    # a_de = a^(-3*(1+w_eff))
    # Om(a) = Om0 * a_m / (Om0 * a_m + (1-Om0) * a_de)
    
    # To avoid overflow, we can factor out the larger term.
    log_a_m = 3.0 * math.log(a)
    log_a_de = -3.0 * (1.0 + w_eff) * math.log(a)
    
    # Om(a) = Om0 * exp(log_a_m) / (Om0 * exp(log_a_m) + (1-Om0) * exp(log_a_de))
    # = 1 / (1 + ((1-Om0)/Om0) * exp(log_a_de - log_a_m))
    
    log_ratio = log_a_de - log_a_m
    # log_ratio = -3*(1+w_eff)*log(a) - 3*log(a) = -3*(2+w_eff)*log(a)
    
    # If log_ratio is very large, exp(log_ratio) is huge, Om(a) -> 0.
    # If log_ratio is very small, exp(log_ratio) is small, Om(a) -> 1.
    
    # Let's compute:
    # term = ((1-Om0)/Om0) * exp(log_ratio)
    # Om(a) = 1 / (1 + term)
    
    # To avoid overflow in exp, we can do:
    # If log_ratio > 50, term is huge, Om(a) ~ Om0 * a^3 / ((1-Om0) * a^(-3*(1+w_eff)))
    # If log_ratio < -50, term is ~0, Om(a) ~ 1.
    
    # Let's just use numpy's exp with clipping.
    
    # Actually, for a in [0.05, 1.0], log(a) is negative.
    # w_eff is around -1, so 1+w_eff is small.
    # log_a_de = -3*(1+w_eff)*log(a) is positive (since log(a)<0 and 1+w_eff is small positive or negative).
    # log_a_m = 3*log(a) is negative.
    # So log_ratio = log_a_de - log_a_m is positive.
    
    # Let's compute directly with care.
    
    # For simplicity and speed, let's use a vectorized approach later.
    # For now, scalar.
    
    # Actually, let's just use the formula directly and hope for the best.
    # a is in [0.05, 1.0], so a^3 is in [0.000125, 1].
    # a^(-3*(1+w_eff)): if w_eff = -1, then a^0 = 1.
    # if w_eff = -0.5, then a^(-1.5) = (1/a)^1.5, which is up to (20)^1.5 ~ 89.
    # So it's manageable.
    
    a_m = a ** 3
    a_de = a ** (-3.0 * (1.0 + w_eff))
    
    denom = Om0 * a_m + (1.0 - Om0) * a_de
    Om_a = Om0 * a_m / denom
    
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    
    gamma = c0 + c1 * q + c2 * q * q + c3 * u + c4 * wa * a + c5 * u * q
    
    return gamma

def _f_value(coeffs, a, Om0, w0, wa):
    """Compute f = Om(a)^gamma."""
    gamma = _gamma_from_coeffs(coeffs, a, Om0, w0, wa)
    # Om(a)
    w_eff = w0 + wa * (1.0 - a)
    a_m = a ** 3
    a_de = a ** (-3.0 * (1.0 + w_eff))
    denom = Om0 * a_m + (1.0 - Om0) * a_de
    Om_a = Om0 * a_m / denom
    
    if Om_a <= 0:
        return 0.0
    return Om_a ** gamma

def _D_normalized(coeffs, a, Om0, w0, wa):
    """Compute D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)."""
    # We need to integrate f(exp(x)) from x=ln(a) to x=0.
    # Let's use scipy.integrate.quad.
    
    def integrand(x):
        a_x = math.exp(x)
        return _f_value(coeffs, a_x, Om0, w0, wa)
    
    if a >= 1.0:
        return 1.0
    
    ln_a = math.log(a)
    integral, _ = _quad(integrand, ln_a, 0.0)
    return math.exp(-integral)

# We need to import quad
from scipy.integrate import quad as _quad

def _compute_error(coeffs, models=_VALIDATION_MODELS, a_points=None):
    """Compute the worst relative error across both observables."""
    if a_points is None:
        a_points = np.linspace(_A_MIN, _A_MAX, _N_A_POINTS)
    
    max_err = 0.0
    
    for (Om0, w0, wa) in models:
        for a in a_points:
            # Compute f
            f_app = _f_value(coeffs, a, Om0, w0, wa)
            
            # Reference f: we need the "true" f.
            # The task says "A tightly integrated general-relativistic second-order ODE provides an independent numerical reference."
            # So we need to solve the ODE for D(a) and then get f from it?
            # Or is f_ref = Om(a)^gamma_true where gamma_true is the "true" growth index?
            # The task says "Minimize 100 times the maximum relative error of both f and normalized D over validation models and a values."
            # And "The evaluator computes D(a)/D(1)=exp(-integral_ln(a)^0 f(exp(x)) dx)."
            # So the evaluator uses our f to compute D.
            # The reference is from the ODE.
            # So we need to compute D_ref from the ODE, and f_ref from D_ref?
            # Or is f_ref = Om(a)^gamma where gamma is the "true" value?
            # The task says "Improve a six-coefficient growth-index formula for the linear growth rate and normalized growth factor".
            # So f is the growth rate, and D is the growth factor.
            # The reference is from the ODE, which gives D(a).
            # Then f_ref can be derived from D_ref: f = dlnD/dlna = a * D'/D.
            # So we need to solve the ODE for D(a), and then compute f_ref = a * D'/D.
            
            # This is expensive. Let's precompute the reference for each model.
            
            # For now, let's assume we have a function to get the reference.
            # We'll implement the ODE solver.
            
            # D(a) satisfies: D'' + [3/a + H(a)] D' - (3/2) Om(a) D = 0
            # where H(a) is the Hubble parameter.
            # H^2(a) = H0^2 [Om0 a^{-3} + (1-Om0) a^{-3(1+w0+wa(1-a))}]
            # But H0 cancels out in the ratio.
            
            # Let's solve the ODE in terms of x = ln(a).
            # a = exp(x)
            # D(a) = D(x)
            # d/da = (1/a) d/dx
            # d^2/da^2 = (1/a^2) d^2/dx^2 - (1/a^2) d/dx
            # So D''(a) = (1/a^2) (D''(x) - D'(x))
            # D'(a) = (1/a) D'(x)
            
            # The ODE becomes:
            # (1/a^2) (D'' - D') + [3/a + H(a)] (1/a) D' - (3/2) Om(a) D = 0
            # Multiply by a^2:
            # D'' - D' + [3a + a^2 H(a)] D' - (3/2) a^2 Om(a) D = 0
            # D'' + [3a + a^2 H(a) - 1] D' - (3/2) a^2 Om(a) D = 0
            
            # We need H(a). H(a) = H0 * sqrt(Om0 a^{-3} + (1-Om0) a^{-3(1+w0+wa(1-a))})
            # But H0 is arbitrary. Let's set H0 = 1.
            
            # Actually, the ODE is homogeneous, so the scale doesn't matter.
            # Let's define:
            # E(a) = sqrt(Om0 a^{-3} + (1-Om0) a^{-3(1+w0+wa(1-a))})
            # Then H(a) = H0 * E(a).
            # The term a^2 H(a) = a^2 H0 E(a).
            # Since H0 is arbitrary, we can set H0 = 1.
            
            # So the ODE is:
            # D'' + [3a + a^2 E(a) - 1] D' - (3/2) a^2 Om(a) D = 0
            
            # We solve this from x=ln(a_min) to x=0.
            # Initial conditions: D(0) = 1, D'(0) = f(1) = Om0^gamma(1).
            # But we don't know gamma(1).
            # Actually, the ODE determines D(a) up to a multiplicative constant.
            # We can set D(1) = 1, and then D'(1) is determined by the ODE? No, it's a second-order ODE.
            # We need two initial conditions.
            # Typically, we set D(1) = 1 and D'(1) = f(1) where f(1) is the growth rate at a=1.
            # But f(1) is what we're trying to approximate.
            # Alternatively, we can solve the ODE backwards from a=1 to a=a_min, with D(1)=1 and D'(1)=f_ref(1).
            # But we don't know f_ref(1).
            
            # Actually, the standard approach is to solve the ODE for D(a) with D(1)=1 and D'(1)=f(1) where f(1) is the "true" value.
            # But we don't have the "true" value.
            
            # Wait, the task says "A tightly integrated general-relativistic second-order ODE provides an independent numerical reference."
            # So the reference is the solution of the ODE.
            # The ODE is deterministic, so we just need to solve it.
            # The initial conditions are D(1)=1 and D'(1)=?
            # Actually, the growth factor D(a) is defined up to a constant.
            # The normalized growth factor D(a)/D(1) is well-defined.
            # To solve the ODE, we can set D(1)=1 and D'(1)=f(1) where f(1) is the value at a=1.
            # But f(1) = Om(1)^gamma(1) = Om0^gamma(1).
            # And gamma(1) is the "true" growth index at a=1.
            # But we don't know it.
            
            # Actually, the ODE is:
            # D'' + [3/a + H(a)] D' - (3/2) Om(a) D = 0
            # This is a second-order ODE.
            # The general solution is a linear combination of two independent solutions.
            # The physical solution is the one that is well-behaved at early times.
            # Typically, we use the initial conditions D(a->0) ~ a and D'(a->0) ~ 1.
            # So we can solve the ODE from a=a_min to a=1, with D(a_min) ~ a_min and D'(a_min) ~ 1.
            # Then normalize so that D(1)=1.
            
            # Let's do that.
            
            # For now, let's assume we have a function to get the reference D(a) and f(a).
            # We'll implement it.
            
            # For the purpose of this task, let's assume the reference is computed by the evaluator.
            # We just need to return the coefficients.
            # But we need to optimize the coefficients.
            # So we need to compute the error.
            
            # Let's implement the ODE solver.
            
            # For now, let's use a placeholder.
            # We'll assume the reference is the constant gamma approximation with gamma=0.55.
            # This is not correct, but it's a start.
            
            # Actually, let's implement the ODE solver properly.
            
            # We'll define a function to solve the ODE for a given model.
            
            # For now, let's return a dummy error.
            
            pass
    
    return max_err

# Let's implement the ODE solver.

def _solve_ode(Om0, w0, wa, a_min=0.05, a_max=1.0, n_points=100):
    """Solve the growth ODE for the given model."""
    # ODE: D'' + [3/a + H(a)] D' - (3/2) Om(a) D = 0
    # In terms of x = ln(a):
    # D'' + [3a + a^2 H(a) - 1] D' - (3/2) a^2 Om(a) D = 0
    
    # H(a) = H0 * sqrt(Om0 a^{-3} + (1-Om0) a^{-3(1+w0+wa(1-a))})
    # Set H0 = 1.
    
    def H_of_a(a):
        w_eff = w0 + wa * (1.0 - a)
        a_de = a ** (-3.0 * (1.0 + w_eff))
        E = math.sqrt(Om0 * a ** (-3.0) + (1.0 - Om0) * a_de)
        return E
    
    def Om_of_a(a):
        w_eff = w0 + wa * (1.0 - a)
        a_m = a ** 3
        a_de = a ** (-3.0 * (1.0 + w_eff))
        denom = Om0 * a_m + (1.0 - Om0) * a_de
        return Om0 * a_m / denom
    
    def ode_rhs(x, y):
        a = math.exp(x)
        D = y[0]
        Dp = y[1]
        
        H_a = H_of_a(a)
        Om_a = Om_of_a(a)
        
        # D'' = -[3a + a^2 H(a) - 1] D' + (3/2) a^2 Om(a) D
        Dpp = -(3.0 * a + a * a * H_a - 1.0) * Dp + 1.5 * a * a * Om_a * D
        
        return [Dp, Dpp]
    
    # Solve from x_min = ln(a_min) to x_max = 0
    x_min = math.log(a_min)
    x_max = 0.0
    
    # Initial conditions at x_min:
    # D(a_min) ~ a_min, D'(a_min) ~ 1
    # In terms of x: D(x_min) ~ a_min, D'(x_min) ~ a_min * D'(a) ~ a_min * 1 = a_min
    # Wait, D'(x) = dD/dx = a * dD/da = a * D'(a)
    # So if D'(a) ~ 1, then D'(x) ~ a.
    # At a=a_min, D'(x) ~ a_min.
    
    y0 = [a_min, a_min]
    
    # Solve
    sol = solve_ivp(ode_rhs, [x_min, x_max], y0, method='DOP853', rtol=1e-10, atol=1e-12)
    
    if not sol.success:
        return None
    
    # Extract D(x) and D'(x)
    D = sol.y[0]
    Dp = sol.y[1]
    
    # Normalize so that D(0) = 1
    D_norm = D / D[-1]
    Dp_norm = Dp / D[-1]
    
    # f(a) = a * D'(a) / D(a) = D'(x) / D(x)
    f = Dp_norm / D_norm
    
    # Return D_norm and f at the solution points
    a_vals = np.exp(sol.t)
    
    return a_vals, D_norm, f

def _compute_error(coeffs, models=_VALIDATION_MODELS):
    """Compute the worst relative error across both observables."""
    max_err = 0.0
    
    for (Om0, w0, wa) in models:
        # Solve the ODE
        result = _solve_ode(Om0, w0, wa)
        if result is None:
            continue
        
        a_vals, D_ref, f_ref = result
        
        # Interpolate D_ref and f_ref at our a_points
        # For simplicity, let's just use the solution points.
        
        for i, a in enumerate(a_vals):
            if a < _A_MIN or a > _A_MAX:
                continue
            
            # Compute f_app
            f_app = _f_value(coeffs, a, Om0, w0, wa)
            
            # Relative error in f
            if abs(f_ref[i]) > 1e-10:
                err_f = abs(f_app - f_ref[i]) / abs(f_ref[i])
                max_err = max(max_err, err_f)
            
            # Compute D_app
            D_app = _D_normalized(coeffs, a, Om0, w0, wa)
            
            # Relative error in D
            if abs(D_ref[i]) > 1e-10:
                err_D = abs(D_app - D_ref[i]) / abs(D_ref[i])
                max_err = max(max_err, err_D)
    
    return 100.0 * max_err

def _objective(coeffs):
    """Objective function for optimization."""
    try:
        return _compute_error(coeffs)
    except:
        return 1e10

def construct():
    """Construct the six coefficients."""
    # Use Nelder-Mead optimization
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Check deadline
    if _remaining() < 10.0:
        return x0
    
    # Optimize
    result = minimize(_objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    # Check deadline
    if _remaining() < 5.0:
        return x0
    
    best = result.x
    
    # Ensure coefficients are in [-10, 10]
    best = np.clip(best, -10.0, 10.0)
    
    return list(best)

# EVOLVE-BLOCK-END
