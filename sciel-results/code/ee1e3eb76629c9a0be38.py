"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp, cumulative_trapezoid
import time

_start_time = time.time()
_DEADLINE = 85.0

_cosmologies = [
    (0.3, -1.0, 0.0),
    (0.3, -0.9, 0.1),
    (0.3, -1.1, 0.2),
    (0.3, -0.8, -0.2),
    (0.3, -1.2, 0.3),
    (0.2, -1.0, 0.0),
    (0.4, -1.0, 0.0),
    (0.3, -0.5, 0.0),
]
_a_grid = np.linspace(0.05, 1.0, 30)
_ref_data = {}

def _precompute_references():
    for Om0, w0, wa in _cosmologies:
        Ode0 = 1.0 - Om0
        a_vals = _a_grid
        
        def dlnHdx(a):
            Om_a = Om0 * a**(-3)
            if Ode0 == 0:
                Ode_a = 0.0
                dOde_da = 0.0
            else:
                log_Ode_a = np.log(Ode0) - 3*(1+w0)*np.log(a) - 3*wa*(1-a)
                Ode_a = np.exp(log_Ode_a)
                dOde_da = Ode_a * (-3*(1+w0)/a + 3*wa)
            H2 = Om_a + Ode_a
            dH2_da = -3*Om0*a**(-4) + dOde_da
            dH2_dx = dH2_da * a
            return 0.5 * dH2_dx / H2
        
        def ode_system(x, y):
            a = np.exp(x)
            Om = Om0 * a**(-3)
            dlnHdx_val = dlnHdx(a)
            dDdx = y[1]
            d2Ddx2 = -3 * Om * y[0] - (3 + dlnHdx_val) * y[1]
            return [dDdx, d2Ddx2]
        
        f1 = Om0**0.6
        D1 = 1.0
        D1_prime = f1 * D1
        x_vals = np.log(a_vals)
        x_min = min(x_vals)
        
        sol = solve_ivp(ode_system, [0.0, x_min], [D1, D1_prime], 
                        rtol=1e-10, atol=1e-12, method='DOP853',
                        dense_output=True)
        
        D_vals = np.zeros(len(a_vals))
        f_vals = np.zeros(len(a_vals))
        for i, x in enumerate(x_vals):
            y = sol.sol(x)
            D_vals[i] = y[0]
            f_vals[i] = y[1] / y[0]
        
        D_norm = D_vals / D_vals[-1]
        _ref_data[(Om0, w0, wa)] = (f_vals, D_norm)

def _compute_model_fast(c0, c1, c2, c3, c4, c5, Om0, w0, wa, a_vals):
    a_vals = np.array(a_vals)
    Om_a = Om0 * a_vals**(-3)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa*(1.0 - a_vals)
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_vals + c5*u*q
    gamma = np.clip(gamma, 0.1, 1.2)
    f_vals = Om_a**gamma
    
    x_vals = np.log(a_vals)
    
    # Compute integral from a to 1 using cumulative trapezoid
    # integral(a) = ∫_a^1 f(x) dx = -∫_1^a f(x) dx
    # cumulative_trapezoid gives ∫_x[0]^x, so we need to reverse
    integral_forward = cumulative_trapezoid(f_vals, x_vals, initial=0.0)
    # integral_forward[i] = ∫_x[0]^x[i] f dx
    # We want integral_to_0[i] = ∫_x[i]^x[-1] f dx = integral_forward[-1] - integral_forward[i]
    integral_to_0 = integral_forward[-1] - integral_forward
    
    D_norm = np.exp(-integral_to_0)
    
    return f_vals, D_norm

def _evaluate_error(coeffs):
    c0, c1, c2, c3, c4, c5 = coeffs
    max_err = 0.0
    for Om0, w0, wa in _cosmologies:
        f_ref, D_ref = _ref_data[(Om0, w0, wa)]
        f_mod, D_mod = _compute_model_fast(c0, c1, c2, c3, c4, c5, Om0, w0, wa, _a_grid)
        
        err_f = np.max(np.abs((f_mod - f_ref) / f_ref))
        err_D = np.max(np.abs((D_mod - D_ref) / D_ref))
        
        max_err = max(max_err, err_f, err_D)
    
    return max_err

def _is_valid(coeffs, cosmologies, a_grid):
    c0, c1, c2, c3, c4, c5 = coeffs
    for Om0, w0, wa in cosmologies:
        Om_a = Om0 * a_grid**(-3)
        q = 1.0 - Om_a
        u = 1.0 + w0 + wa*(1.0 - a_grid)
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_grid + c5*u*q
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return False
    return True

def construct():
    global _start_time
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    _precompute_references()
    
    def objective(coeffs):
        if time.time() - _start_time > _DEADLINE - 5:
            return 1e10
        return _evaluate_error(coeffs)
    
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_result = None
    best_score = float('inf')
    
    # Try initial guess with L-BFGS-B
    try:
        result = minimize(objective, x0, method='L-BFGS-B', 
                         options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-10})
        if result.fun < best_score and _is_valid(result.x, _cosmologies, _a_grid):
            best_score = result.fun
            best_result = result.x
    except:
        pass
    
    if time.time() - _start_time > _DEADLINE - 15:
        if best_result is not None:
            return np.clip(best_result, -10, 10).tolist()
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Try other starting points with both L-BFGS-B and Nelder-Mead
    starting_points = [
        [0.5, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.1, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.1, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.1, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.1],
        [0.6, -0.1, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.05, 0.05, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.05, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.05],
    ]
    
    for x0_try in starting_points:
        if time.time() - _start_time > _DEADLINE - 15:
            break
        # Try L-BFGS-B first
        try:
            result = minimize(objective, x0_try, method='L-BFGS-B', 
                             options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-10})
            if result.fun < best_score and _is_valid(result.x, _cosmologies, _a_grid):
                best_score = result.fun
                best_result = result.x
        except:
            pass
        
        # Try Nelder-Mead if time permits
        if time.time() - _start_time > _DEADLINE - 10:
            break
        try:
            result = minimize(objective, x0_try, method='Nelder-Mead', 
                             options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < best_score and _is_valid(result.x, _cosmologies, _a_grid):
                best_score = result.fun
                best_result = result.x
        except:
            pass
    
    if best_result is not None:
        return np.clip(best_result, -10, 10).tolist()
    
    # Fallback to safe constant gamma
    return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
# EVOLVE-BLOCK-END
