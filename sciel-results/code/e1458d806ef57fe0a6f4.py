"""Growth-index fitting for CPL dark-energy cosmologies."""
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# Deadline guard: 85s hard limit (5s margin before 90s kill)
_T0 = time.time()
_DEADLINE = 85.0

def _elapsed():
    return time.time() - _T0

def _time_left():
    return _DEADLINE - _elapsed()

def _gamma_of(c, a, Om, w0, wa):
    """Compute gamma from the 6-coefficient formula."""
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    g = (c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q)
    return g

def _f_of(c, a, Om, w0, wa):
    """Growth rate f = Om^gamma."""
    g = _gamma_of(c, a, Om, w0, wa)
    # Clamp gamma to valid range to avoid issues
    g = np.clip(g, 0.1, 1.2)
    return Om ** g

def _numerical_f_and_D(a_arr, Om0, w0, wa):
    """Solve the GR growth ODE numerically to get f and D(a)/D(1).
    
    The growth equation: f' + (3/2)(1+Om) f^2 + (1/2) Om' f - (3/2) Om = 0
    where ' = d/d ln a
    D(a) satisfies: D'' + (3/2)(1+Om) D' - (1/2) Om' D - (3/2) Om D = 0
    where Om(a) = Om0 * a^-3 / (Om0*a^-3 + (1-Om0)*a^{-3(1+w0+wa/2)})
    
    Actually let's use the standard form:
    Om(a) = Om0 * a^-3 / [Om0*a^-3 + (1-Om0)*a^{-3(1+w0+wa/2)}]
    For CPL: w(a) = w0 + wa*(1-a), so w_eff = w0 + wa/2 (approx for integral)
    Actually the exact DE density: rho_de(a) = rho_de(1) * a^{-3(1+w0+wa/2)}
    Wait, that's only for constant w. For CPL:
    rho_de(a) = rho_de(1) * exp(-3*wa*ln(a) - 3*w0*(1-a))... no.
    
    For CPL: w(a) = w0 + wa*(1-a)
    rho_de(a) = rho_de(1) * exp(-3 * integral_1^a (1+w(a'))/a' da')
    = rho_de(1) * exp(-3 * [w0*ln(a) + wa*(1-a) + wa*ln(a) - wa*(1-a)] ... 
    
    Let me be more careful:
    integral_1^a (1+w(a'))/a' da' = integral_1^a (1+w0+wa*(1-a'))/a' da'
    = (1+w0) * ln(a) + wa * integral_1^a (1-a')/a' da'
    = (1+w0) * ln(a) + wa * [a' - ln(a')]_1^a  ... no
    integral (1-a')/a' da' = integral (1/a' - 1) da' = ln(a') - a'
    So integral_1^a = (ln(a) - a) - (ln(1) - 1) = ln(a) - a + 1
    
    So rho_de(a) = rho_de(1) * exp(-3*[(1+w0)*ln(a) + wa*(ln(a) - a + 1)])
    = rho_de(1) * a^{-(1+w0)} * exp(-3*wa*(ln(a) - a + 1))
    
    For the ODE, I'll work in terms of x = ln(a):
    Om(x) = Om0 * e^{-3x} / [Om0*e^{-3x} + (1-Om0) * exp(-3*[(1+w0)*x + wa*(x - e^x + 1)])]
    
    Let me define the DE factor more carefully.
    Let me just use the standard approach with x = ln(a).
    """
    # Work in x = ln(a) space
    # x ranges from ln(a_min) to 0 (a=1)
    x_arr = np.log(a_arr)
    
    def Om_of_x(x):
        a = np.exp(x)
        # DE density factor relative to a=1:
        # rho_de(a)/rho_de(1) = exp(-3 * integral_1^a (1+w)/a' da')
        # integral = (1+w0)*ln(a) + wa*(ln(a) - a + 1)
        integ = (1.0 + w0) * x + wa * (x - np.exp(x) + 1.0)
        de_factor = np.exp(-3.0 * integ)
        m_factor = np.exp(-3.0 * x)  # a^{-3}
        denom = Om0 * m_factor + (1.0 - Om0) * de_factor
        return Om0 * m_factor / denom
    
    # ODE for f: df/dx + (3/2)(1+Om) f^2 + (1/2) dOm/dx * f - (3/2) Om = 0
    # dOm/dx = -3*Om + 3*Om*(1+Om)*... let me compute numerically or analytically
    # Actually: dOm/dx = Om * (3 - 3(1+w_eff)) where w_eff is effective
    # More precisely: dOm/dx = 3*Om*(1 - (1+Om)*... 
    
    # Let me just use the standard form:
    # f' = -3/2*(1+Om)*f^2 - 1/2*Om'*f + 3/2*Om
    # where Om' = dOm/dx
    # And Om' = 3*Om - 3*Om*(1+Om)*H'/H... 
    
    # Simpler: Om' = -3*Om + 3*Om*(1+w_de)*Om_de/Om_total... 
    # Let me just compute Om' numerically via finite difference for robustness,
    # or use the analytical form.
    
    # Analytical: Om_m = Om0*exp(-3x), Om_de = (1-Om0)*exp(-3*integ(x))
    # Om = Om_m / (Om_m + Om_de)
    # Om' = (Om_m'*(Om_m+Om_de) - Om_m*(Om_m'+Om_de')) / (Om_m+Om_de)^2
    #     = (Om_m'*Om_de - Om_m*Om_de') / (Om_m+Om_de)^2
    # Om_m' = -3*Om_m
    # Om_de' = -3*(1+w(x))*Om_de where w(x) = w0+wa*(1-exp(x))
    
    def Om_and_dOm(x):
        a = np.exp(x)
        w = w0 + wa * (1.0 - a)
        integ = (1.0 + w0) * x + wa * (x - a + 1.0)
        Om_m = Om0 * np.exp(-3.0 * x)
        Om_de = (1.0 - Om0) * np.exp(-3.0 * integ)
        total = Om_m + Om_de
        Om = Om_m / total
        Om_m_prime = -3.0 * Om_m
        Om_de_prime = -3.0 * (1.0 + w) * Om_de
        dOm = (Om_m_prime * Om_de - Om_m * Om_de_prime) / (total * total)
        return Om, dOm
    
    # Solve for f using the ODE, starting from a small value and integrating forward
    # Initial condition: at early times (matter dominated), f ~ Om^0.55
    # We'll integrate from x_start to x_end = 0 (a=1)
    
    x_start = np.log(0.01)  # a=0.01
    x_end = 0.0             # a=1
    
    def ode_f(x, y):
        f = y[0]
        Om, dOm = Om_and_dOm(x)
        # f' = -3/2*(1+Om)*f^2 - 1/2*dOm*f + 3/2*Om
        return [-1.5*(1.0+Om)*f*f - 0.5*dOm*f + 1.5*Om]
    
    # Initial condition at x_start: f ~ Om^0.55
    Om_init, _ = Om_and_dOm(x_start)
    f_init = Om_init ** 0.55
    
    # Integrate from x_start to x_end
    sol = solve_ivp(ode_f, [x_start, x_end], [f_init], 
                    t_eval=x_arr, method='RK45', rtol=1e-10, atol=1e-12)
    
    if not sol.success:
        # Fallback: use constant gamma
        f_num = np.array([Om0 * (a**(-3)/(Om0*a**(-3)+(1-Om0)*a**(-3*(1+w0))))**0.55 for a in a_arr])
        D_num = np.ones_like(a_arr)
        return f_num, D_num
    
    f_num = sol.y[0]
    
    # Compute D(a)/D(1) = exp(-integral from a to 1 of f dx)
    # D(a)/D(1) = exp(-integral_{ln(a)}^{0} f(x) dx)
    # We need to integrate f from each x_i to 0
    # Use cumulative integration
    # integral from x_i to 0 = integral from x_start to 0 - integral from x_start to x_i
    
    # Compute integral from x_start to each point using cumulative trapezoidal
    from scipy.integrate import cumulative_trapezoid
    cum_int = cumulative_trapezoid(f_num, x_arr, initial=0.0)
    total_int = cum_int[-1]  # integral from x_start to x_end=0
    # integral from x_i to 0 = total_int - cum_int[i]
    int_to_0 = total_int - cum_int
    D_num = np.exp(-int_to_0)
    # Normalize: D(1)/D(1) = 1, so at x_end=0, D should be 1
    # int_to_0 at last point should be ~0, so D_num[-1] ~ 1. Good.
    
    return f_num, D_num

def _objective(c, test_cosmos, a_test):
    """Compute max relative error over all test cosmologies and a values."""
    max_err_f = 0.0
    max_err_D = 0.0
    
    for (Om0, w0, wa) in test_cosmos:
        a_arr = a_test
        f_num, D_num = _numerical_f_and_D(a_arr, Om0, w0, wa)
        
        # Compute model values
        Om_arr = np.zeros_like(a_arr)
        for i, a in enumerate(a_arr):
            x = np.log(a)
            integ = (1.0 + w0) * x + wa * (x - a + 1.0)
            Om_m = Om0 * a**(-3)
            Om_de = (1.0 - Om0) * np.exp(-3.0 * integ)
            Om_arr[i] = Om_m / (Om_m + Om_de)
        
        f_model = _f_of(c, a_arr, Om_arr, w0, wa)
        
        # Relative errors
        valid = np.abs(f_num) > 1e-10
        if np.any(valid):
            err_f = np.abs(f_model[valid] - f_num[valid]) / np.abs(f_num[valid])
            max_err_f = max(max_err_f, np.max(err_f))
        
        valid_D = np.abs(D_num) > 1e-10
        if np.any(valid_D):
            # D_model: exp(-integral f_model dx)
            x_arr = np.log(a_arr)
            # Integrate f_model from each a to 1
            from scipy.integrate import cumulative_trapezoid
            cum_int_m = cumulative_trapezoid(f_model, x_arr, initial=0.0)
            total_int_m = cum_int_m[-1]
            int_to_0_m = total_int_m - cum_int_m
            D_model = np.exp(-int_to_0_m)
            
            err_D = np.abs(D_model[valid_D] - D_num[valid_D]) / np.abs(D_num[valid_D])
            max_err_D = max(max_err_D, np.max(err_D))
    
    return max(max_err_f, max_err_D)

def construct():
    """Return 6 coefficients for the growth-index formula."""
    # Physics-constrained affine: fix c1=0, c2=0 for standard q-dependence
    # Optimize c0, c3, c4, c5
    
    # Representative CPL test cosmologies (disjoint from validation)
    test_cosmos = [
        (0.30, -1.0, 0.0),    # LCDM
        (0.30, -0.5, 0.5),    # CPL positive wa
        (0.30, -1.5, 0.5),    # CPL negative w0
        (0.30, -0.8, -0.3),   # CPL negative wa
        (0.25, -1.0, 0.3),    # Low Om, positive wa
        (0.35, -1.2, 0.2),    # High Om
        (0.30, -1.0, 0.0),    # LCDM again (weight)
        (0.30, -0.9, 0.4),    # Moderate CPL
    ]
    
    # Test a values
    a_test = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
    
    # Precompute numerical solutions (expensive, do once)
    print(f"Precomputing numerical solutions... elapsed={_elapsed():.1f}s")
    num_solutions = []
    for (Om0, w0, wa) in test_cosmos:
        f_num, D_num = _numerical_f_and_D(a_test, Om0, w0, wa)
        num_solutions.append((f_num, D_num))
    print(f"Precomputation done. elapsed={_elapsed():.1f}s")
    
    # Objective function for optimizer
    def obj(c4):
        # c4 is [c0, c3, c4, c5], with c1=c2=0 fixed
        c = np.array([c4[0], 0.0, 0.0, c4[1], c4[2], c4[3]])
        max_err_f = 0.0
        max_err_D = 0.0
        
        for idx, (Om0, w0, wa) in enumerate(test_cosmos):
            f_num, D_num = num_solutions[idx]
            
            a_arr = a_test
            Om_arr = np.zeros_like(a_arr)
            for i, a in enumerate(a_arr):
                x = np.log(a)
                integ = (1.0 + w0) * x + wa * (x - a + 1.0)
                Om_m = Om0 * a**(-3)
                Om_de = (1.0 - Om0) * np.exp(-3.0 * integ)
                Om_arr[i] = Om_m / (Om_m + Om_de)
            
            f_model = _f_of(c, a_arr, Om_arr, w0, wa)
            
            valid = np.abs(f_num) > 1e-10
            if np.any(valid):
                err_f = np.abs(f_model[valid] - f_num[valid]) / np.abs(f_num[valid])
                max_err_f = max(max_err_f, np.max(err_f))
            
            valid_D = np.abs(D_num) > 1e-10
            if np.any(valid_D):
                x_arr = np.log(a_arr)
                from scipy.integrate import cumulative_trapezoid
                cum_int_m = cumulative_trapezoid(f_model, x_arr, initial=0.0)
                total_int_m = cum_int_m[-1]
                int_to_0_m = total_int_m - cum_int_m
                D_model = np.exp(-int_to_0_m)
                
                err_D = np.abs(D_model[valid_D] - D_num[valid_D]) / np.abs(D_num[valid_D])
                max_err_D = max(max_err_D, np.max(err_D))
        
        return max(max_err_f, max_err_D)
    
    # Initial guess: constant gamma ~ 0.55
    c0_init = 0.55
    c3_init = 0.0
    c4_init = 0.0
    c5_init = 0.0
    x0 = np.array([c0_init, c3_init, c4_init, c5_init])
    
    print(f"Starting optimization. elapsed={_elapsed():.1f}s")
    
    # Use Nelder-Mead for robustness (no gradient needed)
    best_c = x0.copy()
    best_obj = obj(x0)
    print(f"Initial objective: {best_obj:.6f}")
    
    # Multi-start with different initial points
    starts = [
        [0.55, 0.0, 0.0, 0.0],
        [0.50, 0.1, 0.0, 0.0],
        [0.60, -0.1, 0.0, 0.0],
        [0.55, 0.0, 0.1, 0.0],
        [0.55, 0.0, -0.1, 0.0],
        [0.55, 0.0, 0.0, 0.1],
        [0.55, 0.0, 0.0, -0.1],
        [0.52, 0.05, 0.05, 0.0],
    ]
    
    for i, x_start in enumerate(starts):
        if _time_left() < 5.0:
            break
        try:
            res = minimize(obj, x_start, method='Nelder-Mead', 
                          options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10,
                                   'adaptive': True})
            if res.fun < best_obj:
                best_obj = res.fun
                best_c = res.x
                print(f"Start {i}: improved to {best_obj:.6f} at c={best_c}")
        except Exception as e:
            print(f"Start {i} failed: {e}")
    
    # Final result: c = [c0, 0, 0, c3, c4, c5]
    c_final = np.array([best_c[0], 0.0, 0.0, best_c[1], best_c[2], best_c[3]])
    
    # Clamp to [-10, 10]
    c_final = np.clip(c_final, -10.0, 10.0)
    
    print(f"Final coefficients: {c_final}")
    print(f"Final objective: {best_obj:.6f}")
    print(f"Total elapsed: {_elapsed():.1f}s")
    
    return list(c_final)
