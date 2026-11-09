import math
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    DEADLINE = 85.0  # Safety margin

    # ---------- Reference Cosmology Functions ----------
    
    def h2_of_a(a, Om, w0, wa):
        # H^2/H0^2 = Om a^-3 + (1-Om) exp(3*wa*(1-a))
        # a is array
        return Om * a ** (-3.0) + (1.0 - Om) * np.exp(3.0 * wa * (1.0 - a))

    def om_m_of_a(a, Om, w0, wa):
        h2 = h2_of_a(a, Om, w0, wa)
        return Om * a ** (-3.0) / h2

    def growth_rhs(x, y, yp, Om, w0, wa):
        # x = ln a
        a = math.exp(x)
        w = w0 + wa * (1.0 - a)
        dw_dx = wa * a  # d(1+w)/dx = wa * d(1-a)/dx = wa * (-a * -1) ? 
        # Wait. w = w0 + wa(1-a). 
        # dw/da = -wa.
        # dx/da = 1/a.
        # dw/dx = (dw/da) * (da/dx) = (-wa) * a = -wa*a.
        # Let's re-verify the sign.
        # w(a) = w0 + wa(1-a).
        # dw/da = -wa.
        # x = ln a -> a = e^x -> da/dx = a.
        # dw/dx = dw/da * da/dx = -wa * a.
        # The previous code had dw_dx = wa * a. This is a SIGN ERROR in the previous code if w = w0 + wa(1-a).
        # Let's check standard CPL: w(a) = w0 + wa(1-a).
        # If wa > 0, w increases as a decreases (goes to past). w(0) = w0 + wa.
        # dw/da = -wa.
        # So dw/dx = -wa * a.
        
        # Equation: d(y')/dx = -[3/2 + 1.5(1+w) + dw/dx] y' + (3/2) Om y
        # Let's re-derive quickly.
        # Hubble eq: H^2 = H0^2 [Om a^-3 + Ode a^-3(1+w)exp(3wa(1-a))]
        # Matter: dm/dx = -3 Om y
        # DE: dde/dx = -3(1+w) Ode y
        # Poisson: d^2y/dx^2 + [3/2 + 1.5(1+w) + d(1+w)/dx] dy/dx - 3/2 Om y = 0
        # So d(yp)/dx = -[3/2 + 1.5(1+w) + dw/dx] yp + 1.5 Om y
        # dw/dx = -wa * a.
        # So term is -[3/2 + 1.5(1+w) - wa*a] yp + ...
        # Previous code: dw_dx = wa * a. 
        # If previous code was correct, maybe my derivation is wrong?
        # Let's check a known result. For w=const, dw/dx=0.
        # For CPL, w = w0 + wa(1-a).
        # Reference: Linder (2005).
        # The term in the ODE is usually written with d(1+w)/dx.
        # d(1+w)/dx = d(w0 + 1 + wa - wa*a)/dx = -wa * da/dx = -wa * a.
        # So the coefficient of y' is -(3/2 + 1.5(1+w) - wa*a).
        # The previous code had: dw_dx = wa * a. And used -( ... + dw_dx).
        # So previous code effectively used dw/dx = wa*a.
        # If wa > 0, w increases in past.
        # Let's stick to the derivation: dw/dx = -wa*a.
        # I will use the derived sign. If the previous code scored 1.0, it might have been correct or the error was small/cancelled.
        # However, "Improve" suggests the previous code might have suboptimal coefficients due to this or just poor search.
        # I will use the correct physical sign: dw_dx = -wa * a.

        Om_a = om_m_of_a(a, Om, w0, wa)
        dyp_dx = -(1.5 + 1.5 * (1.0 + w) + (-wa * a)) * yp + 1.5 * Om_a * y
        return (yp, dyp_dx)

    def integrate_growth(Om, w0, wa, xs):
        # RK4 from x0 to each x in xs
        n = len(xs)
        Y = np.empty((n, 2))
        Y[0, 0] = 1.0
        Y[0, 1] = 0.5  # Arbitrary start, ratios matter
        
        for i in range(1, n):
            x0 = xs[i - 1]
            x1 = xs[i]
            h = x1 - x0
            y0, yp0 = Y[i - 1, 0], Y[i - 1, 1]
            
            k1y, k1yp = growth_rhs(x0, y0, yp0, Om, w0, wa)
            k2y, k2yp = growth_rhs(x0 + 0.5 * h, y0 + 0.5 * h * k1y, yp0 + 0.5 * h * k1yp, Om, w0, wa)
            k3y, k3yp = growth_rhs(x0 + 0.5 * h, y0 + 0.5 * h * k2y, yp0 + 0.5 * h * k2yp, Om, w0, wa)
            k4y, k4yp = growth_rhs(x0 + h, y0 + h * k3y, yp0 + h * k3yp, Om, w0, wa)
            
            Y[i, 0] = y0 + (h / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)
            Y[i, 1] = yp0 + (h / 6.0) * (k1yp + 2 * k2yp + 2 * k3yp + k4yp)
            
        f = Y[:, 1] / Y[:, 0]
        D = Y[:, 0] / Y[-1, 0]
        return f, D

    # ---------- Model Ansätze ----------
    
    def model_f(c, a, Om, w0, wa):
        c0, c1, c2, c3, c4, c5 = c
        Om_a = om_m_of_a(a, Om, w0, wa)
        q = 1.0 - Om_a
        u = 1.0 + w0 + wa * (1.0 - a)
        gamma = c0 + c1 * q + c2 * q * q + c3 * u + c4 * wa * a + c5 * u * q
        # Clamp gamma to avoid numerical issues in power, though constraint is [0.1, 1.2]
        gamma = np.clip(gamma, 0.01, 10.0)
        return Om_a ** gamma

    def model_D(c, a_grid, Om, w0, wa):
        # D/D(1) = exp(-int_{ln a}^{0} f(x) dx)
        xs = np.log(a_grid)
        fs = model_f(c, a_grid, Om, w0, wa)
        
        # Cumulative integral from x_i to 0
        # xs is ascending (a ascending). x=0 at a=1 (last point).
        # We need integral from x_i to 0.
        # Since xs are negative (a<1) and ascending, x_i < ... < x_N=0.
        # Integral from x_i to 0 = sum of trapezoids from i to N-1.
        
        if len(a_grid) == 1:
            return np.array([1.0])
            
        dx = np.diff(xs)  # positive
        trap = 0.5 * (fs[:-1] + fs[1:]) * dx
        
        cum = np.zeros(len(xs))
        # cum[N-1] = 0
        for j in range(len(xs) - 2, -1, -1):
            cum[j] = cum[j + 1] + trap[j]
            
        D = np.exp(-cum)
        return D

    # ---------- Grids ----------
    
    # Cosmologies for fitting
    # We want to cover the validation space well.
    Om_grid = np.array([0.20, 0.25, 0.30, 0.35, 0.40])
    w0_grid = np.array([-1.5, -1.2, -1.0, -0.8, -0.5, 0.0])
    wa_grid = np.array([-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5])
    
    # a grid for evaluation/integration
    # Dense grid for integration accuracy
    a_int = np.linspace(0.05, 1.0, 500)
    
    # Precompute reference data for a subset of cosmologies
    # To save time, we select a representative grid for fitting
    fit_cosmos = []
    # Sample every other point to keep count manageable (~100-200 models)
    for Om in Om_grid:
        for w0 in w0_grid:
            for wa in wa_grid:
                fit_cosmos.append((Om, w0, wa))
                
    # Reduce set if too large
    if len(fit_cosmos) > 150:
        # Simple stride
        fit_cosmos = fit_cosmos[::2]

    ref_data = {}
    for (Om, w0, wa) in fit_cosmos:
        if time.time() > DEADLINE - 10.0:
            break
        try:
            f_ref, D_ref = integrate_growth(Om, w0, wa, np.log(a_int))
            ref_data[(Om, w0, wa)] = (f_ref, D_ref)
        except Exception:
            pass

    if not ref_data:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # ---------- Objective Functions ----------
    
    # True Objective: Max relative error
    def objective_true(c):
        c = np.asarray(c)
        max_err = 0.0
        for key, (f_ref, D_ref) in ref_data.items():
            Om, w0, wa = key
            f_mod = model_f(c, a_int, Om, w0, wa)
            D_mod = model_D(c, a_int, Om, w0, wa)
            
            # Relative errors
            err_f = np.abs(f_mod - f_ref) / np.maximum(np.abs(f_ref), 1e-8)
            err_D = np.abs(D_mod - D_ref) / np.maximum(np.abs(D_ref), 1e-8)
            
            # Check gamma constraint
            Om_a = om_m_of_a(a_int, Om, w0, wa)
            q = 1.0 - Om_a
            u = 1.0 + w0 + wa * (1.0 - a_int)
            gamma = c[0] + c[1]*q + c[2]*q*q + c[3]*u + c[4]*wa*a_int + c[5]*u*q
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                # Penalty for violating constraints
                max_err = max(max_err, 10.0)
                
            max_err = max(max_err, float(np.max(err_f)), float(np.max(err_D)))
        return max_err

    # Smooth Surrogate: Log-Sum-Exp of relative errors
    # This makes the objective differentiable, aiding gradient-based optimizers.
    def objective_smooth(c, beta=50.0):
        c = np.asarray(c)
        err_list = []
        gamma_violation = 0.0
        
        for key, (f_ref, D_ref) in ref_data.items():
            Om, w0, wa = key
            f_mod = model_f(c, a_int, Om, w0, wa)
            D_mod = model_D(c, a_int, Om, w0, wa)
            
            err_f = np.abs(f_mod - f_ref) / np.maximum(np.abs(f_ref), 1e-8)
            err_D = np.abs(D_mod - D_ref) / np.maximum(np.abs(D_ref), 1e-8)
            
            err_list.append(err_f)
            err_list.append(err_D)
            
            # Check gamma
            Om_a = om_m_of_a(a_int, Om, w0, wa)
            q = 1.0 - Om_a
            u = 1.0 + w0 + wa * (1.0 - a_int)
            gamma = c[0] + c[1]*q + c[2]*q*q + c[3]*u + c[4]*wa*a_int + c[5]*u*q
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                gamma_violation = 10.0
                
        if gamma_violation > 0:
            return gamma_violation
            
        all_errs = np.concatenate(err_list)
        # Log-sum-exp
        max_e = np.max(all_errs)
        if max_e < 1e-10:
            return 0.0
        lse = max_e + np.log(np.sum(np.exp(all_errs - max_e))) / beta
        return lse

    # ---------- Optimization ----------
    
    best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_true_val = objective_true(best_c)
    
    # Initial guesses
    starts = [
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.50, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.60, -0.1, 0.0, 0.0, 0.0, 0.0],
        [0.55, 0.05, -0.05, 0.0, 0.0, 0.0],
        [0.50, 0.0, 0.0, 0.05, 0.0, 0.0],
        [0.55, 0.1, 0.1, -0.1, 0.0, 0.0],
        [0.55, -0.1, -0.1, 0.1, 0.0, 0.0],
    ]
    
    # Phase 1: Global Search with Smooth Objective
    for s in starts:
        if time.time() > DEADLINE - 15.0:
            break
        s = np.asarray(s)
        try:
            # Use BFGS or L-BFGS-B for smooth objective
            res = minimize(objective_smooth, s, method='L-BFGS-B',
                           options={'maxiter': 500, 'gtol': 1e-6})
            if res.fun < 1.0: # Filter out bad solutions
                c_candidate = res.x
                # Evaluate with true objective
                val_candidate = objective_true(c_candidate)
                if val_candidate < best_true_val:
                    best_true_val = val_candidate
                    best_c = c_candidate
        except Exception:
            pass

    # Phase 2: Local Polish with True Objective (Nelder-Mead)
    # Use the best found so far and a few random perturbations
    polish_starts = [best_c]
    for i in range(5):
        if time.time() > DEADLINE - 10.0:
            break
        noise = np.random.normal(0, 0.05, 6)
        polish_starts.append(best_c + noise)
        
    for s in polish_starts:
        if time.time() > DEADLINE - 5.0:
            break
        s = np.asarray(s)
        try:
            res = minimize(objective_true, s, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-8})
            if res.fun < best_true_val:
                best_true_val = res.fun
                best_c = res.x
        except Exception:
            pass

    # Final Clip and Constraint Check
    best_c = np.clip(best_c, -10.0, 10.0)
    
    # Verify constraints one last time
    valid = True
    for Om in [0.2, 0.4]:
        for w0 in [-1.5, 0.0]:
            for wa in [-0.5, 0.5]:
                a_test = np.linspace(0.1, 1.0, 20)
                Om_a = om_m_of_a(a_test, Om, w0, wa)
                q = 1.0 - Om_a
                u = 1.0 + w0 + wa * (1.0 - a_test)
                g = best_c[0] + best_c[1]*q + best_c[2]*q*q + best_c[3]*u + best_c[4]*wa*a_test + best_c[5]*u*q
                if np.any(g < 0.1) or np.any(g > 1.2):
                    valid = False
                    break
            if not valid: break
        if not valid: break
        
    if not valid:
        # Fallback to constant gamma which is always safe
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    return [float(x) for x in best_c]

# EVOLVE-BLOCK-END
