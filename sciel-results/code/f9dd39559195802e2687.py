import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time

def construct():
    """Optimize 6 coefficients for the growth-index formula."""
    t0 = time.time()
    
    # Reference cosmologies to fit
    models = [
        (0.30, -1.0, 0.0),   # LCDM
        (0.30, -0.9, -0.3),  # Quintessence
        (0.30, -1.1, 0.3),   # Phantom
        (0.35, -1.0, 0.0),   # High matter
        (0.25, -1.0, 0.0),   # Low matter
        (0.30, -0.8, -0.5),  # More quintessence
        (0.30, -1.2, 0.5),   # More phantom
    ]
    
    a_eval = np.linspace(0.1, 1.0, 50)
    
    def compute_om(a, om0, w0, wa):
        """Compute Omega_m(a) for CPL dark energy."""
        w = w0 + wa * (1 - a)
        # EdS scaling: Om(a) = Om0 * a^-3 * (1+w) / (1+w0) ... actually need proper E(z)
        # For CPL: H^2/H0^2 = Om0*a^-3 + Ode0*a^(-3*(1+w0+wa)) * a^(-3*wa)
        # Simpler: Om(a) = Om0 * a^-3 / E(a)^2 where E(a) = H(a)/H0
        # Let me use the standard form
        w_eff = w0 + wa * (1 - a)
        # E(a)^2 = Om0 * a^-3 + (1-Om0) * exp(3*wa*(1-a)) * a^(-3*(1+w0))
        # Actually for CPL: Ode(a) = Ode0 * a^(-3*(1+w0)) * exp(3*wa*(1-a))
        ome0 = om0
        ode0 = 1.0 - om0
        a_term = a ** (-3 * (1 + w0))
        exp_term = np.exp(3 * wa * (1 - a))
        ode_a = ode0 * a_term * exp_term
        e2 = ome0 * a**(-3) + ode_a
        om_a = ome0 * a**(-3) / e2
        return om_a
    
    def gamma_of_a(c, a, om0, w0, wa):
        om_a = compute_om(a, om0, w0, wa)
        q = 1.0 - om_a
        u = 1.0 + w0 + wa * (1 - a)
        gamma = c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q
        return gamma, om_a, q, u
    
    def f_of_a(c, a, om0, w0, wa):
        gamma, om_a, q, u = gamma_of_a(c, a, om0, w0, wa)
        # f = Om(a)^gamma
        f = om_a ** gamma
        return f
    
    def dlnD_of_a(c, a, om0, w0, wa):
        """Compute ln(D(a)/D(1)) = -integral from ln(a) to 0 of f(exp(x)) dx"""
        # Numerical integration
        npts = 200
        x_vals = np.linspace(np.log(a), 0.0, npts)
        a_vals = np.exp(x_vals)
        f_vals = np.array([f_of_a(c, ai, om0, w0, wa) for ai in a_vals])
        # trapezoidal
        integral = np.trapz(f_vals, x_vals)
        return -integral
    
    def reference_f_and_dlnD(om0, w0, wa, a_eval):
        """Solve the growth ODE for reference."""
        def ode(x, y):
            a = np.exp(x)
            om_a = compute_om(a, om0, w0, wa)
            # f = dlnD/dln a
            # f' = 1.5*Om - f^2 - (1.5 + H'/H) * f
            # H'/H = dlnH/dln a = (1/2) dlnE2/dln a
            # E2 = Om0*a^-3 + Ode0*a^(-3*(1+w0))*exp(3*wa*(1-a))
            # dlnE2/dln a = (-3*Om0*a^-3 + Ode0*(-3*(1+w0))*a^(-3*(1+w0))*exp(3*wa*(1-a)) + Ode0*a^(-3*(1+w0))*exp(3*wa*(1-a))*(-3*wa*(-1))) / E2
            # = (-3*Om0*a^-3 - 3*(1+w0)*Ode0*a^(-3*(1+w0))*exp(3*wa*(1-a)) + 3*wa*Ode0*a^(-3*(1+w0))*exp(3*wa*(1-a))) / E2
            ome0 = om0
            ode0 = 1.0 - om0
            a_term = a**(-3)
            ode_term_base = ode0 * a**(-3*(1+w0)) * np.exp(3*wa*(1-a))
            e2 = ome0 * a_term + ode_term_base
            dlnE2_dlna = (-3*ome0*a_term - 3*(1+w0)*ode_term_base + 3*wa*ode_term_base) / e2
            dlnH_dlna = 0.5 * dlnE2_dlna
            
            f = y[0]
            fp = 1.5*om_a - f*f - (1.5 + dlnH_dlna) * f
            return [fp]
        
        # Integrate from x=0 (a=1) where f~1 backward
        sol = solve_ivp(ode, [0.0, np.log(a_eval[0])], [1.0], 
                       t_eval=a_eval[::-1] if a_eval[0] < 1 else a_eval,
                       rtol=1e-10, atol=1e-12, method='RK45')
        
        f_ref = sol.y[0]
        
        # Compute dlnD by integrating f
        dlnD_ref = np.zeros_like(a_eval)
        for i in range(len(a_eval)):
            if i == 0:
                dlnD_ref[i] = 0.0
            else:
                # dlnD(a_i) = -integral from ln(a_i) to 0 of f dx
                # We have f at a_eval points
                x_vals = np.log(a_eval)
                # integrate from x_i to 0
                mask = x_vals >= x_vals[i]
                if np.sum(mask) < 2:
                    dlnD_ref[i] = 0.0
                else:
                    dlnD_ref[i] = -np.trapz(f_ref[mask], x_vals[mask])
        
        return f_ref, dlnD_ref
    
    # Precompute reference solutions
    refs = []
    for (om0, w0, wa) in models:
        f_ref, dlnD_ref = reference_f_and_dlnD(om0, w0, wa, a_eval)
        refs.append((om0, w0, wa, f_ref, dlnD_ref))
    
    def objective(c):
        elapsed = time.time() - t0
        if elapsed > 50:  # safety margin
            return 1e10
        
        max_err = 0.0
        gamma_violation = 0.0
        
        for (om0, w0, wa, f_ref, dlnD_ref) in refs:
            # Compute f for this model
            f_pred = np.array([f_of_a(c, a, om0, w0, wa) for a in a_eval])
            
            # Check gamma constraint
            for a in a_eval:
                gamma, om_a, q, u = gamma_of_a(c, a, om0, w0, wa)
                if gamma < 0.1 or gamma > 1.2:
                    gamma_violation += abs(gamma - np.clip(gamma, 0.1, 1.2))
            
            # Relative error in f
            rel_err_f = np.abs(f_pred - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
            max_err = max(max_err, np.max(rel_err_f))
            
            # Compute dlnD prediction
            dlnD_pred = np.array([dlnD_of_a(c, a, om0, w0, wa) for a in a_eval])
            rel_err_d = np.abs(dlnD_pred - dlnD_ref) / np.maximum(np.abs(dlnD_ref), 1e-10)
            max_err = max(max_err, np.max(rel_err_d))
        
        # Penalty for gamma violations
        total_err = 100.0 * max_err + 1000.0 * gamma_violation
        return total_err
    
    # Optimize
    c0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    bounds = [(-10, 10)] * 6
    
    best_c = c0.copy()
    best_val = objective(c0)
    
    # Try L-BFGS-B
    try:
        res = minimize(objective, c0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6})
        if res.fun < best_val:
            best_c = res.x
            best_val = res.fun
    except:
        pass
    
    # Also try Nelder-Mead from a few starting points
    starts = [
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.1, -0.05, 0.0, 0.0, 0.0],
        [0.6, -0.1, 0.05, 0.0, 0.0, 0.0],
        [0.55, 0.05, 0.0, 0.05, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.1, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.1],
    ]
    
    for start in starts:
        elapsed = time.time() - t0
        if elapsed > 45:
            break
        try:
            res = minimize(objective, start, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            # Check bounds
            if all(-10 <= x <= 10 for x in res.x):
                if res.fun < best_val:
                    best_c = res.x
                    best_val = res.fun
        except:
            pass
    
    # Ensure bounds
    best_c = np.clip(best_c, -10, 10)
    
    return list(best_c)
