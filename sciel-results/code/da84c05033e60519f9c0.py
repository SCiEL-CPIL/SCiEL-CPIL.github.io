"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # 85s safety margin before 90s kill
    
    def om(a, Om0, w0, wa):
        """Matter density parameter at scale factor a."""
        E2 = Om0 * a**(-3) + (1-Om0) * a**(-3*(1+w0)) * np.exp(3*wa*(a-1))
        return Om0 * a**(-3) / E2
    
    def compute_references(test_models, a_values):
        """Pre-compute exact f and D for all test points."""
        refs = {}
        for Om0, w0, wa in test_models:
            key = (Om0, w0, wa)
            f_refs = np.zeros(len(a_values))
            D_refs = np.zeros(len(a_values))
            
            # Solve ODE once for all a_values
            def ode(x, y):
                a_x = np.exp(x)
                om_x = om(a_x, Om0, w0, wa)
                f_x = y[0]
                D_x = y[1]
                dfdx = -f_x**2 - (3/2)*(1-om_x)*f_x + 3/2
                dDdx = f_x * D_x
                return [dfdx, dDdx]
            
            # Use log-spaced grid for better resolution
            x_min = np.log(min(a_values))
            x_grid = np.linspace(0, x_min, 200)
            y0 = [om(1.0, Om0, w0, wa)**0.55, 1.0]
            
            sol = solve_ivp(ode, [0, x_min], y0, rtol=1e-10, atol=1e-12, 
                          t_eval=x_grid, dense_output=True)
            
            if sol.success:
                for i, a in enumerate(a_values):
                    if a >= 0.999:
                        f_refs[i] = om(a, Om0, w0, wa)**0.55
                        D_refs[i] = 1.0
                    else:
                        x_a = np.log(a)
                        # Interpolate from dense output
                        y_a = sol.sol(x_a)
                        f_refs[i] = y_a[0]
                        D_refs[i] = y_a[1]
            
            refs[key] = (f_refs, D_refs)
        
        return refs
    
    def gamma_from_coeffs(c, a, Om0, w0, wa):
        q = 1 - om(a, Om0, w0, wa)
        u = 1 + w0 + wa*(1-a)
        return c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q
    
    def f_approx(c, a, Om0, w0, wa):
        g = gamma_from_coeffs(c, a, Om0, w0, wa)
        return om(a, Om0, w0, wa) ** g
    
    def D_approx(c, a, Om0, w0, wa, f_cache=None):
        """Normalized growth factor D(a)/D(1)."""
        if a >= 0.999:
            return 1.0
        from scipy.integrate import quad
        def integrand(x):
            a_x = np.exp(x)
            return f_approx(c, a_x, Om0, w0, wa)
        val, _ = quad(integrand, 0, np.log(a), limit=50)
        return np.exp(-val)
    
    # Validation cosmologies
    test_models = [
        (0.3, -1.0, 0.0),
        (0.3, -1.0, 0.5),
        (0.3, -0.5, -0.5),
        (0.25, -1.0, 0.3),
        (0.35, -1.2, 0.4),
        (0.3, -0.8, -0.3),
    ]
    a_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    
    # Pre-compute references
    refs = compute_references(test_models, a_values)
    
    def objective(c):
        max_err = 0.0
        for Om0, w0, wa in test_models:
            key = (Om0, w0, wa)
            f_refs, D_refs = refs[key]
            for i, a in enumerate(a_values):
                g = gamma_from_coeffs(c, a, Om0, w0, wa)
                if g < 0.1 or g > 1.2:
                    return 1e6
                
                f_app = f_approx(c, a, Om0, w0, wa)
                f_exact = f_refs[i]
                if abs(f_exact) > 1e-10:
                    err_f = abs(f_app - f_exact) / abs(f_exact)
                else:
                    err_f = 0.0
                
                D_app = D_approx(c, a, Om0, w0, wa)
                D_exact = D_refs[i]
                if abs(D_exact) > 1e-10:
                    err_D = abs(D_app - D_exact) / abs(D_exact)
                else:
                    err_D = 0.0
                
                max_err = max(max_err, err_f, err_D)
        
        return 100.0 * max_err
    
    # Start from a reasonable initial guess
    c0 = np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])
    
    best_c = c0.copy()
    best_obj = objective(c0)
    
    try:
        # Use Nelder-Mead with adaptive options
        result = minimize(objective, c0, method='Nelder-Mead', 
                          options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        
        if result.fun < best_obj:
            best_c = result.x
            best_obj = result.fun
        
        # Try local refinement with L-BFGS-B
        if time.time() < deadline - 10:
            result2 = minimize(objective, best_c, method='L-BFGS-B', 
                              bounds=[(-10, 10)]*6,
                              options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if result2.fun < best_obj:
                best_c = result2.x
                best_obj = result2.fun
        
    except Exception:
        pass
    
    best_c = np.clip(best_c, -10, 10)
    
    return list(best_c)

# EVOLVE-BLOCK-END
