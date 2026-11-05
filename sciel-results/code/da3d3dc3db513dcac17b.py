"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80  # 80s safety margin before 90s kill
    
    def om(a, Om0, w0, wa):
        """Matter density parameter at scale factor a."""
        de = (1 - Om0) * a**(-3*(1+w0+wa*(1-a)))
        # More careful: for CPL, rho_de ~ a^(-3(1+w(a)))
        # Integrate: rho_de(a) = rho_de(1) * exp(-3*int_1^a (1+w(a'))/a' da')
        # For CPL: w(a) = w0 + wa*(1-a)
        # 1+w(a) = 1+w0+wa-wa*a
        # int_1^a (1+w0+wa-wa*a')/a' da' = (1+w0+wa)*ln(a) - wa*(a-1)
        # So rho_de(a) = (1-Om0) * a^(-(1+w0+wa)) * exp(wa*(a-1))
        # Let's use the standard CPL formula
        E2 = Om0 * a**(-3) + (1-Om0) * a**(-3*(1+w0)) * np.exp(3*wa*(a-1))
        return Om0 * a**(-3) / E2
    
    def f_ode(a, Om0, w0, wa):
        """Exact growth rate from ODE integration."""
        if a <= 0:
            return 0.0
        om_a = om(a, Om0, w0, wa)
        H2 = Om0 * a**(-3) + (1-Om0) * a**(-3*(1+w0)) * np.exp(3*wa*(a-1))
        # f = d ln D / d ln a = Om(a)^(gamma)
        # From ODE: f' + f^2 + (3/2)(1+q)f - 3/2 = 0 where q = 1-Om
        # We need to solve the ODE numerically
        # Use a simple approach: integrate from a=1 backwards
        if a >= 0.999:
            return om_a ** 0.55  # approximate for near 1
        
        # Solve ODE from a=1 to a
        # Let y = [f, D], d/dln(a)
        def ode(x, y):
            a_x = np.exp(x)
            om_x = om(a_x, Om0, w0, wa)
            f_x = y[0]
            D_x = y[1]
            dfdx = -f_x**2 - (3/2)*(1-om_x)*f_x + 3/2
            dDdx = f_x * D_x
            return [dfdx, dDdx]
        
        x0 = 0.0  # ln(1) = 0
        x1 = np.log(a)
        y0 = [om(1.0, Om0, w0, wa)**0.55, 1.0]
        
        sol = solve_ivp(ode, [x0, x1], y0, rtol=1e-8, atol=1e-10, dense_output=True)
        if sol.success and len(sol.y[0]) > 0:
            return sol.y[0, -1]
        return om_a ** 0.55
    
    def gamma_from_coeffs(c, a, Om0, w0, wa):
        q = 1 - om(a, Om0, w0, wa)
        u = 1 + w0 + wa*(1-a)
        return c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q
    
    def f_approx(c, a, Om0, w0, wa):
        g = gamma_from_coeffs(c, a, Om0, w0, wa)
        return om(a, Om0, w0, wa) ** g
    
    def D_approx(c, a, Om0, w0, wa):
        """Normalized growth factor D(a)/D(1)."""
        if a >= 0.999:
            return 1.0
        # Integrate -int_0^ln(a) f(exp(x)) dx
        from scipy.integrate import quad
        def integrand(x):
            a_x = np.exp(x)
            return f_approx(c, a_x, Om0, w0, wa)
        val, _ = quad(integrand, 0, np.log(a), limit=50)
        return np.exp(-val)
    
    def D_ode(a, Om0, w0, wa):
        """Exact normalized growth from ODE."""
        if a >= 0.999:
            return 1.0
        def ode(x, y):
            a_x = np.exp(x)
            om_x = om(a_x, Om0, w0, wa)
            f_x = y[0]
            D_x = y[1]
            dfdx = -f_x**2 - (3/2)*(1-om_x)*f_x + 3/2
            dDdx = f_x * D_x
            return [dfdx, dDdx]
        
        x0 = 0.0
        x1 = np.log(a)
        y0 = [om(1.0, Om0, w0, wa)**0.55, 1.0]
        
        sol = solve_ivp(ode, [x0, x1], y0, rtol=1e-8, atol=1e-10)
        if sol.success and len(sol.y[1]) > 0:
            return sol.y[1, -1]
        return 1.0
    
    # Validation cosmologies (typical ranges)
    test_models = [
        (0.3, -1.0, 0.0),   # LCDM-like
        (0.3, -1.0, 0.5),   # CPL with wa>0
        (0.3, -0.5, -0.5),  # w0 > -1, wa < 0
        (0.25, -1.0, 0.3),  # different Om0
        (0.35, -1.2, 0.4),  # more negative w0
        (0.3, -0.8, -0.3),  # quintessence-like
    ]
    a_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    
    def objective(c):
        max_err = 0.0
        for Om0, w0, wa in test_models:
            for a in a_values:
                # Check gamma bounds
                g = gamma_from_coeffs(c, a, Om0, w0, wa)
                if g < 0.1 or g > 1.2:
                    return 1e6  # penalty
                
                f_exact = f_ode(a, Om0, w0, wa)
                f_app = f_approx(c, a, Om0, w0, wa)
                if abs(f_exact) > 1e-10:
                    err_f = abs(f_app - f_exact) / abs(f_exact)
                else:
                    err_f = 0.0
                
                D_exact = D_ode(a, Om0, w0, wa)
                D_app = D_approx(c, a, Om0, w0, wa)
                if abs(D_exact) > 1e-10:
                    err_D = abs(D_app - D_exact) / abs(D_exact)
                else:
                    err_D = 0.0
                
                max_err = max(max_err, err_f, err_D)
        
        return 100.0 * max_err
    
    # Start from a reasonable initial guess
    c0 = [0.55, 0.0, 0.0, -0.5, 0.0, 0.0]
    
    try:
        result = minimize(objective, c0, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
        
        if time.time() > deadline - 5:
            # Return current best if close to deadline
            best_c = result.x if hasattr(result, 'x') else c0
        else:
            best_c = result.x
        
        # Ensure coefficients are in [-10, 10]
        best_c = np.clip(best_c, -10, 10)
        
        return list(best_c)
        
    except Exception:
        # Fallback to initial guess
        return [0.55, 0.0, 0.0, -0.5, 0.0, 0.0]

# EVOLVE-BLOCK-END
