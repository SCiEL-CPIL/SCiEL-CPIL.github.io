"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time

def _omega_m(a, Om0, w0, wa):
    """Matter density parameter at scale factor a for CPL dark energy."""
    a = np.asarray(a, dtype=float)
    OmL0 = 1.0 - Om0
    # Dark energy density: rho_DE ∝ a^{-3(1+w0+wa)} but with CPL: w = w0 + wa*(1-a)
    # Actually for CPL: w(a) = w0 + wa*(1-a), so rho_DE ∝ exp(-3*∫(1+w)da/a)
    # ∫(1+w0+wa*(1-a))/a da = ∫(1+w0)/a + wa*(1/a - 1) da
    # = (1+w0)*ln(a) + wa*(ln(a) - a)
    # So rho_DE/rho_DE0 = a^{-3(1+w0)} * exp(-3*wa*(ln(a)-a))
    # Wait, let me be more careful.
    # w(a) = w0 + wa*(1-a)
    # 1+w(a) = 1+w0+wa-wa*a
    # ∫_1^a (1+w(a'))/a' da' = ∫_1^a (1+w0+wa)/a' da' - wa*∫_1^a da'
    # = (1+w0+wa)*ln(a) - wa*(a-1)
    # So rho_DE/rho_DE0 = a^{-3(1+w0+wa)} * exp(3*wa*(a-1))
    # Hmm, let me just use the standard form.
    # Actually the problem states u = 1+w0+wa*(1-a)
    # And the ODE reference is "tightly integrated general-relativistic second-order ODE"
    # Let me use the standard CPL parameterization.
    
    # Standard CPL: w(a) = w0 + wa*(1-a)
    # rho_DE(a)/rho_DE(1) = a^{-3(1+w0+wa)} * exp(3*wa*(1-a))
    # Wait no. Let me derive:
    # d(rho_DE)/rho_DE = -3(1+w(a)) da/a
    # = -3(1+w0+wa-wa*a)/a da
    # = -3(1+w0+wa)/a da + 3*wa da
    # Integrate from 1 to a:
    # ln(rho_DE(a)/rho_DE(1)) = -3(1+w0+wa)*ln(a) + 3*wa*(a-1)
    # rho_DE(a)/rho_DE(1) = a^{-3(1+w0+wa)} * exp(3*wa*(a-1))
    
    # But actually, the problem says u = 1+w0+wa*(1-a), which is just 1+w(a)
    # Let me use a numerical approach or the standard formula.
    
    # For simplicity and correctness, let's use:
    # E(a)^2 = Om0*a^{-3} + OmL0 * a^{-3(1+w0+wa)} * exp(3*wa*(1-a))
    # Hmm, this is getting complicated. Let me use a cleaner form.
    
    # Actually, re-reading: "u=1+w0+wa*(1-a)" is just 1+w(a)
    # The standard CPL dark energy density:
    # rho_DE(a) = rho_DE(1) * a^{-3(1+w0+wa)} * exp(3*wa*(a-1))
    
    # Let me just compute it properly:
    # E(a)^2 = H(a)^2/H0^2 = Om0*a^{-3} + (1-Om0)*exp(-3*∫_1^a (1+w(a'))/a' da')
    # ∫_1^a (1+w0+wa*(1-a'))/a' da' = ∫_1^a [(1+w0+wa)/a' - wa] da'
    # = (1+w0+wa)*ln(a) - wa*(a-1)
    
    integral = (1.0 + w0 + wa) * np.log(a) - wa * (a - 1.0)
    rho_DE_ratio = np.exp(-3.0 * integral)
    E2 = Om0 * a**(-3) + (1.0 - Om0) * rho_DE_ratio
    return Om0 * a**(-3) / E2

def _reference_f(a, Om0, w0, wa):
    """Compute the reference growth rate f from the GR ODE."""
    # The growth factor D(a) satisfies:
    # D'' + [3/(a*E)] D' - [3*Om(a)/(2*a^2*E^2)] D = 0  (in terms of a)
    # Or in terms of ln(a) = x: d²D/dx² + [1.5/E(a)] dD/dx - [1.5*Om(a)/E(a)] D = 0
    # where E(a) = H(a)/H0
    # 
    # f = d ln D / d ln a = (1/D) dD/dx
    # 
    # We solve the ODE numerically and extract f.
    
    a_arr = np.asarray(a, dtype=float)
    x_arr = np.log(a_arr)
    
    # Define E(a)
    def E_of_a(a_val):
        a_val = np.asarray(a_val, dtype=float)
        integral = (1.0 + w0 + wa) * np.log(a_val) - wa * (a_val - 1.0)
        rho_DE_ratio = np.exp(-3.0 * integral)
        E2 = Om0 * a_val**(-3) + (1.0 - Om0) * rho_DE_ratio
        return np.sqrt(E2)
    
    def Om_of_a(a_val):
        E2 = Om0 * a_val**(-3) + (1.0 - Om0) * np.exp(-3.0 * ((1.0 + w0 + wa) * np.log(a_val) - wa * (a_val - 1.0)))
        return Om0 * a_val**(-3) / E2
    
    # ODE in x = ln(a): y = [D, dD/dx]
    # D'' + [1.5/E] D' - [1.5*Om/E] D = 0
    def odefun(x, y):
        a_val = np.exp(x)
        E_val = E_of_a(a_val)
        Om_val = Om_of_a(a_val)
        D, Dp = y
        Dpp = -1.5/E_val * Dp + 1.5*Om_val/E_val * D
        return [Dp, Dpp]
    
    # Initial conditions at a=1 (x=0): D(1)=1, and we need D'(1)
    # For early times (matter dominated), D ∝ a, so D' = 1 at a=1 in matter dom
    # But more generally, we can use the fact that for a << 1, D ∝ a
    # Let's integrate from a small value where D ≈ a
    # Actually, let's use a=1e-3 as initial point where D ≈ a, D' ≈ 1
    
    x_init = np.log(1e-3)
    y_init = [1e-3, 1.0]  # D ≈ a, dD/dln(a) ≈ 1
    
    # Integrate from x_init to max(x_arr)
    x_max = max(0.0, np.max(x_arr))
    x_eval = np.linspace(x_init, x_max, 1000)
    
    sol = solve_ivp(odefun, [x_init, x_max], y_init, t_eval=x_eval, rtol=1e-10, atol=1e-12, method='DOP853')
    
    if not sol.success:
        # Fallback: use constant gamma
        gamma = 0.55
        return Om_of_a(a_arr)**gamma
    
    # Interpolate to get D and D' at our desired a values
    D_vals = np.interp(x_arr, sol.t, sol.y[0])
    Dp_vals = np.interp(x_arr, sol.t, sol.y[1])
    
    # f = D'/D
    f_vals = Dp_vals / D_vals
    
    return f_vals

def _compute_error(coeffs, Om0, w0, wa, a_values):
    """Compute max relative error for given coefficients."""
    c0, c1, c2, c3, c4, c5 = coeffs
    
    a_arr = np.asarray(a_values, dtype=float)
    
    # Compute q, u
    Om_a = _omega_m(a_arr, Om0, w0, wa)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a_arr)
    
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_arr + c5*u*q
    
    # Check gamma bounds
    if np.any(gamma < 0.1) or np.any(gamma > 1.2):
        return 1e6
    
    f_approx = Om_a ** gamma
    
    # Reference f
    f_ref = _reference_f(a_arr, Om0, w0, wa)
    
    # Relative error in f
    rel_err_f = np.abs(f_approx - f_ref) / np.abs(f_ref)
    
    # For D(a)/D(1): exp(-∫_ln(a)^0 f(exp(x)) dx)
    # = exp(∫_0^ln(a) f(exp(x)) dx) for a < 1
    # Let's compute the integral numerically
    
    x_arr = np.log(a_arr)
    
    # For each a, compute D(a)/D(1) = exp(-∫_{ln(a)}^0 f(exp(x)) dx) = exp(∫_0^{ln(a)} f(exp(x)) dx)
    # Wait: D(a)/D(1) = exp(-∫_{ln(a)}^0 f(exp(x)) dx)
    # If a < 1, ln(a) < 0, so ∫_{ln(a)}^0 = ∫_0^{ln(a)} * (-1), so -∫_{ln(a)}^0 = ∫_0^{ln(a)}
    # Hmm, let me be careful:
    # D(a)/D(1) = exp(-∫_{ln(a)}^{0} f(e^x) dx)
    # For a < 1: ln(a) < 0, so ∫_{ln(a)}^{0} is positive (going from negative to 0)
    # -∫_{ln(a)}^{0} = ∫_{0}^{ln(a)} which is negative
    # So D(a)/D(1) < 1 for a < 1, which makes sense.
    
    # Compute the integral for each a
    D_ratio_approx = np.zeros_like(a_arr)
    D_ratio_ref = np.zeros_like(a_arr)
    
    for i, ai in enumerate(a_arr):
        xi = np.log(ai)
        if abs(xi) < 1e-10:
            D_ratio_approx[i] = 1.0
            D_ratio_ref[i] = 1.0
            continue
        
        # Create integration points from 0 to xi
        n_pts = 100
        x_pts = np.linspace(0, xi, n_pts)
        
        # Approximate integral
        Om_pts = _omega_m(np.exp(x_pts), Om0, w0, wa)
        q_pts = 1.0 - Om_pts
        u_pts = 1.0 + w0 + wa * (1.0 - np.exp(x_pts))
        gamma_pts = c0 + c1*q_pts + c2*q_pts**2 + c3*u_pts + c4*wa*np.exp(x_pts) + c5*u_pts*q_pts
        f_pts = Om_pts ** gamma_pts
        integral_approx = np.trapz(f_pts, x_pts)
        D_ratio_approx[i] = np.exp(-integral_approx)
        
        # Reference integral
        f_ref_pts = _reference_f(np.exp(x_pts), Om0, w0, wa)
        integral_ref = np.trapz(f_ref_pts, x_pts)
        D_ratio_ref[i] = np.exp(-integral_ref)
    
    rel_err_D = np.abs(D_ratio_approx - D_ratio_ref) / np.abs(D_ratio_ref)
    
    return np.max(np.concatenate([rel_err_f, rel_err_D]))

def construct():
    """Find optimal 6 coefficients by fitting to reference ODE."""
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    # Representative cosmologies to fit
    cosmologies = [
        (0.3, -1.0, 0.0),    # LambdaCDM
        (0.3, -0.9, 0.3),    # w0wa
        (0.3, -1.2, 0.5),    # more variable
        (0.25, -0.8, 0.2),   # lower Om0
        (0.35, -1.0, 0.0),   # higher Om0
    ]
    
    # Scale factors to evaluate
    a_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    # Precompute reference values
    ref_data = []
    for Om0, w0, wa in cosmologies:
        f_ref = _reference_f(a_values, Om0, w0, wa)
        ref_data.append((Om0, w0, wa, f_ref))
    
    # Objective function
    def objective(coeffs):
        if time.time() > deadline:
            return 1e6
        max_err = 0.0
        for Om0, w0, wa, f_ref in ref_data:
            c0, c1, c2, c3, c4, c5 = coeffs
            a_arr = a_values
            Om_a = _omega_m(a_arr, Om0, w0, wa)
            q = 1.0 - Om_a
            u = 1.0 + w0 + wa * (1.0 - a_arr)
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_arr + c5*u*q
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return 1e6
            f_approx = Om_a ** gamma
            rel_err_f = np.abs(f_approx - f_ref) / np.abs(f_ref)
            max_err = max(max_err, np.max(rel_err_f))
        
        # Also check D ratio for a few key points
        for Om0, w0, wa, f_ref in ref_data[:2]:
            for ai in [0.3, 0.5, 0.7]:
                xi = np.log(ai)
                n_pts = 50
                x_pts = np.linspace(0, xi, n_pts)
                a_pts = np.exp(x_pts)
                Om_pts = _omega_m(a_pts, Om0, w0, wa)
                q_pts = 1.0 - Om_pts
                u_pts = 1.0 + w0 + wa * (1.0 - a_pts)
                gamma_pts = coeffs[0] + coeffs[1]*q_pts + coeffs[2]*q_pts**2 + coeffs[3]*u_pts + coeffs[4]*wa*a_pts + coeffs[5]*u_pts*q_pts
                f_pts = Om_pts ** gamma_pts
                integral_approx = np.trapz(f_pts, x_pts)
                D_approx = np.exp(-integral_approx)
                
                f_ref_pts = _reference_f(a_pts, Om0, w0, wa)
                integral_ref = np.trapz(f_ref_pts, x_pts)
                D_ref = np.exp(-integral_ref)
                
                rel_err_D = abs(D_approx - D_ref) / abs(D_ref)
                max_err = max(max_err, rel_err_D)
        
        return max_err
    
    # Start from constant gamma
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    best_x = x0.copy()
    best_obj = objective(x0)
    
    # Try Nelder-Mead
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
        if result.fun < best_obj:
            best_x = result.x
            best_obj = result.fun
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_x.tolist()
    
    # Try L-BFGS-B with bounds
    bounds = [(-10, 10)] * 6
    try:
        result2 = minimize(objective, best_x, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        if result2.fun < best_obj:
            best_x = result2.x
            best_obj = result2.fun
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_x.tolist()
    
    # Polish with tighter Nelder-Mead
    try:
        result3 = minimize(objective, best_x, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12})
        if result3.fun < best_obj:
            best_x = result3.x
            best_obj = result3.fun
    except Exception:
        pass
    
    # Clip to valid range
    best_x = np.clip(best_x, -10.0, 10.0)
    
    return best_x.tolist()
# EVOLVE-BLOCK-END
