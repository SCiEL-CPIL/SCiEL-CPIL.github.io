"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Derive six coefficients for the growth-index ansatz:
    gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    
    where q = 1 - Om(a), u = 1 + w0 + wa*(1-a)
    
    We fit to the numerically integrated growth index derived from the
    general-relativistic second-order ODE for the growth factor.
    """
    import numpy as np
    from scipy.integrate import solve_ivp

    # Grid of cosmological parameters
    Om0_vals = np.linspace(0.2, 0.35, 10)
    w0_vals = np.linspace(-1.0, 0.0, 8)
    wa_vals = np.linspace(-0.5, 0.5, 8)
    
    # Scale factor grid
    a_vals = np.linspace(0.1, 1.0, 20)
    
    # Build design matrix and target
    A_rows = []
    y_vals = []
    
    for Om0 in Om0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                
                # Define E^2(a) and Om(a)
                def E2_of_a(a):
                    if abs(wa) < 1e-10:
                        return Om0 * a**3 + (1 - Om0) * a**2
                    else:
                        return Om0 * a**3 + (1 - Om0) * a**2 * (1 + wa)**(2*wa) * np.exp(3*wa*(1-a))
                
                def Om_of_a(a):
                    E2 = E2_of_a(a)
                    if E2 <= 0: return 0
                    return Om0 * a**3 / E2
                
                # d ln H / d ln a = 0.5 * d ln E^2 / d ln a = 0.5 * (a/E^2) * dE^2/da
                # dE^2/da can be computed analytically or numerically.
                # Analytical is better for accuracy.
                def dlnH_dlna(a):
                    if abs(wa) < 1e-10:
                        E2 = Om0 * a**3 + (1 - Om0) * a**2
                        dE2_da = 3 * Om0 * a**2 + 2 * (1 - Om0) * a
                    else:
                        # E2 = Om0 a^3 + (1-Om0) a^2 (1+wa)^(2wa) exp(3wa(1-a))
                        # Let K = (1+wa)^(2wa)
                        # E2 = Om0 a^3 + (1-Om0) K a^2 exp(3wa(1-a))
                        # dE2/da = 3 Om0 a^2 + (1-Om0) K [ 2a exp(3wa(1-a)) + a^2 exp(3wa(1-a)) (-3wa) ]
                        K = (1 + wa)**(2*wa)
                        term = K * np.exp(3*wa*(1-a))
                        dE2_da = 3 * Om0 * a**2 + (1 - Om0) * (2*a*term - 3*wa*a**2*term)
                        E2 = Om0 * a**3 + (1 - Om0) * a**2 * term
                    if E2 <= 0: return 0
                    return 0.5 * (a / E2) * dE2_da

                # ODE for f: df/dlna = 1.5*Om - f*(3 + dlnH/dlna)
                # Let x = ln(a), so a = exp(x)
                # df/dx = 1.5*Om(exp(x)) - f*(3 + dlnH/dlna(exp(x)))
                
                # We integrate from a=1.0 (x=0) backwards to a=0.1?
                # Or from a=0.1 to 1.0?
                # Initial condition at a=1.0: f(1) ~ 1 (or slightly less for DE).
                # Actually, f(1) is determined by the ODE.
                # We can integrate forward from a=0.1 if we have an initial guess.
                # Or integrate backward from a=1.0 with f(1)=1.0?
                # At a=1, Om=Om0. f(1) is not exactly 1.
                # However, the error in f(1) is small.
                # Let's integrate from a=1.0 down to a=0.1.
                # x goes from 0 to ln(0.1).
                
                def ode_rhs(x, f):
                    a = np.exp(x)
                    Om = Om_of_a(a)
                    dlnH = dlnH_dlna(a)
                    return 1.5 * Om - f[0] * (3.0 + dlnH)
                
                # Solve from x=0 to x=ln(0.1)
                x_span = (0.0, np.log(0.1))
                x_eval = np.log(a_vals) # These are negative, so we need to reverse or handle direction
                # solve_ivp integrates from t0 to tf.
                # We want x from 0 to ln(0.1) which is negative.
                # So t0=0, tf=ln(0.1).
                
                try:
                    sol = solve_ivp(ode_rhs, x_span, [1.0], t_eval=np.sort(x_eval)[::-1], method='RK45', rtol=1e-8, atol=1e-10)
                    if not sol.success:
                        continue
                except:
                    continue
                
                # Map solutions back to a_vals
                # sol.y[0] corresponds to x_eval sorted descending (0 to ln(0.1))
                # We need to align with a_vals (0.1 to 1.0)
                # x_eval was sorted descending. a_vals is ascending.
                # So sol.y[0][0] is at x=0 (a=1), sol.y[0][-1] is at x=ln(0.1) (a=0.1)
                # We need f for each a in a_vals.
                # Let's just iterate a_vals and find the corresponding x in sol.t
                
                for i, a in enumerate(a_vals):
                    Om_a = Om_of_a(a)
                    if Om_a <= 0 or Om_a >= 1:
                        continue
                    
                    # Find f at this a
                    x = np.log(a)
                    # Interpolate from sol.t, sol.y
                    # sol.t is descending x.
                    idx = np.searchsorted(sol.t[::-1], x)
                    if idx == 0 or idx == len(sol.t):
                        continue
                    # Linear interpolation
                    x1, x2 = sol.t[::-1][idx-1], sol.t[::-1][idx]
                    f1, f2 = sol.y[0][::-1][idx-1], sol.y[0][::-1][idx]
                    if x2 == x1:
                        f_a = f1
                    else:
                        f_a = f1 + (f2 - f1) * (x - x1) / (x2 - x1)
                    
                    if f_a <= 0:
                        continue
                    
                    # Target gamma = ln(f) / ln(Om)
                    if Om_a < 1e-10:
                        continue
                    gamma_target = np.log(f_a) / np.log(Om_a)
                    
                    if gamma_target < 0.1 or gamma_target > 1.2:
                        continue
                    
                    q = 1 - Om_a
                    u = 1 + w0 + wa * (1 - a)
                    if u <= 0:
                        continue
                    
                    row = np.array([1.0, q, q**2, u, wa * a, u * q])
                    A_rows.append(row)
                    y_vals.append(gamma_target)
    
    if len(A_rows) == 0:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    A = np.array(A_rows)
    y = np.array(y_vals)
    
    # Solve least squares
    coeffs, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)
    
    # Clip coefficients to [-10, 10]
    coeffs = np.clip(coeffs, -10.0, 10.0)
    
    c0, c1, c2, c3, c4, c5 = coeffs
    
    # Quick validation on a small subset to ensure stability
    # If it fails, return constant
    valid = True
    for Om0 in Om0_vals[::3]:
        for w0 in w0_vals[::3]:
            for wa in wa_vals[::3]:
                for a in a_vals[::3]:
                    if abs(wa) < 1e-10:
                        E2 = Om0 * a**3 + (1 - Om0) * a**2
                    else:
                        E2 = Om0 * a**3 + (1 - Om0) * a**2 * (1 + wa)**(2*wa) * np.exp(3*wa*(1-a))
                    if E2 <= 0: continue
                    Om_a = Om0 * a**3 / E2
                    if Om_a <= 0 or Om_a >= 1: continue
                    q = 1 - Om_a
                    u = 1 + w0 + wa * (1 - a)
                    if u <= 0: continue
                    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
                    if gamma < 0.1 or gamma > 1.2:
                        valid = False
                        break
                if not valid: break
            if not valid: break
        if not valid: break
    
    if not valid:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    return [float(c0), float(c1), float(c2), float(c3), float(c4), float(c5)]
# EVOLVE-BLOCK-END
