"""Growth-index formula optimizer for CPL dark-energy cosmologies."""
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import minimize
import time

_START_TIME = time.time()
_DEADLINE = 82.0  # safety margin before 90s limit

# Cache for reference data
_ref_cache = {}

def _Om_of_a(a, Om0, w0, wa):
    """Compute Omega_matter(a) for CPL dark energy using analytic integral."""
    if a == 1.0:
        return Om0
    
    # Analytic integral of (1+w0+wa*(1-x))/x from a to 1
    # = (1+w0)*ln(1/a) + wa*(1/a - 1)
    integral_de = (1.0 + w0) * np.log(1.0 / a) + wa * (1.0 / a - 1.0)
    
    ratio_de = np.exp(-3.0 * integral_de)
    ratio_m = a ** (-3.0)
    
    rho_m_a = Om0 * ratio_m
    rho_de_a = (1.0 - Om0) * ratio_de
    
    return rho_m_a / (rho_m_a + rho_de_a)

def _gamma_exact(a, Om0, w0, wa):
    """Compute exact gamma = ln(f)/ln(Om) using ODE integration."""
    Om_a = _Om_of_a(a, Om0, w0, wa)
    if Om_a <= 0 or Om_a >= 1 or abs(Om_a - 1.0) < 1e-12:
        return 0.55
    
    def ode_rhs(lna, y):
        a_local = np.exp(lna)
        Om_local = _Om_of_a(a_local, Om0, w0, wa)
        u_local = 1.0 + w0 + wa * (1.0 - a_local)
        dlnH2_dlna = 3.0 * Om_local + 3.0 * (1.0 - Om_local) * u_local - 3.0 * (1.0 - Om_local) * wa
        Hprime_over_H = 0.5 * dlnH2_dlna
        f_local = y[0]
        df_dlna = -(3.0/2.0 + Hprime_over_H) * f_local + (3.0/2.0) * Om_local
        return [df_dlna]
    
    lna_target = np.log(a)
    lna_start = -5.0
    y0 = [2.0/3.0]
    
    sol = solve_ivp(ode_rhs, [lna_start, lna_target], y0, method='RK45', rtol=1e-9, atol=1e-11)
    if sol.success:
        f_val = sol.y[0, -1]
        if f_val > 0 and Om_a > 0 and Om_a != 1:
            return np.log(f_val) / np.log(Om_a)
    return 0.55

def _gamma_from_coeffs(coeffs, q, u, wa, a):
    """Compute gamma from coefficients."""
    c0, c1, c2, c3, c4, c5 = coeffs
    return c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q

def _compute_max_error(coeffs, cosmologies, a_values):
    """Compute max relative error for given coefficients against ODE reference."""
    max_rel_err = 0.0
    
    for Om0, w0, wa in cosmologies:
        for a in a_values:
            Om_a = _Om_of_a(a, Om0, w0, wa)
            if Om_a <= 0 or Om_a >= 1:
                continue
            q = 1.0 - Om_a
            u = 1.0 + w0 + wa * (1.0 - a)
            
            gamma = _gamma_from_coeffs(coeffs, q, u, wa, a)
            
            if gamma < 0.1 or gamma > 1.2:
                return 1e10
            
            f_approx = Om_a ** gamma
            
            # Get reference f from ODE
            def ode_rhs(lna, y):
                a_local = np.exp(lna)
                Om_local = _Om_of_a(a_local, Om0, w0, wa)
                u_local = 1.0 + w0 + wa * (1.0 - a_local)
                dlnH2_dlna = 3.0 * Om_local + 3.0 * (1.0 - Om_local) * u_local - 3.0 * (1.0 - Om_local) * wa
                Hprime_over_H = 0.5 * dlnH2_dlna
                f_local = y[0]
                df_dlna = -(3.0/2.0 + Hprime_over_H) * f_local + (3.0/2.0) * Om_local
                return [df_dlna]
            
            lna_target = np.log(a)
            lna_start = -5.0
            y0 = [2.0/3.0]
            sol = solve_ivp(ode_rhs, [lna_start, lna_target], y0, method='RK45', rtol=1e-9, atol=1e-11)
            if sol.success:
                f_ref = sol.y[0, -1]
            else:
                f_ref = 0.55
            
            if abs(f_ref) > 1e-10:
                rel_err_f = abs(f_approx - f_ref) / abs(f_ref)
                max_rel_err = max(max_rel_err, rel_err_f)
            
            # Compute D(a)/D(1) reference and approximate
            if a == 1.0:
                D_approx = 1.0
                D_ref = 1.0
            else:
                # Reference D
                def integrand_ref(x):
                    a_local = np.exp(x)
                    Om_local = _Om_of_a(a_local, Om0, w0, wa)
                    u_local = 1.0 + w0 + wa * (1.0 - a_local)
                    dlnH2_dlna = 3.0 * Om_local + 3.0 * (1.0 - Om_local) * u_local - 3.0 * (1.0 - Om_local) * wa
                    Hprime_over_H = 0.5 * dlnH2_dlna
                    return Hprime_over_H  # placeholder, need f
                    
                # For reference D, we need to integrate f_ref
                # Use the ODE solution on a grid
                if lna_target < 0:
                    lna_grid = np.linspace(lna_target, 0.0, 50)
                else:
                    lna_grid = np.linspace(0.0, lna_target, 50)
                
                sol_grid = solve_ivp(ode_rhs, [lna_start, lna_grid[-1]], y0, 
                                     t_eval=lna_grid, method='RK45', rtol=1e-9, atol=1e-11)
                if sol_grid.success:
                    f_grid = sol_grid.y[0]
                    if lna_target < 0:
                        integral_ref = np.trapz(f_grid, lna_grid)
                        D_ref = np.exp(integral_ref)
                    else:
                        integral_ref = np.trapz(f_grid, lna_grid)
                        D_ref = np.exp(-integral_ref)
                else:
                    D_ref = 1.0
                
                # Approximate D
                def integrand_approx(x):
                    a_local = np.exp(x)
                    Om_local = _Om_of_a(a_local, Om0, w0, wa)
                    q_local = 1.0 - Om_local
                    u_local = 1.0 + w0 + wa * (1.0 - a_local)
                    gamma_local = _gamma_from_coeffs(coeffs, q_local, u_local, wa, a_local)
                    if gamma_local < 0.1 or gamma_local > 1.2:
                        return 0.55
                    return Om_local ** gamma_local
                
                if lna_target < 0:
                    integral_approx, _ = quad(integrand_approx, lna_target, 0.0, limit=100)
                    D_approx = np.exp(integral_approx)
                else:
                    integral_approx, _ = quad(integrand_approx, 0.0, lna_target, limit=100)
                    D_approx = np.exp(-integral_approx)
            
            if abs(D_ref) > 1e-10:
                rel_err_D = abs(D_approx - D_ref) / abs(D_ref)
                max_rel_err = max(max_rel_err, rel_err_D)
    
    return 100.0 * max_rel_err

def construct():
    """Construct the six coefficients using Chebyshev spectral projection + local refinement."""
    global _START_TIME, _DEADLINE
    
    # Define validation cosmologies and a-values
    cosmologies = [
        (0.3, -1.0, 0.0),
        (0.3, -1.0, 0.3),
        (0.3, -0.9, 0.0),
        (0.25, -1.0, 0.5),
        (0.35, -1.2, 0.0),
        (0.28, -1.0, 0.2),
        (0.32, -1.1, -0.3),
    ]
    
    a_eval = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    
    # Phase 1: Chebyshev spectral projection
    # Build a dense grid of (Om0, w0, wa) and a values, compute exact gamma,
    # then fit our 6-coefficient ansatz to minimize integrated squared error
    Om0_grid = np.array([0.25, 0.28, 0.30, 0.32, 0.35])
    w0_grid = np.array([-1.2, -1.0, -0.9])
    wa_grid = np.array([-0.3, 0.0, 0.2, 0.3])
    a_fine = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
    
    A_rows = []
    b_rows = []
    
    for Om0 in Om0_grid:
        for w0 in w0_grid:
            for wa in wa_grid:
                for a in a_fine:
                    if time.time() > _DEADLINE - 40.0:
                        break
                    Om_a = _Om_of_a(a, Om0, w0, wa)
                    if Om_a <= 0.05 or Om_a >= 0.95:
                        continue
                    
                    gamma_exact = _gamma_exact(a, Om0, w0, wa)
                    if gamma_exact < 0.1 or gamma_exact > 1.2:
                        continue
                    
                    q = 1.0 - Om_a
                    u = 1.0 + w0 + wa * (1.0 - a)
                    
                    row = [1.0, q, q**2, u, wa * a, u * q]
                    A_rows.append(row)
                    b_rows.append(gamma_exact)
    
    best_coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    if len(A_rows) > 6:
        A = np.array(A_rows)
        b = np.array(b_rows)
        
        try:
            coeffs_taylor, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            coeffs_taylor = list(coeffs_taylor)
            
            if all(-10 <= c <= 10 for c in coeffs_taylor):
                # Verify gamma stays in [0.1, 1.2]
                valid = True
                for Om0, w0, wa in cosmologies:
                    for a in a_eval:
                        Om_a = _Om_of_a(a, Om0, w0, wa)
                        q = 1.0 - Om_a
                        u = 1.0 + w0 + wa * (1.0 - a)
                        gamma = _gamma_from_coeffs(coeffs_taylor, q, u, wa, a)
                        if gamma < 0.1 or gamma > 1.2:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    best_coeffs = coeffs_taylor
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead local optimization to refine
    if time.time() < _DEADLINE - 25.0:
        def nm_objective(x):
            if time.time() > _DEADLINE - 5.0:
                return 1e10
            if any(c < -10 or c > 10 for c in x):
                return 1e10
            return _compute_max_error(x, cosmologies, a_eval)
        
        try:
            nm_result = minimize(nm_objective, best_coeffs, method='Nelder-Mead',
                                options={'maxiter': 400, 'xatol': 1e-8, 'fatol': 1e-8,
                                        'adaptive': True})
            if nm_result.fun < 1e9:
                candidate = list(nm_result.x)
                if all(-10 <= c <= 10 for c in candidate):
                    valid = True
                    for Om0, w0, wa in cosmologies:
                        for a in a_eval:
                            Om_a = _Om_of_a(a, Om0, w0, wa)
                            q = 1.0 - Om_a
                            u = 1.0 + w0 + wa * (1.0 - a)
                            gamma = _gamma_from_coeffs(candidate, q, u, wa, a)
                            if gamma < 0.1 or gamma > 1.2:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        best_coeffs = candidate
        except Exception:
            pass
    
    # Phase 3: Fine-tune
    if time.time() < _DEADLINE - 10.0:
        def nm_objective_fine(x):
            if time.time() > _DEADLINE - 3.0:
                return 1e10
            if any(c < -10 or c > 10 for c in x):
                return 1e10
            return _compute_max_error(x, cosmologies, a_eval)
        
        try:
            nm_result2 = minimize(nm_objective_fine, best_coeffs, method='Nelder-Mead',
                                 options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10,
                                         'adaptive': True})
            if nm_result2.fun < 1e9:
                candidate = list(nm_result2.x)
                if all(-10 <= c <= 10 for c in candidate):
                    valid = True
                    for Om0, w0, wa in cosmologies:
                        for a in a_eval:
                            Om_a = _Om_of_a(a, Om0, w0, wa)
                            q = 1.0 - Om_a
                            u = 1.0 + w0 + wa * (1.0 - a)
                            gamma = _gamma_from_coeffs(candidate, q, u, wa, a)
                            if gamma < 0.1 or gamma > 1.2:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        best_coeffs = candidate
        except Exception:
            pass
    
    # Ensure coefficients are in valid range
    for i in range(6):
        if best_coeffs[i] < -10:
            best_coeffs[i] = -10
        elif best_coeffs[i] > 10:
            best_coeffs[i] = 10
    
    return best_coeffs
