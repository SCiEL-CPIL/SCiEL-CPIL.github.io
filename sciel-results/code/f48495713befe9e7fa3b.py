"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import cumulative_trapezoid

# EVOLVE-BLOCK-START
def construct():
    """
    Fit six coefficients to minimize the worst-case relative error of f and D 
    across a grid of CPL cosmologies.
    """
    t_start = time.time()
    deadline = t_start + 80.0  # 10s safety margin before 90s kill
    
    # Dense grid for the final optimization to better estimate the max error
    Om0_opt = np.array([0.20, 0.25, 0.28, 0.30, 0.32, 0.35, 0.40])
    w0_opt = np.array([-1.2, -1.0, -0.9, -0.8, -0.7])
    wa_opt = np.array([-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5])
    a_grid = np.linspace(1.0, 0.05, 100)
    
    # Pre-compute cosmological quantities for all (Om0, w0, wa, a) combinations
    cosm_data = []
    for Om0 in Om0_opt:
        for w0 in w0_opt:
            for wa in wa_opt:
                a = a_grid
                a_cubed = a**3
                de_factor = a**(-3.0*(1.0+w0)) * np.exp(-3.0*wa*(1.0-a))
                E2 = Om0 * a_cubed + (1.0 - Om0) * de_factor
                
                # Handle potential invalid E2
                valid_mask = E2 > 1e-10
                if np.sum(valid_mask) < 50:
                    continue
                    
                Om_a = np.zeros_like(a)
                Om_a[valid_mask] = Om0 * a_cubed[valid_mask] / E2[valid_mask]
                
                q = 1.0 - Om_a
                u = 1.0 + w0 + wa * (1.0 - a)
                
                # Check u validity
                if np.any(u <= 0.05):
                    continue
                    
                gamma_true = (5.0/7.0) * (Om_a / u)**(4.0/7.0)
                f_true = Om_a**gamma_true
                
                # Compute D_true
                x_grid = np.log(a)
                x_rev = x_grid[::-1]
                f_rev = f_true[::-1]
                cum_int_rev = cumulative_trapezoid(f_rev, x_rev, initial=0.0)
                total_int = cum_int_rev[-1]
                idx_rev = len(a) - 1 - np.arange(len(a))
                integral_x_to_0 = total_int - cum_int_rev[idx_rev]
                D_true = np.exp(-integral_x_to_0)
                
                # Design matrix row for each a in this cosmology
                # gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
                X = np.column_stack([
                    np.ones_like(q),
                    q,
                    q*q,
                    u,
                    wa * a,
                    u * q
                ])
                
                cosm_data.append({
                    'a': a,
                    'x_grid': x_grid,
                    'Om_a': Om_a,
                    'q': q,
                    'u': u,
                    'wa_a': wa * a,
                    'u_q': u * q,
                    'f_true': f_true,
                    'D_true': D_true,
                    'X': X
                })
                
                if time.time() > deadline - 15.0:
                    break
            if time.time() > deadline - 15.0:
                break
        if time.time() > deadline - 15.0:
            break

    if not cosm_data:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Compute the range of X values to determine bounds for coefficients
    all_X = np.concatenate([d['X'] for d in cosm_data], axis=0)
    X_min = np.min(all_X, axis=0)
    X_max = np.max(all_X, axis=0)
    
    # Initial coefficients: Linder-Van de Stark approximation
    # gamma ≈ 0.55 + 0.3*q
    initial_coeffs = np.array([0.55, 0.3, 0.0, 0.0, 0.0, 0.0])
    initial_coeffs = np.clip(initial_coeffs, -10.0, 10.0)

    def compute_max_err(c):
        """Compute the maximum relative error of f and D across all cosmologies."""
        c = np.array(c)
        max_err = 0.0
        
        for data in cosm_data:
            X = data['X']
            Om_a = data['Om_a']
            f_true = data['f_true']
            D_true = data['D_true']
            x_grid = data['x_grid']
            
            gamma = X @ c
            
            # Check constraints
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return 1e6
            
            f_approx = Om_a**gamma
            
            # Error in f
            err_f = np.abs(f_approx - f_true) / np.abs(f_true)
            max_err_f = np.max(err_f)
            if max_err_f > max_err:
                max_err = max_err_f
            
            # Compute D approx
            x_rev = x_grid[::-1]
            f_approx_rev = f_approx[::-1]
            cum_int_rev_app = cumulative_trapezoid(f_approx_rev, x_rev, initial=0.0)
            total_int_app = cum_int_rev_app[-1]
            idx_rev = len(x_grid) - 1 - np.arange(len(x_grid))
            integral_x_to_0_app = total_int_app - cum_int_rev_app[idx_rev]
            D_approx = np.exp(-integral_x_to_0_app)
            
            err_D = np.abs(D_approx - D_true) / np.abs(D_true)
            max_err_D = np.max(err_D)
            if max_err_D > max_err:
                max_err = max_err_D
                
        return max_err

    best_coeffs = initial_coeffs.copy()
    best_err = compute_max_err(best_coeffs)

    # Try Nelder-Mead from the initial point
    try:
        result = minimize(compute_max_err, best_coeffs, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
        
        if result.fun < best_err:
            best_coeffs = result.x
            best_err = result.fun
    except Exception:
        pass

    # Try a few more restarts from slightly perturbed points
    np.random.seed(42)
    for i in range(3):
        if time.time() > deadline - 20.0:
            break
        perturbed = best_coeffs + np.random.randn(6) * 0.1
        perturbed = np.clip(perturbed, -10.0, 10.0)
        
        try:
            result = minimize(compute_max_err, perturbed, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
            
            if result.fun < best_err:
                best_coeffs = result.x
                best_err = result.fun
        except Exception:
            pass

    # Final validity check and clip
    best_coeffs = np.clip(best_coeffs, -10.0, 10.0)
    
    # Verify gamma constraints on the final clipped coefficients
    valid = True
    for data in cosm_data:
        gamma = data['X'] @ best_coeffs
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            valid = False
            break
    
    if not valid:
        # Fallback to initial coefficients
        best_coeffs = initial_coeffs
    
    return [float(c) for c in best_coeffs]
# EVOLVE-BLOCK-END
