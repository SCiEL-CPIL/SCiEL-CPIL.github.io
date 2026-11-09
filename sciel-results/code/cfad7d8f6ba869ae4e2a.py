"""Improved growth-index fit with direct objective optimization."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import cumulative_trapezoid

# EVOLVE-BLOCK-START
def construct():
    """Fit six coefficients to minimize max relative error of f and D."""
    t_start = time.time()
    deadline = t_start + 80.0  # 10s safety margin before 90s kill

    # Grid of CPL parameters (Om0, w0, wa) - physically reasonable range
    # Keep grid small enough for fast evaluation in the optimizer
    Om0_vals = np.array([0.22, 0.28, 0.32])
    w0_vals = np.array([-1.2, -1.0, -0.8])
    wa_vals = np.array([-0.4, 0.0, 0.4])

    # Scale factor grid for fitting
    # Denser grid helps with the integral calculation for D
    a_vals = np.linspace(0.05, 0.99, 50)

    # Pre-compute cosmological parameters for the grid to speed up objective function
    # We store: Om_a, q, u, wa_a, u_q, and the Linder gamma target
    # Also store E2 to check validity
    points = []
    for Om0 in Om0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                for a in a_vals:
                    a_cubed = a**3
                    de_factor = a**(-3.0*(1.0+w0)) * np.exp(-3.0*wa*(1.0-a))
                    E2 = Om0 * a_cubed + (1.0 - Om0) * de_factor
                    
                    if E2 <= 0 or E2 < 1e-12:
                        continue
                    Om_a = Om0 * a_cubed / E2
                    q = 1.0 - Om_a
                    u = 1.0 + w0 + wa * (1.0 - a)
                    
                    if u <= 0.01:
                        continue
                    
                    # Linder-Van de Stark: gamma = (5/7) * (Om(a)/u)^(4/7)
                    gamma_target = (5.0/7.0) * (Om_a / u)**(4.0/7.0)
                    
                    # Design matrix row: [1, q, q^2, u, wa*a, u*q]
                    row = [1.0, q, q*q, u, wa*a, u*q]
                    
                    # Store necessary info for f and D calculation
                    # We need Om_a for f = Om^gamma
                    # We need gamma_target for reference f and D
                    points.append({
                        'row': row,
                        'Om_a': Om_a,
                        'gamma_target': gamma_target,
                        'a': a,
                        'Om0': Om0,
                        'w0': w0,
                        'wa': wa
                    })

    if len(points) < 10:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Convert to arrays for vectorization where possible
    A = np.array([p['row'] for p in points])
    Om_a_arr = np.array([p['Om_a'] for p in points])
    gamma_target_arr = np.array([p['gamma_target'] for p in points])
    
    # Group points by cosmology to calculate D(a) correctly (integration over a for fixed cosmology)
    # We need to compute D(a)/D(1) = exp(-integral_ln(a)^1 f(exp(x)) dx)
    # Note: The prompt says integral from ln(a) to 0. 
    # Let x = ln(a). dx = da/a. 
    # Integral I(a) = \int_{\ln a}^0 f(e^x) dx = \int_a^1 f(a') \frac{da'}{a'}
    
    # To calculate D, we need to integrate f over a for each cosmology.
    # We will group the points by (Om0, w0, wa).
    
    cosmology_groups = {}
    for i, p in enumerate(points):
        key = (p['Om0'], p['w0'], p['wa'])
        if key not in cosmology_groups:
            cosmology_groups[key] = []
        cosmology_groups[key].append(i)

    # Pre-compute reference f and D for each group
    # We will store the indices and the pre-computed reference values
    # ref_f[i] and ref_D[i]
    ref_f = np.zeros(len(points))
    ref_D = np.zeros(len(points))
    
    for key, indices in cosmology_groups.items():
        # Sort indices by 'a' to ensure correct integration order
        indices_sorted = sorted(indices, key=lambda i: points[i]['a'])
        
        # Extract a, f_ref for this group
        a_group = np.array([points[i]['a'] for i in indices_sorted])
        f_ref_group = np.array([points[i]['Om_a']**points[i]['gamma_target'] for i in indices_sorted])
        
        # Calculate integral \int_a^1 f(a') da'/a'
        # Let y = ln(a). dy = da/a.
        # Integral = \int_{ln(a)}^0 f(e^y) dy
        # We have points at specific a. We can use cumulative_trapezoid on the y-grid.
        y_group = np.log(a_group)
        
        # We need the integral from y to 0.
        # cumulative_trapezoid(y) gives integral from start to current.
        # We want integral from current to end (0).
        # So, total_integral = trapezoid(f, y)
        # integral_to_0[i] = total_integral - cumulative_integral[i]
        
        # f is a function of y. We have f at y_group.
        # trapezoid(f, y)
        total_int = np.trapz(f_ref_group, y_group)
        cum_int = cumulative_trapezoid(f_ref_group, y_group, initial=0)
        
        # integral from y[i] to 0 is total_int - cum_int[i]
        # Note: y_group is increasing (a from 0.05 to 0.99).
        # cum_int[0] = 0. cum_int[-1] = total_int.
        # So integral from y[i] to y[-1] (which is ~0) is total_int - cum_int[i].
        # Wait, y[-1] = ln(0.99) ~ -0.01. It's close to 0.
        # The prompt says integral to 0. We can extrapolate or just use the last point as 0.
        # Given a=0.99, ln(0.99) is very close to 0.
        
        int_vals = total_int - cum_int
        # D = exp(-int_vals)
        D_group = np.exp(-int_vals)
        
        for j, idx in enumerate(indices_sorted):
            ref_f[idx] = f_ref_group[j]
            ref_D[idx] = D_group[j]

    # Objective function: Max relative error of f and D
    def objective(coeffs):
        # Compute ansatz gamma
        gamma_ansatz = A @ coeffs
        
        # Check gamma range [0.1, 1.2]
        if np.any(gamma_ansatz < 0.1) or np.any(gamma_ansatz > 1.2):
            return 1e6 # Penalty
        
        # Compute ansatz f and D
        f_ansatz = Om_a_arr ** gamma_ansatz
        
        # Compute D_ansatz for each group
        D_ansatz = np.zeros(len(points))
        
        for key, indices in cosmology_groups.items():
            indices_sorted = sorted(indices, key=lambda i: points[i]['a'])
            a_group = np.array([points[i]['a'] for i in indices_sorted])
            f_ans_group = np.array([f_ansatz[i] for i in indices_sorted])
            
            y_group = np.log(a_group)
            total_int = np.trapz(f_ans_group, y_group)
            cum_int = cumulative_trapezoid(f_ans_group, y_group, initial=0)
            int_vals = total_int - cum_int
            D_group = np.exp(-int_vals)
            
            for j, idx in enumerate(indices_sorted):
                D_ansatz[idx] = D_group[j]
        
        # Relative errors
        # Avoid division by zero
        eps = 1e-12
        rel_err_f = np.abs(f_ansatz - ref_f) / (np.abs(ref_f) + eps)
        rel_err_D = np.abs(D_ansatz - ref_D) / (np.abs(ref_D) + eps)
        
        # Max error
        max_err = np.max(np.concatenate([rel_err_f, rel_err_D]))
        
        return max_err

    # Initial guess: Linder constant approx or previous best
    # A good starting point is the coefficients that fit gamma_target well
    # Use least squares as initial guess
    try:
        coeffs_init, _, _, _ = np.linalg.lstsq(A, gamma_target_arr, rcond=None)
        coeffs_init = np.clip(coeffs_init, -10.0, 10.0)
    except:
        coeffs_init = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Optimize
    # Nelder-Mead is robust for max functions
    bounds = [(-10, 10)] * 6
    
    # Check time
    if time.time() > deadline - 10.0:
        return [float(c) for c in coeffs_init]

    try:
        result = minimize(objective, coeffs_init, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        
        if result.success or result.fun < objective(coeffs_init):
            coeffs_opt = np.clip(result.x, -10.0, 10.0)
        else:
            coeffs_opt = coeffs_init
    except Exception:
        coeffs_opt = coeffs_init

    # Final check and return
    # Ensure gamma range is respected
    gamma_check = A @ coeffs_opt
    if np.any(gamma_check < 0.1) or np.any(gamma_check > 1.2):
        # If violated, fall back to a safer constant or LS solution
        # Try to find a constant gamma that is within range and close to LS
        # Or just use the LS solution if it was valid
        gamma_ls = A @ coeffs_init
        if not (np.any(gamma_ls < 0.1) or np.any(gamma_ls > 1.2)):
            coeffs_opt = coeffs_init
        else:
            # Fallback to constant 0.55
            coeffs_opt = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])

    return [float(c) for c in coeffs_opt]
# EVOLVE-BLOCK-END
