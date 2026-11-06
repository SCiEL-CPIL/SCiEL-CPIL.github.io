"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def _compute_om_a_vec(a, Om0, w0, wa):
    """Compute Omega_m(a) for CPL dark energy (vectorized)."""
    a = np.asarray(a, dtype=float)
    a_safe = np.maximum(a, 1e-10)
    a3 = a_safe ** 3
    Omd0 = 1.0 - Om0
    w_eff = 1.0 + w0
    if abs(w_eff) < 1e-12:
        de_term = Omd0 * np.ones_like(a_safe)
    else:
        de_term = Omd0 * np.exp(3.0 * w_eff * a_safe)
    total = Om0 / a3 + de_term
    return Om0 / a3 / total

def _compute_gamma_vec(c, a, Om0, w0, wa):
    """Compute gamma from the six-coefficient formula (vectorized)."""
    a = np.asarray(a, dtype=float)
    Om_a = _compute_om_a_vec(a, Om0, w0, wa)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q
    return gamma

def _compute_f_vec(c, a, Om0, w0, wa):
    """Compute f = Om(a)^gamma (vectorized)."""
    gamma = _compute_gamma_vec(c, a, Om0, w0, wa)
    Om_a = _compute_om_a_vec(a, Om0, w0, wa)
    return Om_a ** gamma

def _compute_D_normalized_vec(c, a, Om0, w0, wa, n_points=500):
    """Compute D(a)/D(1) = exp(-integral_ln(a)^0 f(exp(x)) dx) (vectorized)."""
    a = np.asarray(a, dtype=float)
    result = np.ones_like(a)
    mask = a < 1.0
    if not np.any(mask):
        return result
    
    a_m = a[mask]
    log_a = np.log(a_m)
    
    # For each a, integrate from log(a) to 0
    # Use a shared x-grid and trapezoidal integration
    # To be efficient, use a fixed fine grid and interpolate
    x_fine = np.linspace(log_a.min(), 0.0, n_points)
    a_fine = np.exp(x_fine)
    f_fine = _compute_f_vec(c, a_fine, Om0, w0, wa)
    
    # For each target a, find the integral from log(a) to 0
    for i in range(len(a_m)):
        x_target = log_a[i]
        # Find points where x_fine >= x_target
        idx = np.searchsorted(x_fine, x_target)
        if idx >= len(x_fine):
            result[mask][i] = 1.0
        elif idx == 0:
            # Integrate over the full range
            integral = np.trapz(f_fine, x_fine)
            result[mask][i] = np.exp(-integral)
        else:
            # Interpolate f at x_target
            x0, x1 = x_fine[idx-1], x_fine[idx]
            f0, f1 = f_fine[idx-1], f_fine[idx]
            if x1 != x0:
                t = (x_target - x0) / (x1 - x0)
                f_target = (1-t)*f0 + t*f1
            else:
                f_target = f0
            # Create a sub-grid from x_target to 0
            x_sub = np.concatenate([[x_target], x_fine[idx:]])
            f_sub = np.concatenate([[f_target], f_fine[idx:]])
            integral = np.trapz(f_sub, x_sub)
            result[mask][i] = np.exp(-integral)
    
    return result

def _solve_reference_ode(Om0, w0, wa, a_values):
    """Solve the growth ODE to get reference f and D values at given a_values."""
    # x = ln(a), integrate from x=0 (a=1) downward
    # Riccati: f' + (3 + H'/H) f - 1.5*Om_m + f^2 = 0
    # D' = f*D
    
    # Sort a_values in descending order (from 1 down to smallest)
    sorted_a = np.sort(np.asarray(a_values, dtype=float))[::-1]
    n = len(sorted_a)
    
    # Create fine grid from 0 down to log(smallest_a)
    x_min = np.log(sorted_a[-1])
    n_fine = 3000
    x_fine = np.linspace(0.0, x_min, n_fine)
    
    # Initialize
    f_val = 0.55  # at a=1, f(1) ≈ 0.55 for LCDM
    D_val = 1.0
    
    f_array = np.zeros(n_fine)
    D_array = np.zeros(n_fine)
    
    f_array[0] = f_val
    D_array[0] = D_val
    
    # Precompute Om on the fine grid for speed
    a_fine = np.exp(x_fine)
    Om_fine = _compute_om_a_vec(a_fine, Om0, w0, wa)
    Omd_fine = 1.0 - Om_fine
    w_fine = w0 + wa * (1.0 - a_fine)
    HpH_fine = 1.5 * Om_fine - 1.5 * (1.0 + w_fine) * Omd_fine
    
    for i in range(n_fine - 1):
        dx = x_fine[i+1] - x_fine[i]  # negative
        
        # RK4 for f
        def f_deriv(x_idx, f):
            # Use precomputed values with linear interpolation if needed
            if 0 <= x_idx < n_fine:
                idx = int(x_idx)
                frac = x_idx - idx
                if frac < 1e-10:
                    Om_x = Om_fine[idx]
                    HpH_x = HpH_fine[idx]
                elif idx + 1 < n_fine:
                    Om_x = (1-frac)*Om_fine[idx] + frac*Om_fine[idx+1]
                    HpH_x = (1-frac)*HpH_fine[idx] + frac*HpH_fine[idx+1]
                else:
                    Om_x = Om_fine[-1]
                    HpH_x = HpH_fine[-1]
            else:
                # Fallback: compute directly
                a_x = np.exp(x_idx)
                Om_x = _compute_om_a_vec(np.array([a_x]), Om0, w0, wa)[0]
                Omd_x = 1.0 - Om_x
                w_x = w0 + wa * (1.0 - a_x)
                HpH_x = 1.5 * Om_x - 1.5 * (1.0 + w_x) * Omd_x
            return 1.5 * Om_x - f**2 - (3.0 + HpH_x) * f
        
        k1 = f_deriv(float(i), f_val)
        k2 = f_deriv(float(i) + 0.5*dx, f_val + 0.5*dx*k1)
        k3 = f_deriv(float(i) + 0.5*dx, f_val + 0.5*dx*k2)
        k4 = f_deriv(float(i) + dx, f_val + dx*k3)
        
        f_new = f_val + (dx/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        
        # D' = f*D, use average f for D update
        f_avg = 0.5 * (f_val + f_new)
        D_new = D_val * np.exp(f_avg * dx)
        
        f_val = f_new
        D_val = D_new
        f_array[i+1] = f_val
        D_array[i+1] = D_val
    
    # Interpolate to get f and D at the requested a_values
    f_results = {}
    D_results = {}
    for a_j in sorted_a:
        x_j = np.log(a_j)
        # Find position in x_fine
        idx = np.searchsorted(x_fine, x_j)
        if idx == 0:
            f_j = f_array[0]
            D_j = D_array[0]
        elif idx >= n_fine:
            f_j = f_array[-1]
            D_j = D_array[-1]
        else:
            # Linear interpolation
            x0, x1 = x_fine[idx-1], x_fine[idx]
            t = (x_j - x0) / (x1 - x0) if x1 != x0 else 0
            f_j = (1-t)*f_array[idx-1] + t*f_array[idx]
            D_j = (1-t)*D_array[idx-1] + t*D_array[idx]
        
        f_results[a_j] = f_j
        D_results[a_j] = D_j
    
    return f_results, D_results

def _precompute_references(models, a_values):
    """Precompute reference f and D values for all models and a values."""
    ref_f = {}
    ref_D = {}
    
    for (Om0, w0, wa) in models:
        f_res, D_res = _solve_reference_ode(Om0, w0, wa, a_values)
        ref_f[(Om0, w0, wa)] = f_res
        ref_D[(Om0, w0, wa)] = D_res
    
    return ref_f, ref_D

def _evaluate_score_fast(c, models, a_values, ref_f, ref_D, use_smooth=False, smooth_beta=50.0):
    """Compute score using precomputed references (vectorized)."""
    all_errs = []
    
    for (Om0, w0, wa) in models:
        # Filter valid a values
        valid_a = [a for a in a_values if 0 < a <= 1.0]
        if not valid_a:
            continue
        
        a_arr = np.array(valid_a, dtype=float)
        
        # Compute approximations
        f_approx = _compute_f_vec(c, a_arr, Om0, w0, wa)
        D_approx = _compute_D_normalized_vec(c, a_arr, Om0, w0, wa)
        
        # Get references
        f_ref_arr = np.array([ref_f[(Om0, w0, wa)].get(a, np.nan) for a in valid_a])
        D_ref_arr = np.array([ref_D[(Om0, w0, wa)].get(a, np.nan) for a in valid_a])
        
        # Compute errors
        valid_mask = ~np.isnan(f_ref_arr) & ~np.isnan(D_ref_arr)
        if np.any(valid_mask):
            err_f = np.abs(f_approx[valid_mask] - f_ref_arr[valid_mask]) / np.maximum(np.abs(f_ref_arr[valid_mask]), 1e-10)
            err_D = np.abs(D_approx[valid_mask] - D_ref_arr[valid_mask]) / np.maximum(np.abs(D_ref_arr[valid_mask]), 1e-10)
            all_errs.extend(err_f.tolist())
            all_errs.extend(err_D.tolist())
    
    if not all_errs:
        return 1e6
    
    all_errs = np.array(all_errs)
    
    if use_smooth:
        beta = smooth_beta
        scaled = all_errs * beta
        max_scaled = np.max(scaled)
        if max_scaled > 700:
            smooth_max = max_scaled / beta
        else:
            smooth_max = (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))) / beta
        return 100.0 * smooth_max
    else:
        return 100.0 * np.max(all_errs)

def _gamma_constraint_check(c, models, a_values):
    """Check that gamma is in [0.1, 1.2] at all locations (vectorized)."""
    for (Om0, w0, wa) in models:
        valid_a = [a for a in a_values if 0 < a <= 1.0]
        if not valid_a:
            continue
        a_arr = np.array(valid_a, dtype=float)
        gamma = _compute_gamma_vec(c, a_arr, Om0, w0, wa)
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return False
    return True

def _penalized_score_fast(c, models, a_values, ref_f, ref_D, use_smooth=False, smooth_beta=50.0):
    """Score with penalty for gamma constraint violation using precomputed refs."""
    base_score = _evaluate_score_fast(c, models, a_values, ref_f, ref_D, use_smooth, smooth_beta)
    if not _gamma_constraint_check(c, models, a_values):
        return base_score + 1000.0
    return base_score

def _optimize_coeffs():
    """Optimize the six coefficients using precomputed references."""
    start_time = time.time()
    deadline = start_time + 80.0
    
    models = [
        (0.3, 0.0, 0.0),
        (0.3, -0.5, 0.5),
        (0.3, -1.2, 0.8),
        (0.25, -0.9, 0.3),
        (0.35, -0.7, -0.2),
        (0.28, -1.0, 0.6),
        (0.3, -0.8, 0.0),
        (0.27, -1.1, 0.4),
    ]
    
    a_values = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    
    bounds = [(-10, 10)] * 6
    
    # Precompute references
    print("Precomputing references...")
    ref_f, ref_D = _precompute_references(models, a_values)
    print(f"References precomputed in {time.time() - start_time:.1f}s")
    
    best_c = None
    best_score = 1e6
    
    try:
        # Phase 1: Differential Evolution with more iterations and larger population
        de_result = differential_evolution(
            _penalized_score_fast,
            bounds,
            args=(models, a_values, ref_f, ref_D, False, 50.0),
            maxiter=150,
            popsize=25,
            tol=1e-10,
            seed=42,
            polish=False,
            init='sobol'
        )
        if de_result.fun < best_score:
            best_score = de_result.fun
            best_c = de_result.x
        print(f"DE done in {time.time() - start_time:.1f}s, score={best_score:.4f}")
        
        # Phase 2: Nelder-Mead polish with smooth surrogate
        if time.time() < deadline - 20.0:
            nm_result = minimize(
                _penalized_score_fast,
                best_c,
                args=(models, a_values, ref_f, ref_D, True, 30.0),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if nm_result.fun < best_score:
                best_score = nm_result.fun
                best_c = nm_result.x
            print(f"NM smooth done in {time.time() - start_time:.1f}s, score={best_score:.4f}")
        
        # Phase 3: High-precision polish with exact objective
        if time.time() < deadline - 5.0:
            polish_result = minimize(
                _penalized_score_fast,
                best_c,
                args=(models, a_values, ref_f, ref_D, False, 50.0),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14}
            )
            if polish_result.fun < best_score:
                best_score = polish_result.fun
                best_c = polish_result.x
            print(f"NM exact done in {time.time() - start_time:.1f}s, score={best_score:.4f}")
        
    except Exception as e:
        print(f"Optimization error: {e}")
        best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    best_c = np.clip(best_c, -10.0, 10.0)
    
    return list(best_c)

# Pre-compute optimized coefficients at import time
_OPTIMIZED_COEFFS = _optimize_coeffs()

def construct():
    return _OPTIMIZED_COEFFS
# EVOLVE-BLOCK-END
