"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_Om_a(a, w0, wa, Om0=0.3):
    """Compute Omega_m(a) for flat CPL universe."""
    u = 1.0 + w0 + wa * (1.0 - a)
    if u <= 0.01:
        return None
    Om_de0 = 1.0 - Om0
    denom = Om0 * a**3 + Om_de0 / u
    if denom <= 0:
        return None
    Om_a = Om0 * a**3 / denom
    if Om_a <= 0 or Om_a >= 1:
        return None
    return Om_a

def _target_gamma_LW(Om_a, u):
    """Linder-Waugh target gamma."""
    if u <= 0.01 or Om_a <= 0:
        return None
    return (5.0/7.0) * (Om_a / u) ** 0.38

def _gamma_from_coeffs(coeffs, q, u, wa, a):
    """Compute gamma from coefficients."""
    c = coeffs
    return c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q

def _build_eval_grid(w0_vals, wa_vals, a_vals, Om0=0.3):
    """Build the evaluation grid: returns arrays of q, u, wa, a, Om_a."""
    qs, us, was, as_, Oms = [], [], [], [], []
    for w0 in w0_vals:
        for wa in wa_vals:
            for a in a_vals:
                u = 1.0 + w0 + wa * (1.0 - a)
                if u <= 0.05:
                    continue
                Om_a = _compute_Om_a(a, w0, wa, Om0)
                if Om_a is None:
                    continue
                q = 1.0 - Om_a
                qs.append(q)
                us.append(u)
                was.append(wa)
                as_.append(a)
                Oms.append(Om_a)
    return (np.array(qs), np.array(us), np.array(was), np.array(as_), np.array(Oms))

def _compute_ln_f_and_ln_D(coeffs, q, u, wa, a, Om_a):
    """
    Compute ln(f) = gamma * ln(Om_a) and ln(D(a)/D(1)) = -integral_ln(a)^0 f(exp(x)) dx.
    Use trapezoidal rule for the integral.
    """
    gamma = _gamma_from_coeffs(coeffs, q, u, wa, a)
    # Enforce gamma in [0.1, 1.2] for evaluation
    gamma_clipped = np.clip(gamma, 0.1, 1.2)
    
    ln_Om = np.log(Om_a)
    ln_f = gamma_clipped * ln_Om
    
    # For the integral, we need f at each point: f = Om_a^gamma
    f_vals = Om_a ** gamma_clipped
    
    # The integral is from ln(a) to 0 of f(exp(x)) dx
    # x goes from ln(a) to 0, so let's parametrize: x = ln(a) + t*(0 - ln(a)) = ln(a)*(1-t) for t in [0,1]
    # dx = -ln(a) dt, so integral = -ln(a) * integral_0^1 f(exp(ln(a)*(1-t))) dt
    # Actually, let's just compute it directly: integral from ln(a) to 0 of f(exp(x)) dx
    # = - integral from 0 to ln(a) of f(exp(x)) dx
    # For a < 1, ln(a) < 0, so integral from ln(a) to 0 is positive
    # Let's use: for each a, compute integral numerically
    # Since we have a grid of a values, we can precompute the integral for each a
    
    # Simpler approach: for each a, the integral is -sum using trapezoidal rule
    # But we need f at intermediate points. Since we're on a grid, let's approximate.
    # Actually, the evaluator computes D(a)/D(1) = exp(-integral_ln(a)^0 f(exp(x)) dx)
    # So ln(D) = -integral_ln(a)^0 f(exp(x)) dx
    
    # For a given a, integral from ln(a) to 0 of f(exp(x)) dx
    # Let x = ln(a) + s*(0 - ln(a)) = ln(a)*(1-s) for s in [0,1]
    # dx = -ln(a) ds
    # integral = integral_0^1 f(exp(ln(a)*(1-s))) * (-ln(a)) ds = -ln(a) * integral_0^1 f(exp(ln(a)*(1-s))) ds
    # For a < 1, ln(a) < 0, so -ln(a) > 0, and the integral is positive
    
    # We'll compute this for each a in the grid
    ln_D = np.zeros(len(a))
    for i in range(len(a)):
        a_i = a[i]
        ln_a = np.log(a_i)
        if abs(ln_a) < 1e-10:
            ln_D[i] = 0.0
            continue
        
        # Integrate from ln(a) to 0: use trapezoidal rule with a few points
        # x goes from ln_a to 0
        n_pts = 20
        xs = np.linspace(ln_a, 0.0, n_pts)
        # f(exp(x)) = Om(exp(x))^gamma
        # We need Om at these x values, which is tricky
        # Approximate: use the fact that at each x, a_x = exp(x), and we can compute Om(a_x)
        # But we don't have a direct formula for arbitrary a_x in the grid
        # Instead, let's use a simpler approach: since we're fitting, we can use the 
        # analytical approximation that the integral is approximately related to f at the endpoints
        
        # For a better approach, let's compute Om(a_x) for x in the integration range
        # This requires knowing w0, wa, which we have from the grid
        # But we're in a vectorized context... Let's do it per-point
        pass
    
    # Alternative: use the fact that for the purpose of fitting, we can use a 
    # simplified integral approximation. The key insight is that the error in ln(D)
    # is dominated by the error in gamma, so we can use a linearized approximation.
    # 
    # Actually, let's just compute the integral properly for each point in the grid.
    # We'll need w0 and wa for each point, so let's restructure.
    
    return ln_f, ln_D

def _fit_coefficients():
    """Fit the six growth-index coefficients via constrained Chebyshev approximation."""
    t_start = time.time()
    deadline = 80.0  # Safety margin before 90s kill
    
    # Grid of CPL parameters and scale factors
    w0_vals = np.linspace(-1.2, -0.8, 20)
    wa_vals = np.linspace(-0.5, 0.5, 20)
    a_vals = np.linspace(0.3, 1.0, 40)
    
    Omega_m0 = 0.3
    
    # Build evaluation grid
    q, u, wa_arr, a_arr, Om_a = _build_eval_grid(w0_vals, wa_vals, a_vals, Omega_m0)
    
    if len(q) < 10:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Design matrix
    A = np.column_stack([np.ones_like(q), q, q**2, u, wa_arr * a_arr, u * q])
    
    # Initial guess from least squares to Linder-Waugh
    targets = (5.0/7.0) * (Om_a / u) ** 0.38
    coeffs_ls, _, _, _ = np.linalg.lstsq(A, targets, rcond=None)
    coeffs_ls = np.clip(coeffs_ls, -10.0, 10.0)
    
    def _check_gamma_range(coeffs_to_check):
        gamma = _gamma_from_coeffs(coeffs_to_check, q, u, wa_arr, a_arr)
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return False
        return True
    
    if not _check_gamma_range(coeffs_ls):
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Objective: minimize max |ln_f - ln_f_ref| and max |ln_D - ln_D_ref|
    # For the reference, we use the Linder-Waugh gamma to compute f and D
    gamma_ref = targets
    ln_f_ref = gamma_ref * np.log(Om_a)
    
    # For ln_D, we need to compute the integral. Let's use a simplified approach:
    # The integral from ln(a) to 0 of f(exp(x)) dx can be approximated.
    # For the purpose of fitting, we'll use the fact that the error in ln(D)
    # is approximately related to the error in gamma.
    
    # Actually, let's compute ln_D properly. For each point, we need to integrate.
    # This is expensive, so let's use a coarser grid for the integral.
    
    # Precompute the integral for the reference and for the model
    def _compute_ln_D_array(coeffs, q_arr, u_arr, wa_arr_, a_arr_, Om_a_arr, w0_arr, wa_arr_2):
        """Compute ln(D(a)/D(1)) for an array of points."""
        gamma = _gamma_from_coeffs(coeffs, q_arr, u_arr, wa_arr_, a_arr_)
        gamma_clipped = np.clip(gamma, 0.1, 1.2)
        f_vals = Om_a_arr ** gamma_clipped
        
        ln_D = np.zeros(len(a_arr_))
        for i in range(len(a_arr_)):
            a_i = a_arr_[i]
            ln_a = np.log(a_i)
            if abs(ln_a) < 1e-10:
                ln_D[i] = 0.0
                continue
            
            # Integrate from ln_a to 0 using trapezoidal rule
            # We need f(exp(x)) for x in [ln_a, 0]
            # This requires computing Om at arbitrary a values, which needs w0, wa
            # For simplicity, use a rough approximation: 
            # integral ≈ (0 - ln_a) * (f(a) + f(1))/2 = -ln_a * (f(a) + 1)/2
            # since f(1) = Om(1)^gamma = 1^gamma = 1
            ln_D[i] = -ln_a * (f_vals[i] + 1.0) / 2.0
        
        return ln_D
    
    # We need w0 for each point in the grid
    w0_grid = []
    for w0 in w0_vals:
        for wa in wa_vals:
            for a in a_vals:
                u = 1.0 + w0 + wa * (1.0 - a)
                if u <= 0.05:
                    continue
                Om_a_i = _compute_Om_a(a, w0, wa, Omega_m0)
                if Om_a_i is None:
                    continue
                w0_grid.append(w0)
    w0_grid = np.array(w0_grid)
    
    # Compute reference ln_D
    ln_D_ref = _compute_ln_D_array(coeffs_ls, q, u, wa_arr, a_arr, Om_a, w0_grid, wa_arr)
    
    # Objective function: minimize max of |ln_f - ln_f_ref| and |ln_D - ln_D_ref|
    def _objective(coeffs_vec):
        gamma = _gamma_from_coeffs(coeffs_vec, q, u, wa_arr, a_arr)
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e10
        
        gamma_clipped = np.clip(gamma, 0.1, 1.2)
        ln_f = gamma_clipped * np.log(Om_a)
        ln_D = _compute_ln_D_array(coeffs_vec, q, u, wa_arr, a_arr, Om_a, w0_grid, wa_arr)
        
        err_f = np.abs(ln_f - ln_f_ref)
        err_D = np.abs(ln_D - ln_D_ref)
        
        return np.max(np.concatenate([err_f, err_D]))
    
    # Use Nelder-Mead for the initial optimization
    bounds = [(-10, 10)] * 6
    
    best_coeffs = coeffs_ls.copy()
    best_obj = _objective(coeffs_ls)
    
    if time.time() - t_start < deadline - 10:
        try:
            result = minimize(_objective, coeffs_ls, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < best_obj:
                best_coeffs = np.clip(result.x, -10.0, 10.0)
                best_obj = result.fun
        except Exception:
            pass
    
    # Polish with L-BFGS-B using a smooth surrogate (log-sum-exp)
    if time.time() - t_start < deadline - 5:
        def _smooth_objective(coeffs_vec):
            gamma = _gamma_from_coeffs(coeffs_vec, q, u, wa_arr, a_arr)
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return 1e10
            
            gamma_clipped = np.clip(gamma, 0.1, 1.2)
            ln_f = gamma_clipped * np.log(Om_a)
            ln_D = _compute_ln_D_array(coeffs_vec, q, u, wa_arr, a_arr, Om_a, w0_grid, wa_arr)
            
            err_f = np.abs(ln_f - ln_f_ref)
            err_D = np.abs(ln_D - ln_D_ref)
            all_errs = np.concatenate([err_f, err_D])
            
            # Log-sum-exp for smooth max
            scale = 100.0
            max_err = np.max(all_errs)
            if max_err < 1e-10:
                return max_err
            shifted = all_errs / scale
            log_sum_exp = max_err / scale + np.log(np.sum(np.exp(shifted - np.max(shifted))))
            return log_sum_exp * scale
        
        try:
            result2 = minimize(_smooth_objective, best_coeffs, method='L-BFGS-B',
                              bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12})
            if result2.fun < _objective(best_coeffs):
                candidate = np.clip(result2.x, -10.0, 10.0)
                if _check_gamma_range(candidate):
                    best_coeffs = candidate
        except Exception:
            pass
    
    # Final check
    if not _check_gamma_range(best_coeffs):
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    return best_coeffs.tolist()

def construct():
    """Construct the six growth-index coefficients."""
    t_start = time.time()
    
    try:
        coeffs = _fit_coefficients()
    except Exception:
        coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Ensure finite and in range
    for i in range(6):
        if not np.isfinite(coeffs[i]) or coeffs[i] < -10 or coeffs[i] > 10:
            coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
            break
    
    return coeffs
# EVOLVE-BLOCK-END
