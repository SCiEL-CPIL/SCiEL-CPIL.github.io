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

def _gamma_from_coeffs(coeffs, q, u, wa, a):
    """Compute gamma from coefficients."""
    c = coeffs
    return c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q

def _target_gamma_LW(Om_a, u):
    """Linder-Waugh target gamma."""
    if u <= 0.01 or Om_a <= 0:
        return None
    return (5.0/7.0) * (Om_a / u) ** 0.38

def _fit_coefficients():
    """Fit the six growth-index coefficients via least squares to the Linder-Waugh formula,
    then refine with a bounded optimization to minimize worst-case error."""
    t_start = time.time()
    deadline = 80.0  # Safety margin before 90s kill
    
    # Grid of CPL parameters and scale factors - dense grid for fitting
    w0_vals = np.linspace(-1.2, -0.8, 40)
    wa_vals = np.linspace(-0.5, 0.5, 40)
    a_vals = np.linspace(0.2, 1.0, 80)
    
    Omega_m0 = 0.3
    Omega_de0 = 1.0 - Omega_m0
    
    # Build the design matrix and target vector
    rows = []
    targets = []
    
    for w0 in w0_vals:
        for wa in wa_vals:
            for a in a_vals:
                if time.time() - t_start > deadline:
                    break
                
                # Compute u = 1 + w0 + wa*(1-a)
                u = 1.0 + w0 + wa * (1.0 - a)
                if u <= 0.05:  # Avoid singularities
                    continue
                
                # Compute Omega_m(a) for flat CPL universe
                Om_a = _compute_Om_a(a, w0, wa, Omega_m0)
                if Om_a is None:
                    continue
                
                q = 1.0 - Om_a
                target_gamma = _target_gamma_LW(Om_a, u)
                if target_gamma is None:
                    continue
                
                # Design matrix row: [1, q, q^2, u, wa*a, u*q]
                row = np.array([1.0, q, q**2, u, wa * a, u * q])
                rows.append(row)
                targets.append(target_gamma)
            
            if time.time() - t_start > deadline:
                break
        if time.time() - t_start > deadline:
            break
    
    if len(rows) < 10:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    A = np.array(rows)
    b = np.array(targets)
    
    # Solve least squares
    coeffs, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    
    # Clip to [-10, 10]
    coeffs = np.clip(coeffs, -10.0, 10.0)
    
    # Verify gamma stays in [0.1, 1.2] on the training grid
    def _check_gamma_range(coeffs_to_check):
        for w0 in w0_vals:
            for wa in wa_vals:
                for a in a_vals:
                    u = 1.0 + w0 + wa * (1.0 - a)
                    if u <= 0.05:
                        continue
                    Om_a = _compute_Om_a(a, w0, wa, Omega_m0)
                    if Om_a is None:
                        continue
                    q = 1.0 - Om_a
                    gamma = _gamma_from_coeffs(coeffs_to_check, q, u, wa, a)
                    if gamma < 0.1 or gamma > 1.2:
                        return False
        return True
    
    if not _check_gamma_range(coeffs):
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Local refinement: minimize worst-case relative error using scipy
    # We use a subset of points for the optimization to keep it fast
    # but we evaluate the objective on the full grid to be safe? 
    # No, L-BFGS-B with 160k points is too slow. 
    # We will subsample for the optimization objective.
    
    # Subsample indices
    n_total = len(rows)
    n_sub = min(5000, n_total)
    indices = np.random.choice(n_total, n_sub, replace=False)
    A_sub = A[indices]
    b_sub = b[indices]
    
    def soft_max_obj(coeffs_vec, temperature=0.01):
        if time.time() - t_start > deadline:
            return 1e10, None
        
        gamma_pred = A_sub @ coeffs_vec
        errors = np.abs(gamma_pred - b_sub)
        
        # Penalty for gamma range violations
        gamma_min = np.min(gamma_pred)
        gamma_max = np.max(gamma_pred)
        penalty = 0.0
        if gamma_min < 0.1:
            penalty += (0.1 - gamma_min) ** 2 * 1000
        if gamma_max > 1.2:
            penalty += (gamma_max - 1.2) ** 2 * 1000
            
        scaled = errors / temperature
        max_s = np.max(scaled)
        lse = max_s + np.log(np.sum(np.exp(scaled - max_s)))
        return lse * temperature + penalty, None
        
    def soft_max_grad(coeffs_vec, temperature=0.01):
        if time.time() - t_start > deadline:
            return None
            
        gamma_pred = A_sub @ coeffs_vec
        errors = np.abs(gamma_pred - b_sub)
        
        gamma_min = np.min(gamma_pred)
        gamma_max = np.max(gamma_pred)
        
        scaled = errors / temperature
        max_s = np.max(scaled)
        exp_v = np.exp(scaled - max_s)
        weights = exp_v / np.sum(exp_v)
        signs = np.sign(gamma_pred - b_sub)
        grad = A_sub.T @ (weights * signs)
        
        if gamma_min < 0.1:
            viol = (gamma_pred < 0.1).astype(float)
            grad += 2.0 * 1000 * (0.1 - gamma_min) * A_sub.T @ viol / len(gamma_pred)
        if gamma_max > 1.2:
            viol = (gamma_pred > 1.2).astype(float)
            grad += 2.0 * 1000 * (gamma_max - 1.2) * A_sub.T @ viol / len(gamma_pred)
            
        return grad
        
    bounds = [(-10, 10)] * 6
    
    try:
        result = minimize(soft_max_obj, coeffs, method='L-BFGS-B', bounds=bounds,
                         jac=lambda x: soft_max_grad(x),
                         options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
        if result.success and result.fun < 1e9:
            refined = np.clip(result.x, -10.0, 10.0)
            if _check_gamma_range(refined):
                coeffs = refined
    except Exception:
        pass
        
    # Polish with tighter temperature
    if time.time() - t_start < deadline - 5:
        def polish_obj(coeffs_vec, temperature=0.001):
            if time.time() - t_start > deadline:
                return 1e10, None
                
            gamma_pred = A_sub @ coeffs_vec
            errors = np.abs(gamma_pred - b_sub)
            
            gamma_min = np.min(gamma_pred)
            gamma_max = np.max(gamma_pred)
            penalty = 0.0
            if gamma_min < 0.1:
                penalty += (0.1 - gamma_min) ** 2 * 5000
            if gamma_max > 1.2:
                penalty += (gamma_max - 1.2) ** 2 * 5000
                
            scaled = errors / temperature
            max_s = np.max(scaled)
            lse = max_s + np.log(np.sum(np.exp(scaled - max_s)))
            return lse * temperature + penalty, None
            
        def polish_grad(coeffs_vec, temperature=0.001):
            if time.time() - t_start > deadline:
                return None
                
            gamma_pred = A_sub @ coeffs_vec
            errors = np.abs(gamma_pred - b_sub)
            
            gamma_min = np.min(gamma_pred)
            gamma_max = np.max(gamma_pred)
            
            scaled = errors / temperature
            max_s = np.max(scaled)
            exp_v = np.exp(scaled - max_s)
            weights = exp_v / np.sum(exp_v)
            signs = np.sign(gamma_pred - b_sub)
            grad = A_sub.T @ (weights * signs)
            
            if gamma_min < 0.1:
                viol = (gamma_pred < 0.1).astype(float)
                grad += 2.0 * 5000 * (0.1 - gamma_min) * A_sub.T @ viol / len(gamma_pred)
            if gamma_max > 1.2:
                viol = (gamma_pred > 1.2).astype(float)
                grad += 2.0 * 5000 * (gamma_max - 1.2) * A_sub.T @ viol / len(gamma_pred)
                
            return grad
            
        try:
            result2 = minimize(polish_obj, coeffs, method='L-BFGS-B', bounds=bounds,
                              jac=lambda x: polish_grad(x),
                              options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12})
            if result2.success and result2.fun < 1e9:
                refined2 = np.clip(result2.x, -10.0, 10.0)
                if _check_gamma_range(refined2):
                    coeffs = refined2
        except Exception:
            pass

    if not _check_gamma_range(coeffs):
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    return coeffs.tolist()

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
