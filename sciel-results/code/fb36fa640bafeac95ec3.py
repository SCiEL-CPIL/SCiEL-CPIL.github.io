"""Refined Cohn-Goncalves Laguerre construction for Fourier sign uncertainty."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_laguerre, eval_genlaguerre

# EVOLVE-BLOCK-START
def _compute_c4(roots):
    """Compute the C4 upper bound = 1/sqrt(prod(roots))."""
    if len(roots) != 12:
        return float('inf')
    if np.any(roots <= 0):
        return float('inf')
    prod = np.prod(roots)
    if prod <= 0 or not np.isfinite(prod):
        return float('inf')
    return 1.0 / np.sqrt(prod)

def _log_c4(roots):
    """Compute log(C4) = -0.5 * sum(log(roots))."""
    if len(roots) != 12:
        return float('inf')
    if np.any(roots <= 0):
        return float('inf')
    return -0.5 * np.sum(np.log(roots))

def _laguerre_residual(roots, max_order=24):
    """
    Compute the residual when trying to represent P(x) = prod_i (x - r_i)^2
    as a linear combination of Laguerre polynomials L_k^(1)(x) for k=0..max_order.
    
    The key insight: for the Cohn-Goncalves construction, P(x) must be expressible
    in the Laguerre basis. We check this by computing the coefficients via
    generalized Laguerre expansion and measuring the residual.
    
    For alpha=1, the expansion coefficient is:
    c_k = (1/(k!)) * integral_0^inf P(x) L_k^(1)(x) e^{-x} x dx
    
    But more practically, we can use the fact that if P(x) = sum_{k=0}^{24} c_k L_k^(1)(x),
    then the "tail" beyond degree 12 should be small for a valid construction.
    
    We compute this by evaluating P at quadrature points and checking orthogonality
    against L_k^(1) for k > 12.
    """
    if len(roots) != 12 or np.any(roots <= 0):
        return float('inf')
    
    # Use Gauss-Laguerre quadrature with enough points for accuracy
    # For degree 24 polynomial, we need at least 25 points for exact integration
    n_quad = max(50, 2 * max_order + 10)
    x_quad, w_quad = roots_laguerre(n_quad)
    
    # Evaluate P(x) = prod (x - r_i)^2 at quadrature points
    P_vals = np.ones_like(x_quad)
    for r in roots:
        P_vals *= (x_quad - r) ** 2
    
    # Compute the residual: the component of P in the subspace spanned by L_k^(1) for k > 12
    # This is the part that cannot be represented by a degree-12 Laguerre combination
    # 
    # The projection coefficient onto L_k^(1) is:
    # c_k = (1 / h_k) * sum_i w_i * P(x_i) * L_k^(1)(x_i) * x_i  [for alpha=1, weight is x*e^{-x}]
    # where h_k = k! for alpha=1... actually h_k = Gamma(k+alpha+1)/Gamma(alpha+1) = (k+1)! for alpha=1
    # Wait, for generalized Laguerre L_k^(alpha), the orthogonality is:
    # integral_0^inf L_j^(alpha)(x) L_k^(alpha)(x) x^alpha e^{-x} dx = Gamma(k+alpha+1)/k! delta_jk
    # For alpha=1: integral = (k+1)! / k! * delta_jk = (k+1) * delta_jk
    # So h_k = k+1 for alpha=1
    
    # Actually let me be more careful. The standard Laguerre L_k(x) = L_k^(0)(x) has:
    # integral_0^inf L_j(x) L_k(x) e^{-x} dx = k! delta_jk
    # For L_k^(alpha)(x): integral_0^inf L_j^(alpha)(x) L_k^(alpha)(x) x^alpha e^{-x} dx = Gamma(k+alpha+1)/k! delta_jk
    # For alpha=1: Gamma(k+2)/k! = (k+1)!/k! = k+1
    
    # So the projection coefficient is:
    # c_k = (1/(k+1)) * integral_0^inf P(x) L_k^(1)(x) x e^{-x} dx
    #      = (1/(k+1)) * sum_i w_i * P(x_i) * L_k^(1)(x_i) * x_i
    
    # The residual (part not in span of L_0..L_12) is:
    # R(x) = P(x) - sum_{k=0}^{12} c_k L_k^(1)(x)
    # We want ||R||_weighted to be small.
    
    # Compute coefficients c_k for k=0..max_order
    c = np.zeros(max_order + 1)
    for k in range(max_order + 1):
        Lk_vals = eval_genlaguerre(k, 1, x_quad)
        c[k] = (1.0 / (k + 1)) * np.sum(w_quad * P_vals * Lk_vals * x_quad)
    
    # Compute the projection P_proj = sum_{k=0}^{12} c_k L_k^(1)(x)
    P_proj = np.zeros_like(x_quad)
    for k in range(13):
        Lk_vals = eval_genlaguerre(k, 1, x_quad)
        P_proj += c[k] * Lk_vals
    
    # Compute weighted residual norm
    residual = P_vals - P_proj
    resid_norm_sq = np.sum(w_quad * residual ** 2 * x_quad)
    
    # Also compute the norm of P for normalization
    P_norm_sq = np.sum(w_quad * P_vals ** 2 * x_quad)
    
    if P_norm_sq <= 0:
        return float('inf')
    
    # Return relative residual
    return np.sqrt(resid_norm_sq / P_norm_sq)

def _constrained_objective(x, penalty_weight=1e6):
    """
    Objective function for optimization in log-space.
    x = [log(r_1), log(r_2), ..., log(r_12)] with r_1 < r_2 < ... < r_12
    
    Returns: log(C4) + penalty * laguerre_residual
    """
    roots = np.exp(x)
    roots = np.sort(roots)
    
    # Check for valid roots
    if np.any(roots <= 0) or np.any(~np.isfinite(roots)):
        return 1e10
    
    # Check strict ordering
    if np.any(np.diff(roots) < 1e-12):
        return 1e10
    
    # Compute log(C4)
    log_c4 = _log_c4(roots)
    if not np.isfinite(log_c4):
        return 1e10
    
    # Compute Laguerre residual
    residual = _laguerre_residual(roots)
    if not np.isfinite(residual):
        return 1e10
    
    # Penalize large residuals
    penalty = penalty_weight * residual ** 2
    
    return log_c4 + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s kill
    
    best_score = float('inf')
    best_roots = None
    
    # Initial guess: use scaled Laguerre roots of L_12^(1)
    # The roots of L_n^(alpha) are in (0, infinity)
    # For the Cohn-Goncalves construction, we need double roots
    # Start with the Laguerre roots as initial guess
    
    laguerre_roots_12 = roots_laguerre(12, 1)[0]  # L_12^(1) roots
    # Scale to try to make them work as double roots
    # The product should be maximized, so try different scales
    
    best_log_c4 = float('inf')
    best_x0 = None
    
    # Try several initial scales
    for scale in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
        if time.time() > deadline:
            break
        roots_init = laguerre_roots_12 * scale
        if np.any(roots_init <= 0):
            continue
        
        x0 = np.log(roots_init)
        log_c4_val = _log_c4(roots_init)
        residual_val = _laguerre_residual(roots_init)
        
        if np.isfinite(log_c4_val) and np.isfinite(residual_val):
            obj_val = log_c4_val + 1e6 * residual_val ** 2
            if obj_val < best_log_c4:
                best_log_c4 = obj_val
                best_x0 = x0.copy()
                best_roots = roots_init.copy()
    
    # If we have a good initial guess, do local optimization
    if best_x0 is not None and time.time() < deadline - 60:
        try:
            # Use L-BFGS-B with bounds
            bounds = [(np.log(0.01), np.log(10000.0))] * 12
            
            result = minimize(
                _constrained_objective,
                best_x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.success and time.time() < deadline:
                roots_opt = np.sort(np.exp(result.x))
                if np.all(roots_opt > 0) and np.all(np.diff(roots_opt) > 1e-10):
                    score = _compute_c4(roots_opt)
                    residual = _laguerre_residual(roots_opt)
                    if np.isfinite(score) and score < best_score:
                        best_score = score
                        best_roots = roots_opt.copy()
        except Exception:
            pass
    
    # Try Nelder-Mead from multiple starting points
    if time.time() < deadline - 30:
        # Generate diverse starting points
        starting_points = []
        
        # Scaled Laguerre roots
        for scale in [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]:
            roots_s = laguerre_roots_12 * scale
            if np.all(roots_s > 0):
                starting_points.append(np.log(roots_s))
        
        # Geometric progressions with Laguerre-like spread
        for ratio in [1.5, 2.0, 2.5]:
            for base in [1.0, 2.0, 5.0]:
                roots_g = base * (ratio ** np.arange(12))
                if np.all(roots_g > 0) and np.all(roots_g < 1e6):
                    starting_points.append(np.log(roots_g))
        
        # Power-law
        for alpha in [1.5, 2.0, 2.5]:
            for base in [1.0, 3.0, 10.0]:
                roots_p = base * (np.arange(1, 13) ** alpha)
                if np.all(roots_p > 0) and np.all(roots_p < 1e6):
                    starting_points.append(np.log(roots_p))
        
        for x0 in starting_points:
            if time.time() > deadline - 10:
                break
            try:
                result = minimize(
                    _constrained_objective,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                if time.time() > deadline - 5:
                    break
                if result.success:
                    roots_nm = np.sort(np.exp(result.x))
                    if np.all(roots_nm > 0) and np.all(np.diff(roots_nm) > 1e-10):
                        score = _compute_c4(roots_nm)
                        if np.isfinite(score) and score < best_score:
                            best_score = score
                            best_roots = roots_nm.copy()
            except Exception:
                continue
    
    # Final polish: if we have a good candidate, do a high-precision local solve
    if best_roots is not None and time.time() < deadline - 15:
        try:
            x0_final = np.log(best_roots)
            result = minimize(
                _constrained_objective,
                x0_final,
                method='L-BFGS-B',
                bounds=[(np.log(0.001), np.log(100000.0))] * 12,
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if result.success:
                roots_final = np.sort(np.exp(result.x))
                if np.all(roots_final > 0) and np.all(np.diff(roots_final) > 1e-10):
                    score = _compute_c4(roots_final)
                    if np.isfinite(score) and score < best_score:
                        best_score = score
                        best_roots = roots_final.copy()
        except Exception:
            pass
    
    # Fallback: use the best scaled Laguerre roots we found
    if best_roots is None or not np.all(best_roots > 0):
        best_roots = laguerre_roots_12 * 3.0
    
    return list(best_roots)
# EVOLVE-BLOCK-END
