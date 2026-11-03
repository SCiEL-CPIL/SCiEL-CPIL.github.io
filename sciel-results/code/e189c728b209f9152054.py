"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar, differential_evolution
import time

def _compute_c4_exact(log_roots):
    """
    Compute C4 = [∫ P(x)^2 e^{-x} dx] / [∫ P(x) e^{-x} dx]^2
    where P(x) = prod_{i=1}^{12} (x - r_i)^2, r_i = exp(log_roots_i).
    
    Uses exact moment computation:
      ∫ x^k P(x) e^{-x} dx = sum over monomial coefficients.
    
    P(x) = prod (x^2 - 2 r_i x + r_i^2)
    We compute the coefficients of P(x) up to degree 24 via polynomial multiplication,
    then use the fact that ∫ x^m e^{-x} dx = m! (Gamma(m+1)).
    """
    roots = np.exp(log_roots)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf
    
    # Build P(x) = prod_{i=1}^{12} (x - r_i)^2
    # Each factor is x^2 - 2*r_i*x + r_i^2
    # We'll do polynomial multiplication in coefficient form (highest degree first)
    # P has degree 24.
    
    # Start with P = 1
    coeffs = np.array([1.0])  # represents polynomial [1]
    
    for r in roots:
        # factor: x^2 - 2*r*x + r^2
        factor = np.array([1.0, -2.0 * r, r * r])
        # Multiply coeffs by factor (convolution)
        coeffs = np.convolve(coeffs, factor)
    
    # coeffs[k] is the coefficient of x^{24-k} (since we started with [1] and convolved)
    # Actually np.convolve([1],[a,b,c]) = [a,b,c], so coeffs[0] is coeff of x^24, coeffs[24] is constant.
    # Let's verify: after 12 convolutions with degree-2 polynomials, we have degree 24.
    # coeffs has length 25. coeffs[j] = coefficient of x^{24-j}.
    
    # ∫ x^m e^{-x} dx from 0 to ∞ = Gamma(m+1) = m!
    # ∫ P(x) e^{-x} dx = sum_{j=0}^{24} coeffs[j] * Gamma(24-j+1) = sum coeffs[j] * (24-j)!
    
    # Precompute factorials up to 24
    fact = np.ones(25)
    for k in range(1, 25):
        fact[k] = fact[k-1] * k
    
    # int_P = sum_{j=0}^{24} coeffs[j] * fact[24-j]
    # But wait: coeffs[j] is coeff of x^{24-j}, so ∫ x^{24-j} e^{-x} dx = (24-j)!
    # So int_P = sum_{j=0}^{24} coeffs[j] * fact[24-j]
    
    # For int_P2: P(x)^2 has degree 48. We need ∫ x^m e^{-x} dx = m! for m up to 48.
    # P(x)^2 = (sum_{j=0}^{24} coeffs[j] x^{24-j})^2
    # = sum_{j=0}^{24} sum_{k=0}^{24} coeffs[j]*coeffs[k] * x^{48-j-k}
    # ∫ P(x)^2 e^{-x} dx = sum_{j=0}^{24} sum_{k=0}^{24} coeffs[j]*coeffs[k] * (48-j-k)!
    
    # Precompute factorials up to 48
    fact48 = np.ones(49)
    for k in range(1, 49):
        fact48[k] = fact48[k-1] * k
    
    # Compute int_P
    # coeffs[j] corresponds to x^{24-j}
    int_P = 0.0
    for j in range(25):
        m = 24 - j  # power of x
        int_P += coeffs[j] * fact[m]
    
    if int_P <= 0 or np.isinf(int_P) or np.isnan(int_P):
        return np.inf
    
    # Compute int_P2
    int_P2 = 0.0
    for j in range(25):
        for k in range(25):
            m = 48 - j - k  # power of x in x^{24-j} * x^{24-k}
            if m >= 0:
                int_P2 += coeffs[j] * coeffs[k] * fact48[m]
    
    if int_P2 <= 0 or np.isinf(int_P2) or np.isnan(int_P2):
        return np.inf
    
    c4 = int_P2 / (int_P * int_P)
    return c4

def _objective_geometric(log_base, log_ratio):
    """
    Parameterize 12 roots as geometric sequence:
    r_i = exp(log_base + (i-1)*log_ratio) for i=1..12
    This is a 2-DOF parameterization that captures structured spacing.
    """
    log_roots = log_base + np.arange(12) * log_ratio
    return _compute_c4_exact(log_roots)

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    deadline = time.time() + 570  # Safety margin
    
    # Initial guess: standard Laguerre roots of L_12
    from scipy.special import roots_laguerre
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    
    best_roots = x0.copy()
    best_score = _compute_c4_exact(np.log(x0))
    
    # Phase 1: Optimize the geometric parameterization (2 DOF)
    # r_i = exp(log_base + (i-1) * log_ratio)
    # From x0, estimate log_base and log_ratio
    log_x0 = np.log(x0)
    # Fit: log_x0[i] ≈ a + i*b for i=0..11
    A = np.vstack([np.ones(12), np.arange(12)]).T
    params, residuals, rank, sv = np.linalg.lstsq(A, log_x0, rcond=None)
    init_log_base = params[0]
    init_log_ratio = params[1]
    
    if time.time() < deadline - 180:
        # Local optimization of the 2-DOF geometric model
        try:
            result = minimize_scalar(
                lambda lr: _objective_geometric(init_log_base, lr),
                bounds=(init_log_ratio - 0.5, init_log_ratio + 0.5),
                method='bounded',
                options={'xatol': 1e-10, 'maxiter': 500}
            )
            if result.fun < best_score:
                # Now optimize log_base for this log_ratio
                result2 = minimize_scalar(
                    lambda lb: _objective_geometric(lb, result.x),
                    bounds=(init_log_base - 0.5, init_log_base + 0.5),
                    method='bounded',
                    options={'xatol': 1e-10, 'maxiter': 500}
                )
                if result2.fun < best_score:
                    best_score = result2.fun
                    log_roots_opt = result2.x + np.arange(12) * result.x
                    best_roots = np.exp(log_roots_opt)
                    best_roots = np.sort(best_roots)
        except Exception:
            pass
        
        # Alternative: jointly optimize both parameters with a small grid + local
        if time.time() < deadline - 150:
            try:
                # Coarse grid search over log_ratio
                best_grid_score = best_score
                best_grid_params = (init_log_base, init_log_ratio)
                for lr in np.linspace(init_log_ratio - 0.3, init_log_ratio + 0.3, 21):
                    for lb in np.linspace(init_log_base - 0.3, init_log_base + 0.3, 21):
                        s = _objective_geometric(lb, lr)
                        if s < best_grid_score:
                            best_grid_score = s
                            best_grid_params = (lb, lr)
                
                if best_grid_score < best_score:
                    # Refine with local optimization
                    res = minimize_scalar(
                        lambda lr: _objective_geometric(best_grid_params[0], lr),
                        bounds=(best_grid_params[1] - 0.05, best_grid_params[1] + 0.05),
                        method='bounded',
                        options={'xatol': 1e-12, 'maxiter': 1000}
                    )
                    res2 = minimize_scalar(
                        lambda lb: _objective_geometric(lb, res.x),
                        bounds=(best_grid_params[0] - 0.05, best_grid_params[0] + 0.05),
                        method='bounded',
                        options={'xatol': 1e-12, 'maxiter': 1000}
                    )
                    if res2.fun < best_score:
                        best_score = res2.fun
                        log_roots_opt = res2.x + np.arange(12) * res.x
                        best_roots = np.exp(log_roots_opt)
                        best_roots = np.sort(best_roots)
            except Exception:
                pass
    
    # Phase 2: Full 12-DOF local polish using the exact moment method (fast now)
    if time.time() < deadline - 30:
        try:
            # Use L-BFGS-B in log-space with the exact fast objective
            from scipy.optimize import minimize
            
            def full_obj(log_roots):
                return _compute_c4_exact(log_roots)
            
            bounds = [(np.log(0.01), np.log(500.0))] * 12
            x_init = np.log(best_roots)
            
            result = minimize(
                full_obj,
                x_init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 50000}
            )
            
            if result.fun < best_score:
                candidate = np.exp(result.x)
                candidate = np.sort(candidate)
                if np.all(candidate > 0):
                    s = _compute_c4_exact(np.log(candidate))
                    if s < best_score:
                        best_score = s
                        best_roots = candidate
        except Exception:
            pass
    
    # Phase 3: One more round with perturbation if time allows
    if time.time() < deadline - 15:
        try:
            from scipy.optimize import minimize
            rng = np.random.default_rng(123)
            for _ in range(3):
                if time.time() > deadline - 10:
                    break
                pert = np.log(best_roots) + 0.05 * rng.standard_normal(12)
                res = minimize(
                    _compute_c4_exact,
                    pert,
                    method='L-BFGS-B',
                    bounds=[(np.log(0.01), np.log(500.0))] * 12,
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if res.fun < best_score:
                    candidate = np.sort(np.exp(res.x))
                    if np.all(candidate > 0):
                        best_score = res.fun
                        best_roots = candidate
        except Exception:
            pass
    
    # Final validation
    if np.any(best_roots <= 0) or np.any(np.isnan(best_roots)):
        # Fallback to standard Laguerre roots
        x0, _ = roots_laguerre(12)
        best_roots = np.sort(x0)
    
    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
