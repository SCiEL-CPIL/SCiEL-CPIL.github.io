"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre, eval_laguerre
from scipy.optimize import minimize
import time

def _compute_bound(roots):
    """
    Compute the C4 upper bound for given root locations.
    Uses Gauss-Laguerre quadrature to evaluate the integral
    C4_bound = 1 / sqrt(integral_0^inf e^{-x} / prod_i (x - r_i)^2 dx)
    """
    m = 12
    n = 1
    roots = np.asarray(roots, dtype=float)
    
    x_quad, w_quad = np.polynomial.laguerre.laggauss(300)
    
    log_P = np.zeros_like(x_quad)
    for r in roots:
        log_P += 2.0 * np.log(np.abs(x_quad - r) + 1e-300)
    
    integrand = np.exp(-x_quad) / np.exp(log_P)
    integral = np.sum(w_quad * integrand)
    
    if integral <= 0 or not np.isfinite(integral):
        return 1e10
    
    return 1.0 / np.sqrt(integral)

def _density_to_roots(params, x_grid):
    """
    Convert density parameters to root locations via CDF inversion.
    params: [beta, gamma, alpha1, alpha2, ...]
    Density: rho(x) = exp(-beta * x^2) * x^gamma * prod_i (x - c_i)^alpha_i
    We use a simpler parametric form: rho(x) ~ x^a * exp(-b*x) * (1 + sum c_k * x^k)
    """
    a, b = params[0], params[1]
    # Use a flexible density: x^a * exp(-b*x) normalized
    # Ensure positivity
    a = max(a, -1.5)
    b = max(b, 0.1)
    
    rho = np.power(x_grid, a) * np.exp(-b * x_grid)
    # Add shape flexibility
    for i in range(2, len(params)):
        rho *= (1.0 + params[i] * x_grid ** (i - 1))
    
    rho = np.maximum(rho, 1e-15)
    # Normalize to get a proper density
    total = np.trapz(rho, x_grid)
    if total <= 0 or not np.isfinite(total):
        return None
    cdf = np.cumsum(rho) * (x_grid[1] - x_grid[0]) / total
    cdf[0] = 0
    cdf[-1] = 1
    
    # Invert CDF to get 12 equally spaced quantiles
    quantiles = np.linspace(0, 1, 13)[1:-1]  # 12 points
    roots = np.interp(quantiles, cdf, x_grid)
    
    # Ensure positive and sorted
    roots = np.maximum(roots, 1e-6)
    roots = np.sort(roots)
    
    return roots

def _fokker_planck_collocation(roots_init, x_grid, n_iter=50):
    """
    Solve the Fokker-Planck stationary equation for optimal root density.
    The effective potential is V(x) = -log(rho(x)) and we solve for the
    density that minimizes the C4 bound via collocation.
    """
    roots = roots_init.copy()
    dx = x_grid[1] - x_grid[0]
    
    for iteration in range(n_iter):
        # Compute current bound gradient with respect to density
        # The Fokker-Planck equation: d/dx [D(x) d(rho)/dx + mu(x) rho] = 0
        # where D(x) is diffusivity and mu(x) is drift from the potential
        
        # Compute the current root density
        # Use kernel density estimation
        sigma = np.max(x_grid) / 100.0
        rho_current = np.zeros_like(x_grid)
        for r in roots:
            rho_current += np.exp(-0.5 * ((x_grid - r) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        
        # Compute the force field from the C4 bound gradient
        # The optimal density satisfies: rho(x) ~ exp(-V(x)) where V is the potential
        # V(x) is related to the integral kernel of the C4 bound
        
        # Compute effective potential from the bound structure
        log_P = np.zeros_like(x_grid)
        for r in roots:
            log_P += 2.0 * np.log(np.abs(x_grid - r) + 1e-300)
        
        # The C4 integrand is e^{-x} / P(x), so the "energy" landscape is
        # E(x) = x + log(P(x)) = x + 2*sum log|x - r_i|
        E = x_grid + log_P
        
        # The optimal density should concentrate where E is small
        # Use a softmax-like update
        E_min = np.min(E)
        E_shifted = E - E_min
        target_rho = np.exp(-E_shifted)
        target_rho = np.maximum(target_rho, 1e-15)
        
        # Normalize target density
        target_total = np.trapz(target_rho, x_grid)
        if target_total > 0 and np.isfinite(target_total):
            target_cdf = np.cumsum(target_rho) * dx / target_total
            target_cdf[0] = 0
            target_cdf[-1] = 1
            
            # Get new roots from quantiles
            quantiles = np.linspace(0, 1, 13)[1:-1]
            new_roots = np.interp(quantiles, target_cdf, x_grid)
            new_roots = np.maximum(new_roots, 1e-6)
            new_roots = np.sort(new_roots)
            
            # Blend with previous roots for stability
            alpha = 0.3
            roots = (1 - alpha) * roots + alpha * new_roots
            roots = np.sort(roots)
            roots = np.maximum(roots, 1e-6)
    
    return roots

def construct():
    """
    Return 12 positive double-root locations that minimize the upper bound
    on C4 for the Cohn-Goncalves Laguerre polynomial construction (m=12, n=1).
    """
    start_time = time.time()
    deadline = start_time + 590.0
    
    # Create a fine grid for density representation
    x_grid = np.linspace(1e-4, 100.0, 5000)
    
    # Initial guess: Laguerre roots
    x_init, _ = roots_laguerre(12)
    best_roots = np.sort(x_init).astype(float)
    best_bound = _compute_bound(best_roots)
    
    # Phase 1: Fokker-Planck collocation to find optimal density
    try:
        fp_roots = _fokker_planck_collocation(best_roots, x_grid, n_iter=30)
        fp_bound = _compute_bound(fp_roots)
        if fp_bound < best_bound:
            best_roots = fp_roots
            best_bound = fp_bound
    except Exception:
        pass
    
    # Phase 2: Density parameter optimization
    try:
        if time.time() - start_time < deadline - 120:
            # Optimize density parameters [a, b] for rho(x) = x^a * exp(-b*x)
            def density_objective(params):
                elapsed = time.time() - start_time
                if elapsed > deadline - 30:
                    return 1e10
                roots = _density_to_roots(params, x_grid)
                if roots is None or len(roots) != 12:
                    return 1e10
                if any(r <= 0 for r in roots):
                    return 1e10
                return _compute_bound(roots)
            
            # Initial parameters from current best
            # Estimate a and b from current roots
            log_roots = np.log(best_roots + 1e-10)
            a_init = 0.5
            b_init = 1.0 / np.mean(best_roots)
            
            result = minimize(density_objective, [a_init, b_init], method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if result.fun < best_bound:
                new_roots = _density_to_roots(result.x, x_grid)
                if new_roots is not None and len(new_roots) == 12 and all(r > 0 for r in new_roots):
                    best_roots = new_roots
                    best_bound = result.fun
    except Exception:
        pass
    
    # Phase 3: Fine-tune individual roots with coordinate descent
    try:
        if time.time() - start_time < deadline - 60:
            roots = best_roots.copy()
            for sweep in range(3):
                for i in range(12):
                    # Optimize root i while keeping others fixed
                    def single_root_obj(r):
                        r = float(r)
                        if r <= 0:
                            return 1e10
                        new_roots = roots.copy()
                        new_roots[i] = r
                        new_roots = np.sort(new_roots)
                        return _compute_bound(new_roots)
                    
                    res = minimize(single_root_obj, roots[i], method='Nelder-Mead',
                                  options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12})
                    if res.fun < best_bound:
                        roots[i] = max(float(res.x), 1e-6)
                        roots = np.sort(roots)
                        best_bound = res.fun
                        best_roots = roots.copy()
                
                if time.time() - start_time > deadline - 30:
                    break
    except Exception:
        pass
    
    # Phase 4: Final polish with L-BFGS-B on sorted roots
    try:
        if time.time() - start_time < deadline - 30:
            def objective(params):
                elapsed = time.time() - start_time
                if elapsed > deadline - 10:
                    return 1e10
                roots = np.sort(params)
                if any(r <= 0 for r in roots):
                    return 1e10
                return _compute_bound(roots)
            
            result = minimize(objective, best_roots, method='L-BFGS-B',
                             options={'maxiter': 100, 'maxfun': 2000, 'ftol': 1e-14})
            
            if result.fun < best_bound:
                best_roots = np.sort(result.x)
                best_bound = result.fun
    except Exception:
        pass
    
    # Verify constraints
    best_roots = np.sort(best_roots)
    if any(r <= 0 for r in best_roots) or len(best_roots) != 12:
        # Fallback to Laguerre roots
        x_fallback, _ = roots_laguerre(12)
        best_roots = np.sort(x_fallback)
    
    return best_roots.tolist()
# EVOLVE-BLOCK-END
