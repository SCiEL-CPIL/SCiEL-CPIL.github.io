"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import roots_hermite

# Precompute Hermite polynomial coefficients in y = x^2
# H0(x) = 1 => in y: 1
# H4(x) = x^4 - 6x^2 + 3 => in y: y^2 - 6y + 3
# H8(x) = x^8 - 28x^6 + 280x^4 - 840x^2 + 105 => in y: y^4 - 28y^3 + 280y^2 - 840y + 105
# H12(x) = x^12 - 60x^10 + 1260x^8 - 12600x^6 + 45360x^4 - 30240x^2 + 10395
#         => in y: y^6 - 60y^5 + 1260y^4 - 12600y^3 + 45360y^2 - 30240y + 10395

H0_0 = 1.0
H4_0 = 3.0
H8_0 = 105.0
H12_0 = 10395.0


def _compute_value(c0, c1, c2):
    """Compute value = r^2 / (2*pi) for given coefficients."""
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # P(y) = g(sqrt(y))/y, degree 5 polynomial in y
    # Coefficients from highest to lowest power of y
    a5 = c3
    a4 = -60.0 * c3
    a3 = 1260.0 * c3 + c2
    a2 = -12600.0 * c3 - 28.0 * c2
    a1 = 45360.0 * c3 + 280.0 * c2 - 6.0 * c1
    a0 = -30240.0 * c3 - 840.0 * c2 - 6.0 * c1
    
    coeffs = np.array([a5, a4, a3, a2, a1, a0])
    
    # Find roots of P(y)
    roots = np.roots(coeffs)
    
    # Filter real positive roots
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    real_roots = real_roots[real_roots > 0]
    
    if len(real_roots) == 0:
        return 0.0  # No positive real roots, r = 0
    
    # For each real root, check if P changes sign there (odd multiplicity)
    # Evaluate P on both sides
    best_r2 = 0.0
    for r2 in real_roots:
        eps = 1e-10 * max(1.0, abs(r2))
        p_left = np.polyval(coeffs, r2 - eps)
        p_right = np.polyval(coeffs, r2 + eps)
        if p_left * p_right < 0:  # Sign change
            if r2 > best_r2:
                best_r2 = r2
    
    return best_r2 / (2.0 * np.pi)


def _smooth_surrogate(c0, c1, c2, temperature=0.01):
    """Smooth surrogate for the max root objective using log-sum-exp."""
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    a5 = c3
    a4 = -60.0 * c3
    a3 = 1260.0 * c3 + c2
    a2 = -12600.0 * c3 - 28.0 * c2
    a1 = 45360.0 * c3 + 280.0 * c2 - 6.0 * c1
    a0 = -30240.0 * c3 - 840.0 * c2 - 6.0 * c1
    
    coeffs = np.array([a5, a4, a3, a2, a1, a0])
    
    # Sample points in y > 0
    y_samples = np.linspace(0.01, 100.0, 200)
    p_vals = np.polyval(coeffs, y_samples)
    
    # We want to penalize negative values; use soft-max of -p_vals
    # The "effective r^2" is roughly where P crosses zero
    # Use a smooth approximation: find where P becomes positive
    # Simpler: minimize the maximum of -P(y) for y in some range
    # This encourages P(y) > 0 for all y in range
    
    # Actually, let's use a different approach: 
    # The value is determined by the largest sign-changing root.
    # A smooth proxy: integrate -min(P(y), 0) over y, weighted by y
    # But this is complex. Let's just use direct root-finding in the optimizer.
    
    return _compute_value(c0, c1, c2)


def construct():
    start_time = time.time()
    deadline = 200.0  # Safety margin before 210s limit
    
    # Seed coefficients
    seed_coeffs = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    seed_value = _compute_value(*seed_coeffs)
    
    best_coeffs = seed_coeffs.copy()
    best_value = seed_value
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Differential Evolution for global search
    elapsed = time.time() - start_time
    if elapsed < 80.0:
        try:
            bounds = [(-5, 5), (-5, 5), (-5, 5)]
            
            def objective(x):
                return _compute_value(x[0], x[1], x[2])
            
            result = differential_evolution(
                objective, bounds,
                maxiter=200, popsize=20,
                tol=1e-10, seed=42,
                polish=False,  # We'll do our own polishing
                mutation=(0.5, 1.0),
                recombination=0.7
            )
            if result.fun < best_value:
                best_value = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Local refinement with multiple starts
    elapsed = time.time() - start_time
    if elapsed < 160.0:
        # Multiple local starts around best and random points
        starts = [best_coeffs.copy()]
        
        # Add starts near seed
        for _ in range(10):
            s = seed_coeffs + rng.normal(0, 0.3, 3)
            starts.append(s)
        
        # Add starts around current best
        for _ in range(10):
            s = best_coeffs + rng.normal(0, 0.1, 3)
            starts.append(s)
        
        # Add some wider random starts
        for _ in range(5):
            s = rng.uniform(-3, 3, 3)
            starts.append(s)
        
        for s in starts:
            if time.time() - start_time > 170.0:
                break
            try:
                res = minimize(
                    _compute_value, s,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
                )
                if res.fun < best_value:
                    best_value = res.fun
                    best_coeffs = res.x.copy()
            except Exception:
                pass
    
    # Phase 3: High-precision polish on the best solution
    elapsed = time.time() - start_time
    if elapsed < 195.0 and best_value < seed_value:
        try:
            res = minimize(
                _compute_value, best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-16, 'fatol': 1e-18, 'adaptive': True}
            )
            if res.fun < best_value:
                best_value = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Final safety check
    if time.time() - start_time > 198.0:
        return best_coeffs
    
    return best_coeffs
# EVOLVE-BLOCK-END
