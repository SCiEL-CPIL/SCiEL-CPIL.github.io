import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import roots_hermite

# Precompute Hermite polynomial coefficients for H0, H4, H8, H12
# Physicists' Hermite polynomials: H_n(x)
# H0 = 1
# H4 = x^4 - 6x^2 + 3
# H8 = x^8 - 28x^6 + 280x^4 - 840x^2 + 105
# H12 = x^12 - 60x^10 + 1260x^8 - 12600x^6 + 45360x^4 - 30240x^2 + 10395

# Coefficients as lists [c_n, c_{n-1}, ..., c_0] for degree n polynomial
# For even polynomials, we can represent as coefficients of x^0, x^2, x^4, ...

# H0: [1]
# H4: [3, -6, 1] (coeffs of x^0, x^2, x^4)
# H8: [105, -840, 280, -28, 1]
# H12: [10395, -30240, 45360, -12600, 1260, -60, 1]

H0_coeffs = np.array([1.0])
H4_coeffs = np.array([3.0, -6.0, 1.0])
H8_coeffs = np.array([105.0, -840.0, 280.0, -28.0, 1.0])
H12_coeffs = np.array([10395.0, -30240.0, 45360.0, -12600.0, 1260.0, -60.0, 1.0])


def _build_q_coeffs(c0, c1, c2):
    """Build the coefficient vector of q = g/x^2 where g = c0*H0 + c1*H4 + c2*H8 + c3*H12."""
    # c3 = -(c0*H0(0) + c1*H4(0) + c2*H8(0)) / H12(0)
    H0_0 = 1.0
    H4_0 = 3.0
    H8_0 = 105.0
    H12_0 = 10395.0
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # g coefficients (in terms of x^0, x^2, x^4, ..., x^12)
    # Max degree is 12, so 7 terms
    g_coeffs = np.zeros(7)
    for i, c in enumerate(H0_coeffs):
        g_coeffs[i] += c0 * c
    for i, c in enumerate(H4_coeffs):
        g_coeffs[i] += c1 * c
    for i, c in enumerate(H8_coeffs):
        g_coeffs[i] += c2 * c
    for i, c in enumerate(H12_coeffs):
        g_coeffs[i] += c3 * c
    
    # q = g / x^2, so shift down by 2 (divide by x^2)
    # q has degree 10, so 6 terms
    q_coeffs = g_coeffs[1:]  # remove x^0 term, shift
    # q_coeffs[i] is coefficient of x^(2i)
    return q_coeffs


def _largest_sign_changing_root(q_coeffs):
    """Find the largest real root of q where q changes sign."""
    n = len(q_coeffs)
    # Build full polynomial coefficients for x^0 through x^(2*(n-1))
    full_coeffs = np.zeros(2 * (n - 1) + 1)
    for i in range(n):
        full_coeffs[2 * (n - 1) - 2 * i] = q_coeffs[i]
    
    # Find roots
    roots = np.roots(full_coeffs)
    
    # Filter real roots
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    
    if len(real_roots) == 0:
        return 0.0
    
    # For each real root, check if q changes sign there
    best_r = 0.0
    for r in real_roots:
        if r <= 0:
            continue
        # Check sign change: evaluate q at r - eps and r + eps
        eps = 1e-10 * max(1.0, abs(r))
        q_left = np.polyval(full_coeffs, r - eps)
        q_right = np.polyval(full_coeffs, r + eps)
        if q_left * q_right < 0:
            if r > best_r:
                best_r = r
    
    return best_r


def _compute_value(c0, c1, c2):
    """Compute the value = r^2 / (2*pi) for given coefficients."""
    q_coeffs = _build_q_coeffs(c0, c1, c2)
    r = _largest_sign_changing_root(q_coeffs)
    return r * r / (2 * np.pi)


def _compute_value_fast(coeffs):
    """Faster computation for optimization."""
    c0, c1, c2 = coeffs
    return _compute_value(c0, c1, c2)


def construct():
    start_time = time.time()
    deadline = 195.0  # Safety margin before 210s
    
    # Seed value
    seed_coeffs = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    seed_value = _compute_value(*seed_coeffs)
    
    best_coeffs = seed_coeffs.copy()
    best_value = seed_value
    
    rng = np.random.default_rng(42)
    
    # Strategy 1: Try to find coefficients that minimize the largest root
    # Use a penalty-based approach with multiple local optimizations
    
    # First, try a grid of initial points with local optimization
    # Focus on regions where the polynomial might have small roots
    
    # Try various initial conditions
    init_points = [seed_coeffs.copy()]
    
    # Add points near the seed
    for scale in [0.1, 0.5, 1.0, 2.0]:
        for _ in range(10):
            init_points.append(seed_coeffs + rng.normal(0, scale, 3))
    
    # Add some structured points
    for c0 in [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]:
        for c1 in [-0.5, -0.25, 0, 0.25, 0.5]:
            for c2 in [-0.1, 0, 0.1]:
                init_points.append(np.array([c0, c1, c2]))
    
    # Remove duplicates and limit
    init_points = np.array(init_points)
    
    # Local optimization from multiple starts
    for i, s in enumerate(init_points):
        if time.time() - start_time > deadline - 30:
            break
        try:
            res = minimize(
                _compute_value_fast, s,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-14}
            )
            if res.fun < best_value:
                best_value = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Strategy 2: Use differential evolution with tighter bounds
    elapsed = time.time() - start_time
    if elapsed < deadline - 40:
        try:
            bounds = [(-5, 5), (-2, 2), (-1, 1)]
            
            def objective(x):
                return _compute_value_fast(x)
            
            result = differential_evolution(
                objective, bounds,
                maxiter=50, popsize=10,
                tol=1e-10, seed=42,
                polish=True
            )
            if result.fun < best_value:
                best_value = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Strategy 3: Fine-tune the best solution found
    if time.time() - start_time < deadline - 10:
        try:
            # Use COBYLA or Powell for final refinement
            for method in ['Powell', 'COBYLA']:
                res = minimize(
                    _compute_value_fast, best_coeffs,
                    method=method,
                    options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-15}
                )
                if res.fun < best_value:
                    best_value = res.fun
                    best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Verify the result
    final_value = _compute_value(*best_coeffs)
    if final_value < best_value:
        best_value = final_value
    
    return best_coeffs
