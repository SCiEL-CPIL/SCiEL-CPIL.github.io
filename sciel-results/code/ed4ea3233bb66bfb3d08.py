import time
import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar

# Precompute Hermite polynomial coefficients for H0, H4, H8, H12
# Physicists' Hermite polynomials: H_n(x)
# H0 = 1
# H4 = x^4 - 6x^2 + 3
# H8 = x^8 - 28x^6 + 280x^4 - 840x^2 + 105
# H12 = x^12 - 60x^10 + 1260x^8 - 12600x^6 + 45360x^4 - 30240x^2 + 10395

# Coefficients as lists [c_0, c_2, c_4, ...] for terms x^0, x^2, x^4, ...
H0_coeffs = np.array([1.0])
H4_coeffs = np.array([3.0, -6.0, 1.0])
H8_coeffs = np.array([105.0, -840.0, 280.0, -28.0, 1.0])
H12_coeffs = np.array([10395.0, -30240.0, 45360.0, -12600.0, 1260.0, -60.0, 1.0])

# Values at 0
H0_0 = 1.0
H4_0 = 3.0
H8_0 = 105.0
H12_0 = 10395.0

def _build_q_coeffs(c0, c1, c2):
    """Build the coefficient vector of q = g/x^2."""
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # g coefficients (in terms of x^0, x^2, ..., x^12)
    g_coeffs = np.zeros(7)
    for i, c in enumerate(H0_coeffs):
        g_coeffs[i] += c0 * c
    for i, c in enumerate(H4_coeffs):
        g_coeffs[i] += c1 * c
    for i, c in enumerate(H8_coeffs):
        g_coeffs[i] += c2 * c
    for i, c in enumerate(H12_coeffs):
        g_coeffs[i] += c3 * c
    
    # q = g / x^2
    # g_coeffs[0] is coeff of x^0, which is g(0). Should be ~0.
    # q_coeffs[i] is coeff of x^(2i) in q, which corresponds to g_coeffs[i+1]
    q_coeffs = g_coeffs[1:]
    return q_coeffs

def _largest_sign_changing_root(q_coeffs):
    """Find the largest real root of q where q changes sign."""
    n = len(q_coeffs)
    # q(x) = sum q_coeffs[i] * x^(2i)
    # Convert to standard polynomial coeffs for x
    # Degree is 2*(n-1)
    deg = 2 * (n - 1)
    full_coeffs = np.zeros(deg + 1)
    for i in range(n):
        full_coeffs[deg - 2 * i] = q_coeffs[i]
    
    # Find roots
    roots = np.roots(full_coeffs)
    
    # Filter real roots
    real_roots = roots[np.abs(roots.imag) < 1e-9].real
    
    if len(real_roots) == 0:
        return 0.0
    
    best_r = 0.0
    # Sort roots to find largest
    real_roots = np.sort(real_roots)
    
    for r in reversed(real_roots):
        if r <= 1e-12:
            continue
        # Check sign change
        eps = 1e-8 * max(1.0, abs(r))
        q_left = np.polyval(full_coeffs, r - eps)
        q_right = np.polyval(full_coeffs, r + eps)
        if q_left * q_right < 0:
            return r
    
    return best_r

def _compute_value(c0, c1, c2):
    """Compute the value = r^2 / (2*pi)."""
    q_coeffs = _build_q_coeffs(c0, c1, c2)
    r = _largest_sign_changing_root(q_coeffs)
    return r * r / (2 * np.pi)

def _check_positivity(c0, c1, c2, y_grid):
    """Check if g(x) > 0 for all x > 0 by checking P(y) > 0 for y in grid."""
    q_coeffs = _build_q_coeffs(c0, c1, c2)
    # q(y) = g(sqrt(y))/y = P(y)/y? No.
    # g(x) = x^2 q(x). Let x = sqrt(y).
    # g(sqrt(y)) = y q(sqrt(y)).
    # We want g(x) > 0 for x > 0.
    # So we want y * q(sqrt(y)) > 0 for y > 0.
    # Since y > 0, we just need q(sqrt(y)) > 0.
    # q(x) = sum q_coeffs[i] x^(2i) = sum q_coeffs[i] y^i.
    # So we want P_q(y) = sum q_coeffs[i] y^i > 0 for all y > 0.
    
    min_val = np.inf
    for y in y_grid:
        val = np.polyval(q_coeffs[::-1], y) # q_coeffs is [c0, c1, ...] for y^0, y^1...
        # Wait, np.polyval expects [highest, ..., lowest].
        # q_coeffs[i] is coeff of y^i.
        # So coeffs for np.polyval are q_coeffs[::-1]
        if val < min_val:
            min_val = val
            
    return min_val

def construct():
    start_time = time.time()
    deadline = 200.0 # Safety margin
    
    # Seed
    seed_coeffs = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    seed_value = _compute_value(*seed_coeffs)
    
    best_coeffs = seed_coeffs.copy()
    best_value = seed_value
    
    # Strategy 1: Try to find a solution where g(x) > 0 for all x > 0.
    # This would imply no sign-changing positive roots, so r=0, value=0.
    # We use DE to minimize the minimum of P_q(y) on a grid.
    
    y_grid = np.logspace(-6, 4, 100) # y from 1e-6 to 1e4
    
    def positivity_objective(x):
        # x = [c0, c1, c2]
        min_val = _check_positivity(x[0], x[1], x[2], y_grid)
        # We want min_val to be as large as possible (positive).
        # So we minimize -min_val.
        # If min_val > 0, objective is negative.
        # If min_val < 0, objective is positive.
        return -min_val

    # Run DE for positivity
    try:
        bounds = [(-5, 5), (-5, 5), (-5, 5)]
        # Use a seed close to current best or random
        res_pos = differential_evolution(
            positivity_objective, bounds,
            maxiter=50, popsize=20,
            tol=1e-6, seed=1,
            polish=True
        )
        
        if res_pos.fun < 0: # min_val > 0
            # We found a candidate where g(x) > 0 for all x in grid.
            # Verify with a finer grid or root finding.
            c0, c1, c2 = res_pos.x
            val = _compute_value(c0, c1, c2)
            if val < best_value:
                best_value = val
                best_coeffs = res_pos.x.copy()
                
            # If val is 0, we are done.
            if best_value < 1e-10:
                return best_coeffs
    except Exception:
        pass

    # Strategy 2: Minimize the value directly if positivity failed or didn't beat seed.
    # Use DE to find good candidates, then local polish.
    
    def value_objective(x):
        return _compute_value(x[0], x[1], x[2])

    try:
        bounds = [(-10, 10), (-10, 10), (-10, 10)]
        # Start DE
        res_val = differential_evolution(
            value_objective, bounds,
            maxiter=100, popsize=30,
            tol=1e-8, seed=2,
            polish=True
        )
        
        if res_val.fun < best_value:
            best_value = res_val.fun
            best_coeffs = res_val.x.copy()
    except Exception:
        pass

    # Strategy 3: Local refinement around best found and seed
    # Use Nelder-Mead for robustness against non-smoothness
    
    candidates = [
        best_coeffs.copy(),
        seed_coeffs.copy()
    ]
    
    # Add some random perturbations
    rng = np.random.default_rng(3)
    for _ in range(10):
        pert = seed_coeffs + rng.normal(0, 1.0, 3)
        candidates.append(pert)
    
    for c in candidates:
        if time.time() - start_time > deadline - 10:
            break
            
        try:
            res_local = minimize(
                value_objective, c,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-15}
            )
            if res_local.fun < best_value:
                best_value = res_local.fun
                best_coeffs = res_local.x.copy()
                
                # If we get very close to 0, try to polish further
                if best_value < 1e-6:
                     # Try to drive it to 0 using positivity check?
                     # Or just return.
                     pass
        except Exception:
            pass

    return best_coeffs
