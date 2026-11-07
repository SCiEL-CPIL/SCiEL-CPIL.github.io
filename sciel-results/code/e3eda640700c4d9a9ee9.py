"""Structural approach: parameterize by the sign-changing root directly."""
# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import time
import math

def _hermite_poly_coeffs(n):
    """Return coefficients of physicists' Hermite polynomial H_n(x) as list [a_0, a_1, ..., a_n]
    where H_n(x) = sum a_k * x^k."""
    # Recurrence: H_{n+1}(x) = 2x*H_n(x) - 2n*H_{n-1}(x)
    # H_0 = 1, H_1 = 2x
    if n == 0:
        return [1]
    if n == 1:
        return [0, 2]
    h_prev = [1]  # H_0
    h_curr = [0, 2]  # H_1
    for k in range(1, n):
        # H_{k+1} = 2x * H_k - 2k * H_{k-1}
        # 2x * H_k: shift coefficients right by 1, multiply by 2
        new_len = len(h_curr) + 1
        h_next = [0] * new_len
        for i in range(len(h_curr)):
            h_next[i+1] = 2 * h_curr[i]
        # subtract 2k * H_{k-1}
        for i in range(len(h_prev)):
            h_next[i] -= 2 * k * h_prev[i]
        h_prev = h_curr
        h_curr = h_next
    return h_curr

# Precompute Hermite polynomial coefficients (in x, not x^2)
H0_coeffs = _hermite_poly_coeffs(0)   # [1]
H4_coeffs = _hermite_poly_coeffs(4)   # [12, 0, -48, 0, 16]
H8_coeffs = _hermite_poly_coeffs(8)
H12_coeffs = _hermite_poly_coeffs(12)

# For our problem, g is an even polynomial, so only even powers matter.
# Let y = x^2. Then:
# H0(y) = 1
# H4(y) = 12 - 48y + 16y^2
# H8(y) = 168 - 1344y + 2688y^2 - 1344y^3 + 256y^4
# H12(y) = 18480 - 921600y + 1424640y^2 - 1003520y^3 + 332160y^4 - 46080y^5 + 4096y^6

# Extract even-power coefficients (coefficients of y^k in H_n(y))
def _even_coeffs(h_coeffs):
    """From polynomial in x, extract coefficients of even powers (i.e., polynomial in y=x^2)."""
    result = []
    for i in range(0, len(h_coeffs), 2):
        result.append(h_coeffs[i])
    return result

h0_y = _even_coeffs(H0_coeffs)    # [1]
h4_y = _even_coeffs(H4_coeffs)    # [12, -48, 16]
h8_y = _even_coeffs(H8_coeffs)    # [168, -1344, 2688, -1344, 256]
h12_y = _even_coeffs(H12_coeffs)  # [18480, -921600, 1424640, -1003520, 332160, -46080, 4096]

# Verify
assert h0_y == [1]
assert h4_y == [12, -48, 16]
assert h8_y == [168, -1344, 2688, -1344, 256]
assert h12_y == [18480, -921600, 1424640, -1003520, 332160, -46080, 4096]

def compute_q_coeffs(c0, c1, c2):
    """
    Compute coefficients of q(y) = g(x)/x^2 as a polynomial in y = x^2.
    g(y) = c0*h0_y + c1*h4_y + c2*h8_y + c3*h12_y, with g(0) = 0.
    q(y) = g(y)/y, a degree-5 polynomial in y.
    Returns q_coeffs as list [q_0, q_1, ..., q_5] where q(y) = q_0 + q_1*y + ... + q_5*y^5.
    """
    c3 = -(c0 + 12*c1 + 168*c2) / 18480.0
    
    # g(y) coefficients: b_k = c0*h0_y[k] + c1*h4_y[k] + c2*h8_y[k] + c3*h12_y[k]
    max_deg = 6  # h12_y has degree 6 in y
    b = [0.0] * (max_deg + 1)
    for k in range(len(h0_y)):
        b[k] += c0 * h0_y[k]
    for k in range(len(h4_y)):
        b[k] += c1 * h4_y[k]
    for k in range(len(h8_y)):
        b[k] += c2 * h8_y[k]
    for k in range(len(h12_y)):
        b[k] += c3 * h12_y[k]
    
    # q(y) = g(y)/y = b_1 + b_2*y + ... + b_6*y^5
    q_coeffs = b[1:7]
    return q_coeffs, c3

def evaluate_value(c0, c1, c2):
    """
    Compute value = r^2/(2*pi) where r is the largest real root of q(x) at which q changes sign.
    q(x) = g(x)/x^2, which is an even function, so we work with q(y) where y = x^2.
    """
    q_coeffs, c3 = compute_q_coeffs(c0, c1, c2)
    
    # Leading coefficient of g is c3 * 4096 (from H12's leading term 4096*x^12)
    # If c3 < 0, g -> -infinity, so we replace g by -g, which means q -> -q
    lead_coeff = c3 * 4096.0
    if lead_coeff < 0:
        q_coeffs = [-c for c in q_coeffs]
    
    # Find roots of q(y) = 0, a degree-5 polynomial in y
    # Use numpy roots
    try:
        roots_y = np.roots(q_coeffs)
    except Exception:
        return float('inf')
    
    # Filter for real, positive roots (y > 0 means x = sqrt(y) is real and nonzero)
    real_positive_roots = []
    for r in roots_y:
        if abs(r.imag) < 1e-10 and r.real > 1e-12:
            real_positive_roots.append(r.real)
    
    if not real_positive_roots:
        return float('inf')
    
    # For each root, check if q changes sign there (i.e., odd multiplicity)
    # Evaluate q at y_root - eps and y_root + eps
    def q_eval(y):
        """Evaluate q(y) using Horner's method."""
        result = 0.0
        for coeff in q_coeffs:
            result = result * y + coeff
        return result
    
    best_r = 0.0
    for y_root in real_positive_roots:
        # Use a small epsilon relative to the root
        eps = max(1e-10, y_root * 1e-10)
        val_minus = q_eval(y_root - eps) if y_root > eps else q_eval(0)
        val_plus = q_eval(y_root + eps)
        
        # Sign change means odd multiplicity
        if val_minus * val_plus < 0:
            r = math.sqrt(y_root)
            if r > best_r:
                best_r = r
    
    if best_r == 0.0:
        return float('inf')
    
    value = best_r**2 / (2.0 * math.pi)
    return value

def construct():
    """
    Find coefficients [c0, c1, c2] that minimize value = r^2/(2*pi).
    
    Key insight: We want to minimize r, the largest sign-changing root of q.
    This is equivalent to making q(y) negative for all y > r_target^2, where r_target is small.
    
    Strategy: 
    1. The seed gives a baseline.
    2. We can try to make q(y) have its largest sign-changing root as small as possible.
    3. One approach: make q(y) = a*(y - y1)*(y - y2)*... with all roots small, or make
       q(y) negative for all y > some small value.
    
    Actually, the key structural insight: if we can make q(y) < 0 for all y > 0 (after the
    sign flip), then there are no positive sign-changing roots and the value would be... 
    wait, we need at least one sign-changing root for the problem to be well-defined.
    
    Let me reconsider. The value is r^2/(2*pi) where r is the LARGEST sign-changing root.
    To minimize this, we want the largest sign-changing root to be as small as possible.
    
    Approach: Use a parameterization where we directly control the roots.
    If q(y) = a * (y - y1) * (y - y2) * ... * (y - y5), then we can set all yi to be small.
    But we need to satisfy the constraint that q comes from the specific Hermite combination.
    
    Alternative: Direct optimization using scipy's minimize with a good initial guess.
    """
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin
    
    # Seed
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_coeffs = seed.copy()
    best_value = evaluate_value(seed[0], seed[1], seed[2])
    
    # Try to use scipy for optimization if available
    try:
        from scipy.optimize import minimize, differential_evolution
        have_scipy = True
    except ImportError:
        have_scipy = False
    
    if have_scipy:
        # Define objective
        def objective(coeffs):
            val = evaluate_value(coeffs[0], coeffs[1], coeffs[2])
            if not math.isfinite(val):
                return 1e10
            return val
        
        # Try differential evolution for global search
        bounds = [(-10, 10), (-10, 10), (-1, 1)]
        
        # First, try a quick DE with limited budget
        time_remaining = deadline - time.time()
        if time_remaining > 60:
            try:
                result = differential_evolution(
                    objective, bounds, 
                    maxiter=200, popsize=15,
                    seed=42, tol=1e-6,
                    polish=True
                )
                if result.fun < best_value:
                    best_value = result.fun
                    best_coeffs = result.x.copy()
            except Exception:
                pass
        
        # Local refinement
        time_remaining = deadline - time.time()
        if time_remaining > 10:
            try:
                result = minimize(
                    objective, best_coeffs,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
                )
                if result.fun < best_value:
                    best_value = result.fun
                    best_coeffs = result.x.copy()
            except Exception:
                pass
        
        # Try multiple local starts
        time_remaining = deadline - time.time()
        if time_remaining > 30:
            rng = np.random.RandomState(123)
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                # Random perturbation of best
                pert = best_coeffs + rng.normal(0, [0.5, 0.5, 0.05])
                val = objective(pert)
                if val < best_value:
                    best_value = val
                    best_coeffs = pert.copy()
                    # Local refine from new best
                    try:
                        result = minimize(
                            objective, best_coeffs,
                            method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10}
                        )
                        if result.fun < best_value:
                            best_value = result.fun
                            best_coeffs = result.x.copy()
                    except Exception:
                        pass
    
    # Fallback: manual Nelder-Mead if scipy not available
    if not have_scipy:
        def objective(coeffs):
            val = evaluate_value(coeffs[0], coeffs[1], coeffs[2])
            if not math.isfinite(val):
                return 1e10
            return val
        
        # Simple Nelder-Mead
        simplex = [best_coeffs.copy()]
        for i in range(3):
            pert = best_coeffs.copy()
            step = max(abs(best_coeffs[i]), 0.1) * 0.1
            pert[i] += step
            simplex.append(pert)
        
        for _ in range(1000):
            if time.time() > deadline - 5:
                break
            
            fvals = [objective(v) for v in simplex]
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            if abs(fvals[0] - fvals[-1]) < 1e-15:
                break
            
            centroid = np.mean(simplex[:-1], axis=0)
            xr = centroid + (centroid - simplex[-1])
            fr = objective(xr)
            
            if fr < fvals[0]:
                xe = centroid + 2*(centroid - simplex[-1])
                fe = objective(xe)
                simplex[-1] = xe if fe < fr else xr
            elif fr < fvals[-2]:
                simplex[-1] = xr
            else:
                xc = centroid + 0.5*(centroid - simplex[-1])
                fc = objective(xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc
                else:
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + 0.5*(simplex[i] - simplex[0])
        
        for v in simplex:
            val = objective(v)
            if val < best_value:
                best_value = val
                best_coeffs = v.copy()
    
    # Final check
    final_val = evaluate_value(best_coeffs[0], best_coeffs[1], best_coeffs[2])
    if math.isfinite(final_val) and final_val < best_value:
        best_value = final_val
    
    return list(best_coeffs)
# EVOLVE-BLOCK-END
