"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Hermite polynomials evaluated at specific points (physicists' convention)
# H_n(0) for even n: H_n(0) = (-1)^(n/2) * n! / (n/2)!
# H_0(0) = 1
# H_4(0) = -12
# H_8(0) = 1680
# H_12(0) = -19958400

H0_0 = 1.0
H4_0 = -12.0
H8_0 = 1680.0
H12_0 = -19958400.0


def _compute_value(c0, c1, c2):
    """Compute the value r^2/(2*pi) for given coefficients."""
    # c3 is fixed so that g(0) = 0
    # g(0) = c0*H0(0) + c1*H4(0) + c2*H8(0) + c3*H12(0) = 0
    # c3 = -(c0*H0_0 + c1*H4_0 + c2*H8_0) / H12_0
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # g(x) = c0*H0(x) + c1*H4(x) + c2*H8(x) + c3*H12(x)
    # We need q(x) = g(x) / x^2, which must divide exactly.
    # H_n(x) is a polynomial of degree n. For g(x)/x^2 to be a polynomial,
    # g(x) must be divisible by x^2, i.e., g(0) = 0 and g'(0) = 0.
    # We already enforced g(0) = 0 via c3. Now check g'(0) = 0.
    
    # H_n'(0) for even n is 0 (since H_n is even for even n, its derivative is odd, so H_n'(0)=0)
    # So g'(0) = 0 automatically for even n. Good.
    
    # Now q(x) = g(x)/x^2 is a polynomial of degree 10 (since H12 has degree 12).
    # We need to find the largest real root r of q where q changes sign.
    
    # Let's construct q(x) explicitly.
    # H_0(x) = 1
    # H_4(x) = 16x^4 - 48x^2 + 12
    # H_8(x) = 256x^8 - 3072x^6 + 13440x^4 - 13440x^2 + 1680
    # H_12(x) = 4096x^12 - 196608x^10 + 3774720x^8 - 31129600x^6 + 109612800x^4 - 159753600x^2 + 19958400
    
    # g(x) = c0*1 + c1*(16x^4 - 48x^2 + 12) + c2*(256x^8 - 3072x^6 + 13440x^4 - 13440x^2 + 1680) + c3*(4096x^12 - 196608x^10 + 3774720x^8 - 31129600x^6 + 109612800x^4 - 159753600x^2 + 19958400)
    
    # q(x) = g(x)/x^2
    # q(x) = c0/x^2 + c1*(16x^2 - 48 + 12/x^2) + c2*(256x^6 - 3072x^4 + 13440x^2 - 13440 + 1680/x^2) + c3*(4096x^10 - 196608x^8 + 3774720x^6 - 31129600x^4 + 109612800x^2 - 159753600 + 19958400/x^2)
    
    # Wait, this doesn't look like it divides exactly unless the 1/x^2 terms cancel.
    # Let me re-examine. g(0) = 0 means the constant term is 0.
    # For g(x)/x^2 to be a polynomial, g(x) must have no constant or linear term.
    # Since all H_n for even n are even functions, g(x) is even, so no linear term.
    # g(0) = 0 means constant term is 0. So g(x) = x^2 * (polynomial of degree 10).
    
    # Let me collect terms by power of x:
    # x^0: c0 + c1*12 + c2*1680 + c3*19958400 = 0 (by construction)
    # x^2: c1*(-48) + c2*(-13440) + c3*(-159753600)
    # x^4: c1*16 + c2*13440 + c3*109612800
    # x^6: c2*(-3072) + c3*(-31129600)
    # x^8: c2*256 + c3*3774720
    # x^10: c3*(-196608)
    # x^12: c3*4096
    
    # So q(x) = a0 + a1*x^2 + a2*x^4 + a3*x^6 + a4*x^8 + a5*x^10 where:
    a0 = c1 * (-48) + c2 * (-13440) + c3 * (-159753600)
    a1 = c1 * 16 + c2 * 13440 + c3 * 109612800
    a2 = c2 * (-3072) + c3 * (-31129600)
    a3 = c2 * 256 + c3 * 3774720
    a4 = c3 * (-196608)
    a5 = c3 * 4096
    
    # q(x) = a0 + a1*x^2 + a2*x^4 + a3*x^6 + a4*x^8 + a5*x^10
    # Let y = x^2, then q = P(y) = a0 + a1*y + a2*y^2 + a3*y^3 + a4*y^4 + a5*y^5
    # We need the largest real root y > 0 of P(y) where P changes sign.
    # Then r = sqrt(y), value = r^2/(2*pi) = y/(2*pi)
    
    # Find roots of P(y)
    coeffs = [a5, a4, a3, a2, a1, a0]
    roots = np.roots(coeffs)
    
    # Filter for real positive roots
    real_roots = []
    for root in roots:
        if abs(root.imag) < 1e-10 and root.real > 1e-10:
            real_roots.append(root.real)
    
    if not real_roots:
        return float('inf')
    
    # Find the largest root where P changes sign
    # P changes sign at a root if the multiplicity is odd
    # For simplicity, let's check the sign change numerically
    
    def P(y):
        return a5*y**5 + a4*y**4 + a3*y**3 + a2*y**2 + a1*y + a0
    
    best_y = 0
    for y in real_roots:
        # Check sign change
        eps = 1e-8
        sign_before = np.sign(P(y - eps))
        sign_after = np.sign(P(y + eps))
        if sign_before != sign_after:
            if y > best_y:
                best_y = y
    
    if best_y == 0:
        return float('inf')
    
    return best_y / (2 * np.pi)


def construct():
    """Find coefficients c0, c1, c2 to minimize the value."""
    start_time = time.time()
    deadline = start_time + 180  # Safety margin before 210s limit
    
    # Seed
    seed = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_c = seed.copy()
    best_val = _compute_value(*seed)
    
    # Local optimization
    def objective(c):
        return _compute_value(c[0], c[1], c[2])
    
    # Try multiple starting points
    rng = np.random.default_rng(42)
    n_starts = 50
    
    for i in range(n_starts):
        if time.time() > deadline - 10:
            break
        
        if i == 0:
            x0 = seed.copy()
        else:
            # Random perturbation
            scale = 0.5 * (1.0 / (i + 1))
            x0 = seed + scale * rng.standard_normal(3)
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if result.fun < best_val:
                best_val = result.fun
                best_c = result.x.copy()
        except Exception:
            pass
    
    # Final polish with higher precision
    if time.time() < deadline - 5:
        try:
            result = minimize(objective, best_c, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if result.fun < best_val:
                best_val = result.fun
                best_c = result.x.copy()
        except Exception:
            pass
    
    return np.array(best_c)
# EVOLVE-BLOCK-END
