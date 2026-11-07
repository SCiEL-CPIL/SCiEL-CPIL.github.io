"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Precompute Hermite polynomial coefficients (physicists' convention)
# H0(x) = 1
# H4(x) = x^4 - 6x^2 + 3
# H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H12(x) = x^12 - 84x^10 + 2520x^8 - 30240x^6 + 151200x^4 - 221760x^2 + 10395

def _get_hermite_coeffs():
    """Return coefficients [a0, a1, ..., an] for Hn(x) = sum ai*x^i"""
    # H0
    h0 = [1]
    # H4 = x^4 - 6x^2 + 3
    h4 = [3, 0, -6, 0, 1]
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    h8 = [105, 0, -420, 0, 210, 0, -28, 0, 1]
    # H12 = x^12 - 84x^10 + 2520x^8 - 30240x^6 + 151200x^4 - 221760x^2 + 10395
    h12 = [10395, 0, -221760, 0, 151200, 0, -30240, 0, 2520, 0, -84, 0, 1]
    return h0, h4, h8, h12

_H0, _H4, _H8, _H12 = _get_hermite_coeffs()

def _compute_value(c0, c1, c2):
    """
    Compute the objective value for given coefficients.
    Returns r^2 / (2*pi) where r is the largest real root of q = g/x^2 at which q changes sign.
    """
    # Build g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    # g(0) = 0 => c3 = -(c0*H0(0) + c1*H4(0) + c2*H8(0)) / H12(0)
    H0_0 = _H0[0]  # 1
    H4_0 = _H4[0]  # 3
    H8_0 = _H8[0]  # 105
    H12_0 = _H12[0]  # 10395
    
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # Build g(x) coefficients (constant to highest degree)
    # g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    max_deg = 12
    g_coeffs = np.zeros(max_deg + 1)
    
    for i, c in enumerate(_H0):
        g_coeffs[i] += c0 * c
    for i, c in enumerate(_H4):
        g_coeffs[i] += c1 * c
    for i, c in enumerate(_H8):
        g_coeffs[i] += c2 * c
    for i, c in enumerate(_H12):
        g_coeffs[i] += c3 * c
    
    # q(x) = g(x) / x^2, must divide exactly
    # q(x) has coefficients from g_coeffs[2:] (since g(x) = x^2 * q(x))
    q_coeffs = g_coeffs[2:]  # q(x) = q_coeffs[0] + q_coeffs[1]*x + ... + q_coeffs[-1]*x^(max_deg-2)
    
    # Find all real roots of q(x)
    try:
        roots = np.roots(q_coeffs)
    except:
        return float('inf')
    
    # Filter real roots
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    
    if len(real_roots) == 0:
        return float('inf')
    
    # For each real root, check if q changes sign there
    # A root changes sign if it has odd multiplicity
    # We can check by evaluating q at root - eps and root + eps
    sign_changing_roots = []
    eps = 1e-8
    
    for r in real_roots:
        # Evaluate q at r - eps and r + eps
        q_left = np.polyval(q_coeffs, r - eps)
        q_right = np.polyval(q_coeffs, r + eps)
        
        # Check for sign change
        if q_left * q_right < 0:
            sign_changing_roots.append(r)
    
    if len(sign_changing_roots) == 0:
        return float('inf')
    
    # r is the largest real root at which q changes sign
    r = max(sign_changing_roots)
    
    # Check if g tends to negative limit at infinity (if so, g is replaced by -g)
    # g(x) ~ c3 * x^12 as x -> infinity (since H12 is the highest degree)
    # If c3 < 0, g -> -infinity, so g is replaced by -g
    # This doesn't change the roots of q, but let's verify the sign behavior
    # Actually, the problem says "If g tends to a negative limit at infinity, g is replaced by -g"
    # This affects the sign of q, but the roots remain the same
    # The largest sign-changing root r remains the same
    
    value = r**2 / (2 * np.pi)
    return value

def _compute_value_fast(c0, c1, c2):
    """Faster version using precomputed structures"""
    H0_0, H4_0, H8_0, H12_0 = 1, 3, 105, 10395
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # Build q_coeffs directly
    # g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    # q(x) = g(x)/x^2
    # H0/x^2 doesn't work... wait, H0 = 1, so c0*H0/x^2 = c0/x^2 which is not a polynomial
    # Let me re-read: "Let q = g / x^2, which must divide exactly"
    # This means g must be divisible by x^2, i.e., g(0) = 0 and g'(0) = 0
    # We only enforced g(0) = 0. Let me check if g'(0) = 0 automatically.
    # g'(x) = c0*H0' + c1*H4' + c2*H8' + c3*H12'
    # H0'(0) = 0, H4'(0) = 0, H8'(0) = 0, H12'(0) = 0 (all even polynomials have zero derivative at 0)
    # So g'(0) = 0 automatically! Good.
    
    # q_coeffs[i] = sum of coefficients of x^(i+2) in g
    # g(x) = sum_{k=0}^{12} g_k x^k
    # q(x) = sum_{k=2}^{12} g_k x^(k-2) = sum_{j=0}^{10} g_{j+2} x^j
    
    g_coeffs = np.zeros(13)
    for i, c in enumerate(_H0):
        g_coeffs[i] += c0 * c
    for i, c in enumerate(_H4):
        g_coeffs[i] += c1 * c
    for i, c in enumerate(_H8):
        g_coeffs[i] += c2 * c
    for i, c in enumerate(_H12):
        g_coeffs[i] += c3 * c
    
    q_coeffs = g_coeffs[2:]
    
    # Remove leading zeros
    while len(q_coeffs) > 0 and abs(q_coeffs[0]) < 1e-15:
        q_coeffs = q_coeffs[1:]
    
    if len(q_coeffs) == 0:
        return float('inf')
    
    try:
        roots = np.roots(q_coeffs)
    except:
        return float('inf')
    
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    
    if len(real_roots) == 0:
        return float('inf')
    
    sign_changing_roots = []
    eps = 1e-8
    
    for r in real_roots:
        q_left = np.polyval(q_coeffs, r - eps)
        q_right = np.polyval(q_coeffs, r + eps)
        if q_left * q_right < 0:
            sign_changing_roots.append(r)
    
    if len(sign_changing_roots) == 0:
        return float('inf')
    
    r = max(sign_changing_roots)
    value = r**2 / (2 * np.pi)
    return value

def construct():
    """
    Find coefficients that minimize the objective value.
    Uses multi-start Nelder-Mead optimization with deadline guard.
    """
    start_time = time.time()
    deadline = start_time + 180  # 180 seconds max
    
    # Seed point
    seed = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_val = _compute_value_fast(seed[0], seed[1], seed[2])
    best_coeffs = seed.copy()
    
    # Multi-start Nelder-Mead
    np.random.seed(42)
    
    # Generate multiple starting points
    n_starts = 10
    starts = [seed]
    for i in range(n_starts - 1):
        # Random perturbations around seed
        scale = 0.1 * (i + 1)
        perturbation = np.random.randn(3) * scale
        starts.append(seed + perturbation)
    
    # Also try some structured starting points
    starts.append(np.array([-1.0, 0.1, 0.001]))
    starts.append(np.array([-0.5, 0.2, 0.0005]))
    starts.append(np.array([-2.0, 0.05, 0.002]))
    
    for start in starts:
        if time.time() > deadline - 10:  # 10 second safety margin
            break
        
        try:
            result = minimize(
                lambda x: _compute_value_fast(x[0], x[1], x[2]),
                start,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if result.success or result.fun < best_val:
                if result.fun < best_val:
                    best_val = result.fun
                    best_coeffs = result.x.copy()
        except Exception as e:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        try:
            result = minimize(
                lambda x: _compute_value_fast(x[0], x[1], x[2]),
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
