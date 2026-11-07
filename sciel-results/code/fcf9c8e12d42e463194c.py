"""Optimized program for uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

# Physicists' Hermite polynomials H_n(x) from sympy's hermite_poly
# H_0(x) = 1
# H_4(x) = 16x^4 - 48x^2 + 12
# H_8(x) = 65536x^8 - 147456x^6 + 136896x^4 - 32256x^2 + 1680
# H_12(x) = 68719476736x^12 - 332869504512x^10 + 575508845056x^8 - 414219003904x^6 + 128381414912x^4 - 14155780608x^2 + 665280

# Let me verify H_8: The recurrence is H_{n+1} = 2x*H_n - 2n*H_{n-1}
# H_0 = 1
# H_1 = 2x
# H_2 = 2x(2x) - 2(1)(1) = 4x^2 - 2
# H_3 = 2x(4x^2-2) - 4(2x) = 8x^3 - 4x - 8x = 8x^3 - 12x
# H_4 = 2x(8x^3-12x) - 8(4x^2-2) = 16x^4 - 24x^2 - 32x^2 + 16 = 16x^4 - 56x^2 + 16
# Wait, that gives 16x^4 - 56x^2 + 16, not 16x^4 - 48x^2 + 12

# Let me redo: H_{n+1} = 2x H_n - 2n H_{n-1}
# H_0 = 1
# H_1 = 2x
# H_2 = 2x*H_1 - 2*1*H_0 = 2x*2x - 2 = 4x^2 - 2
# H_3 = 2x*H_2 - 2*2*H_1 = 2x(4x^2-2) - 4(2x) = 8x^3 - 4x - 8x = 8x^3 - 12x
# H_4 = 2x*H_3 - 2*3*H_2 = 2x(8x^3-12x) - 6(4x^2-2) = 16x^4 - 24x^2 - 24x^2 + 12 = 16x^4 - 48x^2 + 12
# OK so H_4 = 16x^4 - 48x^2 + 12. Good.
# H_5 = 2x*H_4 - 2*4*H_3 = 2x(16x^4-48x^2+12) - 8(8x^3-12x) = 32x^5 - 96x^3 + 24x - 64x^3 + 96x = 32x^5 - 160x^3 + 120x
# H_6 = 2x*H_5 - 2*5*H_4 = 2x(32x^5-160x^3+120x) - 10(16x^4-48x^2+12) = 64x^6 - 320x^4 + 240x^2 - 160x^4 + 480x^2 - 120 = 64x^6 - 480x^4 + 720x^2 - 120
# H_7 = 2x*H_6 - 2*6*H_5 = 2x(64x^6-480x^4+720x^2-120) - 12(32x^5-160x^3+120x) = 128x^7 - 960x^5 + 1440x^3 - 240x - 384x^5 + 1920x^3 - 1440x = 128x^7 - 1344x^5 + 3360x^3 - 1680x
# H_8 = 2x*H_7 - 2*7*H_6 = 2x(128x^7-1344x^5+3360x^3-1680x) - 14(64x^6-480x^4+720x^2-120)
#     = 256x^8 - 2688x^6 + 6720x^4 - 3360x^2 - 896x^6 + 6720x^4 - 10080x^2 + 1680
#     = 256x^8 - 3584x^6 + 13440x^4 - 13440x^2 + 1680

# H_8(0) = 1680. Let me verify: (-1)^4 * 8!/4! = 40320/24 = 1680. Yes!

# H_9 = 2x*H_8 - 2*8*H_7
# H_10 = 2x*H_9 - 2*9*H_8
# H_11 = 2x*H_10 - 2*10*H_9
# H_12 = 2x*H_11 - 2*11*H_10

# This is getting complex. Let me just compute them programmatically.

def compute_hermite_coeffs():
    """Compute coefficients of physicists' Hermite polynomials H_0, H_4, H_8, H_12."""
    # Use recurrence H_{n+1} = 2x*H_n - 2n*H_{n-1}
    # Represent polynomials as arrays of coefficients (ascending power)
    
    H = [None] * 13
    H[0] = np.array([1.0])  # 1
    H[1] = np.array([0.0, 2.0])  # 2x
    
    for n in range(1, 12):
        # H[n+1] = 2x * H[n] - 2n * H[n-1]
        # 2x * H[n]: shift coefficients up by 1
        len_n = len(H[n])
        two_x_Hn = np.zeros(len_n + 1)
        two_x_Hn[1:] = 2.0 * H[n]
        
        # 2n * H[n-1]
        len_nm1 = len(H[n-1])
        two_n_Hnm1 = np.zeros(len_n + 1)
        two_n_Hnm1[:len_nm1] = 2.0 * n * H[n-1]
        
        H[n+1] = two_x_Hn - two_n_Hnm1
    
    return H[0], H[4], H[8], H[12]

H0_coeffs, H4_coeffs, H8_coeffs, H12_coeffs = compute_hermite_coeffs()

# Values at x=0 (constant terms)
H0_0 = H0_coeffs[0]
H4_0 = H4_coeffs[0]
H8_0 = H8_coeffs[0]
H12_0 = H12_coeffs[0]


def build_q(c0, c1, c2):
    """Build q(x) = g(x)/x^2. Returns coefficients in ascending power order."""
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # g = c0*H0 + c1*H4 + c2*H8 + c3*H12
    # All are even polynomials
    max_deg = 12
    g = np.zeros(max_deg + 1)  # ascending powers
    g[:len(H0_coeffs)] += c0 * H0_coeffs
    g[:len(H4_coeffs)] += c1 * H4_coeffs
    g[:len(H8_coeffs)] += c2 * H8_coeffs
    g[:len(H12_coeffs)] += c3 * H12_coeffs
    
    # q = g / x^2, so q[k] = g[k+2]
    q = np.zeros(max_deg - 1)  # degree 10
    q[:len(g)-2] = g[2:]
    
    # Trim trailing zeros
    while len(q) > 1 and abs(q[-1]) < 1e-30:
        q = q[:-1]
    
    return q


def compute_value(c0, c1, c2):
    """Compute value = r^2/(2*pi) where r is largest sign-changing real root of q."""
    q = build_q(c0, c1, c2)
    
    # Find roots of q (ascending order, need descending for np.roots)
    q_desc = q[::-1]
    roots = np.roots(q_desc)
    real_roots = roots[np.abs(roots.imag) < 1e-9].real
    
    if len(real_roots) == 0:
        return 1e10
    
    # Check sign change at each real root
    eps = 1e-8
    sign_changing = []
    for r in real_roots:
        qm = np.polyval(q_desc, r - eps)
        qp = np.polyval(q_desc, r + eps)
        if qm * qp < 0:
            sign_changing.append(r)
    
    if len(sign_changing) == 0:
        return 1e10
    
    r = max(sign_changing)
    return r**2 / (2.0 * np.pi)


def objective(c):
    try:
        return compute_value(c[0], c[1], c[2])
    except Exception:
        return 1e10


def construct():
    start_time = time.time()
    deadline = start_time + 175.0  # Safety margin
    
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_c = seed.copy()
    best_val = compute_value(seed[0], seed[1], seed[2])
    
    # Local Nelder-Mead from seed
    try:
        res = minimize(objective, seed, method='Nelder-Mead',
                       options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16,
                                'adaptive': True})
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x.copy()
    except Exception:
        pass
    
    if time.time() > deadline - 20:
        return np.array(best_c)
    
    # Random restarts
    rng = np.random.RandomState(123)
    for i in range(8):
        if time.time() > deadline - 20:
            break
        scale = 0.3 / (1.0 + 0.5 * i)
        x0 = best_c + rng.randn(3) * scale
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-15,
                                    'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
    
    # Broader search with differential evolution if time permits
    if time.time() < deadline - 50:
        try:
            bounds = [(-3.0, 3.0), (-0.5, 0.5), (-0.1, 0.1)]
            de_res = differential_evolution(objective, bounds, seed=42,
                                             maxiter=50, popsize=20,
                                             tol=1e-12, polish=True,
                                             mutation=(0.3, 1.0))
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_c = de_res.x.copy()
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 5:
        try:
            res = minimize(objective, best_c, method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-17,
                                    'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_c)
# EVOLVE-BLOCK-END
