"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Precompute Hermite polynomial coefficients
# H0(x) = 1
# H4(x) = 16x^4 - 48x^2 + 12
# H8(x) = 65536x^8 - 327680x^6 + 591360x^4 - 353280x^2 + 10080
# H12(x) = 402653184x^12 - 3019898880x^10 + 7488768000x^8 - 7983360000x^6 + 4408992000x^4 - 831801600x^2 + 47900160

def _hermite_coeffs():
    """Return coefficient arrays for H0, H4, H8, H12 (highest power first)."""
    H0 = np.array([1.0])
    H4 = np.array([16.0, 0.0, -48.0, 0.0, 12.0])
    H8 = np.array([65536.0, 0.0, -327680.0, 0.0, 591360.0, 0.0, -353280.0, 0.0, 10080.0])
    H12 = np.array([402653184.0, 0.0, -3019898880.0, 0.0, 7488768000.0, 0.0, -7983360000.0, 0.0, 4408992000.0, 0.0, -831801600.0, 0.0, 47900160.0])
    return H0, H4, H8, H12

def _compute_value(c0, c1, c2, H0, H4, H8, H12):
    """Compute the objective value for given coefficients."""
    # c3 = -(c0*H0(0) + c1*H4(0) + c2*H8(0)) / H12(0)
    H0_0 = H0[0]  # H0(0) = 1
    H4_0 = H4[-1]  # H4(0) = 12
    H8_0 = H8[-1]  # H8(0) = 10080
    H12_0 = H12[-1]  # H12(0) = 47900160
    
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # Build q(x) = g(x) / x^2
    # g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    # Since g(0) = 0, g(x) is divisible by x^2 (all Hermite polys of even degree have only even powers)
    
    # g(x) = sum of even powers. Let's extract coefficients for x^2, x^4, ..., x^12
    # H0: x^0: 1
    # H4: x^4: 16, x^2: -48, x^0: 12
    # H8: x^8: 65536, x^6: -327680, x^4: 591360, x^2: -353280, x^0: 10080
    # H12: x^12: 402653184, x^10: -3019898880, x^8: 7488768000, x^6: -7983360000, x^4: 4408992000, x^2: -831801600, x^0: 47900160
    
    # q(x) = g(x)/x^2 has terms x^0, x^2, ..., x^10
    # q coefficients (x^0 to x^10):
    # x^0: c0*0 + c1*(-48) + c2*(-353280) + c3*(-831801600)  -- wait, H0 has no x^2 term
    
    # Let me be more careful. g(x) = c0*H0(x) + c1*H4(x) + c2*H8(x) + c3*H12(x)
    # H0(x) = 1
    # H4(x) = 16x^4 - 48x^2 + 12
    # H8(x) = 65536x^8 - 327680x^6 + 591360x^4 - 353280x^2 + 10080
    # H12(x) = 402653184x^12 - 3019898880x^10 + 7488768000x^8 - 7983360000x^6 + 4408992000x^4 - 831801600x^2 + 47900160
    
    # g(x) = (c0 + c1*12 + c2*10080 + c3*47900160) + (c1*(-48) + c2*(-353280) + c3*(-831801600))x^2 + ...
    # Since g(0) = 0, the constant term is 0.
    
    # q(x) = g(x)/x^2 = [coeff of x^2 in g] + [coeff of x^4 in g]x^2 + ... + [coeff of x^12 in g]x^10
    
    # Coefficients of g(x) for x^(2k):
    # k=0 (x^0): c0*1 + c1*12 + c2*10080 + c3*47900160 = 0 (by construction)
    # k=1 (x^2): c1*(-48) + c2*(-353280) + c3*(-831801600)
    # k=2 (x^4): c1*16 + c2*591360 + c3*4408992000
    # k=3 (x^6): c2*(-327680) + c3*(-7983360000)
    # k=4 (x^8): c2*65536 + c3*7488768000
    # k=5 (x^10): c3*(-3019898880)
    # k=6 (x^12): c3*402653184
    
    # q(x) = a0 + a2*x^2 + a4*x^4 + a6*x^6 + a8*x^8 + a10*x^10
    # where:
    a0 = c1 * (-48.0) + c2 * (-353280.0) + c3 * (-831801600.0)
    a2 = c1 * 16.0 + c2 * 591360.0 + c3 * 4408992000.0
    a4 = c2 * (-327680.0) + c3 * (-7983360000.0)
    a6 = c2 * 65536.0 + c3 * 7488768000.0
    a8 = c3 * (-3019898880.0)
    a10 = c3 * 402653184.0
    
    # q(x) = a10*x^10 + a8*x^8 + a6*x^6 + a4*x^4 + a2*x^2 + a0
    # Find roots of q(x) = 0
    q_coeffs = np.array([a10, 0.0, a8, 0.0, a6, 0.0, a4, 0.0, a2, 0.0, a0])
    
    # Find real roots
    roots = np.roots(q_coeffs)
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    
    if len(real_roots) == 0:
        return np.inf
    
    # Find the largest real root at which q changes sign
    # A root changes sign if it has odd multiplicity
    # For numerical stability, check sign change in a small neighborhood
    
    best_root = -np.inf
    for r in real_roots:
        if r <= 0:
            continue
        # Check if q changes sign at r
        eps = 1e-8
        q_minus = np.polyval(q_coeffs, r - eps)
        q_plus = np.polyval(q_coeffs, r + eps)
        if q_minus * q_plus < 0:
            if r > best_root:
                best_root = r
    
    if best_root == -np.inf:
        return np.inf
    
    value = best_root**2 / (2.0 * np.pi)
    return value

def construct():
    start_time = time.time()
    deadline = start_time + 180.0
    
    H0, H4, H8, H12 = _hermite_coeffs()
    
    # Seed value
    seed_c = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    seed_value = _compute_value(seed_c[0], seed_c[1], seed_c[2], H0, H4, H8, H12)
    best_c = seed_c.copy()
    best_value = seed_value
    
    # Objective function for optimizer
    def objective(c):
        if time.time() > deadline - 5.0:
            return best_value
        return _compute_value(c[0], c[1], c[2], H0, H4, H8, H12)
    
    # Local optimization from seed
    try:
        result = minimize(objective, seed_c, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
        if result.fun < best_value:
            best_value = result.fun
            best_c = result.x.copy()
    except Exception:
        pass
    
    # Try a few random perturbations around the seed
    rng = np.random.default_rng(42)
    for i in range(50):
        if time.time() > deadline - 10.0:
            break
        perturbation = rng.normal(0, 0.01, size=3)
        c_trial = seed_c + perturbation
        val = _compute_value(c_trial[0], c_trial[1], c_trial[2], H0, H4, H8, H12)
        if val < best_value:
            best_value = val
            best_c = c_trial.copy()
    
    # Try larger perturbations
    for i in range(30):
        if time.time() > deadline - 5.0:
            break
        perturbation = rng.normal(0, 0.1, size=3)
        c_trial = seed_c + perturbation
        val = _compute_value(c_trial[0], c_trial[1], c_trial[2], H0, H4, H8, H12)
        if val < best_value:
            best_value = val
            best_c = c_trial.copy()
    
    # Final polish around best found
    if time.time() < deadline - 5.0:
        try:
            result = minimize(objective, best_c, method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
            if result.fun < best_value:
                best_value = result.fun
                best_c = result.x.copy()
        except Exception:
            pass
    
    return best_c
# EVOLVE-BLOCK-END
