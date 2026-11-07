"""Robust root-finding and local search around seed."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, brentq

# Precompute Hermite polynomial coefficients
def _hermite_coeffs(n):
    if n == 0: return np.array([1.0])
    if n == 1: return np.array([2.0, 0.0])
    h0 = np.array([1.0])
    h1 = np.array([2.0, 0.0])
    for k in range(2, n + 1):
        term1 = np.zeros(len(h1) + 1)
        term1[:-1] = 2.0 * h1
        term2 = np.zeros(len(h0) + 1)
        term2[:-1] = 2.0 * (k - 1) * h0
        max_len = max(len(term1), len(term2))
        term1_padded = np.zeros(max_len)
        term2_padded = np.zeros(max_len)
        term1_padded[:len(term1)] = term1
        term2_padded[:len(term2)] = term2
        h_new = term1_padded - term2_padded
        h0, h1 = h1, h_new
    return h1

H0_c = _hermite_coeffs(0)
H4_c = _hermite_coeffs(4)
H8_c = _hermite_coeffs(8)
H12_c = _hermite_coeffs(12)
H0_0, H4_0, H8_0, H12_0 = 1.0, 12.0, 1680.0, 2032128.0

def build_q_coeffs(c0, c1, c2):
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    deg_max = 12
    g = np.zeros(deg_max + 1)
    for c, hc in [(c0, H0_c), (c1, H4_c), (c2, H8_c), (c3, H12_c)]:
        pad = np.zeros(deg_max + 1 - len(hc))
        hc_pad = np.concatenate([pad, hc])
        g += c * hc_pad
    return g[:-2]

def find_largest_sign_change_root_robust(q_coeffs):
    """Find largest real root of q where q changes sign, using brentq for refinement."""
    # Get candidate roots from np.roots
    roots = np.roots(q_coeffs)
    real_roots = roots[np.abs(roots.imag) < 1e-8].real
    real_roots = np.unique(np.round(real_roots, decimals=6)) # Remove duplicates
    
    valid_roots = []
    for r in real_roots:
        if r <= 0: continue # We care about positive roots for "largest"
        
        # Refine root using brentq
        # Find interval [a, b] containing r where sign changes
        delta = 1e-4 * max(1.0, abs(r))
        a, b = r - delta, r + delta
        fa = np.polyval(q_coeffs, a)
        fb = np.polyval(q_coeffs, b)
        
        if fa * fb < 0:
            try:
                r_refined = brentq(lambda x: np.polyval(q_coeffs, x), a, b, xtol=1e-12)
                valid_roots.append(r_refined)
            except Exception:
                pass
        else:
            # Check if r is a double root or if brentq interval was too small/large
            # If fa and fb have same sign, it might not be a sign-changing root in this window
            # But np.roots said it's a root. Let's check a wider window.
            a2, b2 = r - 1.0, r + 1.0
            if a2 < 0: a2 = 1e-6
            fa2 = np.polyval(q_coeffs, a2)
            fb2 = np.polyval(q_coeffs, b2)
            if fa2 * fb2 < 0:
                try:
                    r_refined = brentq(lambda x: np.polyval(q_coeffs, x), a2, b2, xtol=1e-12)
                    valid_roots.append(r_refined)
                except Exception:
                    pass

    if not valid_roots:
        return None
    return max(valid_roots)

def compute_value_robust(c0, c1, c2):
    q = build_q_coeffs(c0, c1, c2)
    r = find_largest_sign_change_root_robust(q)
    if r is None:
        return float('inf')
    return r * r / (2.0 * np.pi)

def construct():
    start_time = time.time()
    deadline = start_time + 200.0
    
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = compute_value_robust(*seed)
    best_params = seed.copy()
    
    # Local search around seed
    # The seed is known to be good. We refine it.
    for i in range(10):
        if time.time() > deadline - 10: break
        
        # Multi-start around current best
        for j in range(5):
            if time.time() > deadline - 10: break
            # Random perturbation
            noise = np.random.normal(0, 0.01, 3)
            x0 = best_params + noise
            
            try:
                res = minimize(lambda p: compute_value_robust(*p), x0, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x
            except Exception:
                pass
    
    # Try a few global-ish starts if time permits
    if time.time() < deadline - 30:
        for _ in range(20):
            if time.time() > deadline - 30: break
            x0 = np.random.uniform(-5, 5, 3)
            try:
                res = minimize(lambda p: compute_value_robust(*p), x0, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x
            except Exception:
                pass
                
    return best_params
# EVOLVE-BLOCK-END
