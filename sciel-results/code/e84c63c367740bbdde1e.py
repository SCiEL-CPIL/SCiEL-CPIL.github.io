"""Rewritten program: local search around the seed to minimize the value."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def hermite_coeffs(n):
    """Return coefficients of H_n(x) as a list [c0, c1, ..., cn] where c_i is coeff of x^i."""
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 2.0]
    
    H_nm2 = [1.0]  # H_0
    H_nm1 = [0.0, 2.0] # H_1
    
    for k in range(2, n + 1):
        # H_k(x) = 2x H_{k-1}(x) - 2(k-1) H_{k-2}(x)
        # term1 = 2x * H_{k-1}
        term1 = [0.0] + [2.0 * c for c in H_nm1]
        # term2 = -2(k-1) * H_{k-2}
        term2 = [-2.0 * (k - 1) * c for c in H_nm2]
        
        max_len = max(len(term1), len(term2))
        while len(term1) < max_len:
            term1.append(0.0)
        while len(term2) < max_len:
            term2.append(0.0)
            
        H_n = [t1 + t2 for t1, t2 in zip(term1, term2)]
        
        H_nm2 = H_nm1
        H_nm1 = H_n
        
    return H_nm1

H0_coeffs = np.array(hermite_coeffs(0))
H4_coeffs = np.array(hermite_coeffs(4))
H8_coeffs = np.array(hermite_coeffs(8))
H12_coeffs = np.array(hermite_coeffs(12))

# Precompute values at 0
H0_0 = H0_coeffs[0]
H4_0 = H4_coeffs[0]
H8_0 = H8_coeffs[0]
H12_0 = H12_coeffs[0]

def evaluate_value(c0, c1, c2):
    """
    Compute the value for given coefficients c0, c1, c2 using floats.
    Returns the value (float) or None if invalid.
    """
    try:
        c0_f = float(c0)
        c1_f = float(c1)
        c2_f = float(c2)
    except:
        return None

    # Construct g(x) coefficients
    # g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    # We need c3 such that g(0) = 0
    # c0*H0(0) + c1*H4(0) + c2*H8(0) + c3*H12(0) = 0
    if H12_0 == 0:
        return None
    
    c3_f = -(c0_f * H0_0 + c1_f * H4_0 + c2_f * H8_0) / H12_0

    # g_coeffs of degree 12
    g_coeffs = np.zeros(13)
    g_coeffs[:len(H0_coeffs)] += c0_f * H0_coeffs
    g_coeffs[:len(H4_coeffs)] += c1_f * H4_coeffs
    g_coeffs[:len(H8_coeffs)] += c2_f * H8_coeffs
    g_coeffs[:len(H12_coeffs)] += c3_f * H12_coeffs

    # Check if leading coefficient is negative, flip if so
    if g_coeffs[12] < 0:
        g_coeffs = -g_coeffs

    # Check divisibility by x^2 (should be true by construction of even Hermite polys)
    # g(0) is ~0, g'(0) is ~0
    if abs(g_coeffs[0]) > 1e-9 or abs(g_coeffs[1]) > 1e-9:
        return None

    # q(x) = g(x) / x^2
    # q_coeffs are g_coeffs[2:]
    q_coeffs = g_coeffs[2:] # length 11
    
    # Find roots of q(x)
    # np.roots expects highest power first
    q_reversed = q_coeffs[::-1]
    
    try:
        roots = np.roots(q_reversed)
    except:
        return None
    
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-9:
            real_roots.append(r.real)
    
    if not real_roots:
        return None
    
    real_roots.sort(reverse=True)
    
    # Find largest real root where q changes sign
    # We evaluate q(x) at roots to check sign change
    # q(x) = sum q_coeffs[i] x^i
    def q_val(x):
        return np.polyval(q_coeffs[::-1], x)
        
    r_best = None
    for r in real_roots:
        if r <= 0: # Roots must be positive for r^2 to be meaningful in context? 
            # The task says "largest real root". If the largest is negative, r^2 is still positive.
            # But "changes sign" implies a simple real root.
            pass
        eps = 1e-6
        # Ensure we don't step into another root or complex region
        # Check sign change
        q_left = q_val(r - eps)
        q_right = q_val(r + eps)
        
        if q_left * q_right < 0:
            r_best = r
            break
    
    if r_best is None:
        return None
    
    value = (r_best ** 2) / (2 * np.pi)
    return value

def construct():
    best_c = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_val = evaluate_value(best_c[0], best_c[1], best_c[2])
    
    if best_val is None:
        return best_c

    start_time = time.time()
    deadline = 200.0 # Safety margin for 210s limit

    if HAS_SCIPY:
        def objective(x):
            val = evaluate_value(x[0], x[1], x[2])
            if val is None:
                return 1e10
            return val
        
        # Multi-start Nelder-Mead
        starts = [best_c]
        # Add some random perturbations
        rng = np.random.default_rng(42)
        for _ in range(5):
            perturbation = rng.normal(0, 0.01, size=3)
            starts.append(best_c + perturbation)
        
        for x0 in starts:
            if time.time() > deadline - 10:
                break
            try:
                res = minimize(objective, x0, method='Nelder-Mead', 
                               options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
                if time.time() > deadline - 5:
                    break
                val_new = evaluate_value(res.x[0], res.x[1], res.x[2])
                if val_new is not None and val_new < best_val:
                    best_c = res.x
                    best_val = val_new
            except:
                pass
        
        # Final polish with tighter tolerance on the best found
        if time.time() < deadline - 10:
            try:
                res_polish = minimize(objective, best_c, method='Nelder-Mead', 
                                      options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12})
                val_polish = evaluate_value(res_polish.x[0], res_polish.x[1], res_polish.x[2])
                if val_polish is not None and val_polish < best_val:
                    best_c = res_polish.x
                    best_val = val_polish
            except:
                pass

    return best_c
# EVOLVE-BLOCK-END
