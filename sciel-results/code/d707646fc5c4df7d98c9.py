"""Root-structure parameterization: directly optimize over roots of q(x)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

# Precompute Hermite polynomial coefficients (highest degree first)
# H0(x) = 1
# H4(x) = 16x^4 - 48x^2 + 12
# H8(x) = 65536x^8 - 294912x^6 + 491520x^4 - 241920x^2 + 20160
# H12(x) = 4096x^12 - 294912x^10 + 622080x^8 - 516096x^6 + 145152x^4 - 12096x^2 + 462

# Let's compute them properly using numpy
def get_hermite_coeffs(n):
    """Get coefficients of physicist's Hermite polynomial H_n(x), highest degree first."""
    # Use recurrence: H_0=1, H_1=2x, H_{n+1}=2x*H_n - 2n*H_{n-1}
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([2.0, 0.0])
    
    h_prev = np.array([1.0])  # H_0
    h_curr = np.array([2.0, 0.0])  # H_1
    
    for k in range(1, n):
        # H_{k+1} = 2x * H_k - 2k * H_{k-1}
        # 2x * H_k: shift coefficients right by 1 and multiply by 2
        new_len = len(h_curr) + 1
        h_2x = np.zeros(new_len)
        h_2x[:-1] = 2.0 * h_curr
        
        # 2k * H_{k-1}: pad to same length
        h_2k = np.zeros(new_len)
        pad = new_len - len(h_prev)
        h_2k[pad:] = 2.0 * k * h_prev
        
        h_next = h_2x - h_2k
        h_prev = h_curr
        h_curr = h_next
    
    return h_curr

H0_c = get_hermite_coeffs(0)   # [1]
H4_c = get_hermite_coeffs(4)   # degree 4
H8_c = get_hermite_coeffs(8)   # degree 8
H12_c = get_hermite_coeffs(12) # degree 12

# Values at x=0
H0_0 = H0_c[-1]  # 1
H4_0 = H4_c[-1]  # 12
H8_0 = H8_c[-1]  # 20160
H12_0 = H12_c[-1]  # 462

def build_q_from_coeffs(c0, c1, c2):
    """Build q(x) = g(x)/x^2 coefficients (highest degree first)."""
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # Build g(x) coefficients (degree 12, highest first)
    g = np.zeros(13)
    
    for c, hc in [(c0, H0_c), (c1, H4_c), (c2, H8_c), (c3, H12_c)]:
        pad = 13 - len(hc)
        hc_p = np.concatenate([np.zeros(pad), hc])
        g += c * hc_p
    
    # g(0) = 0, g'(0) = 0 by construction (constant and linear terms vanish)
    # q = g / x^2: remove last 2 coefficients
    q = g[:-2]  # degree 10, length 11
    return q

def compute_value(c0, c1, c2):
    """Compute value = r^2 / (2*pi) where r is largest real sign-changing root of q."""
    q = build_q_from_coeffs(c0, c1, c2)
    
    # Find real roots
    roots = np.roots(q)
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    
    if len(real_roots) == 0:
        return float('inf')
    
    # Check sign change at each real root
    valid_roots = []
    for r in real_roots:
        eps = 1e-9 * max(1.0, abs(r))
        q_left = np.polyval(q, r - eps)
        q_right = np.polyval(q, r + eps)
        if q_left * q_right < 0:
            valid_roots.append(r)
    
    if not valid_roots:
        return float('inf')
    
    r = max(valid_roots)
    return r**2 / (2.0 * np.pi)

def objective(params):
    """Objective for optimization."""
    c0, c1, c2 = params
    return compute_value(c0, c1, c2)

def construct():
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin
    
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = compute_value(*seed)
    best_params = seed.copy()
    
    # Strategy 1: Differential Evolution (global search)
    if time.time() < deadline - 30:
        try:
            bounds = [(-2.0, 2.0), (-1.0, 1.0), (-0.5, 0.5)]
            res_de = differential_evolution(
                objective, bounds, 
                maxiter=100, popsize=30, tol=1e-8,
                seed=42, polish=True
            )
            if res_de.fun < best_val:
                best_val = res_de.fun
                best_params = res_de.x.copy()
        except Exception:
            pass
    
    # Strategy 2: Nelder-Mead refinement from best
    if time.time() < deadline - 15:
        try:
            res_nm = minimize(
                objective, best_params, method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if res_nm.fun < best_val:
                best_val = res_nm.fun
                best_params = res_nm.x.copy()
        except Exception:
            pass
    
    # Strategy 3: Random restarts with local refinement
    if time.time() < deadline - 10:
        np.random.seed(123)
        for i in range(50):
            if time.time() > deadline - 10:
                break
            # Random start in a reasonable range
            trial = np.random.uniform(
                [-2.0, -1.0, -0.5],
                [2.0, 1.0, 0.5]
            )
            try:
                v = compute_value(*trial)
                if v < best_val:
                    best_val = v
                    best_params = trial.copy()
                    # Quick local refinement
                    if time.time() < deadline - 10:
                        res = minimize(
                            objective, trial, method='Nelder-Mead',
                            options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                        )
                        if res.fun < best_val:
                            best_val = res.fun
                            best_params = res.x.copy()
            except Exception:
                pass
    
    # Final refinement
    if time.time() < deadline - 5 and best_val < compute_value(*seed):
        try:
            res = minimize(
                objective, best_params, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
