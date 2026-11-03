"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: LOW-DIMENSIONAL SMOOTH PARAMETRIC OPTIMIZATION
- The optimal solution is likely smooth. We represent f as a small number
  of smooth basis functions (Gaussians with varying width and center,
  or a low-order polynomial bump) to reduce the search space from 600 dims
  to ~10-20 dims.
- We use scipy.optimize.minimize (Nelder-Mead or L-BFGS-B) on these coefficients
  to minimize the exact score.
- We also try a few different parametric families and initial conditions.
- This is structurally different from the current 600-dim active-set gradient
  descent: we're searching a low-dimensional manifold of smooth functions.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 600
TIME_LIMIT = 145  # Safety margin for 150s limit

def compute_score_exact(f):
    """Compute the normalized maximum of the autoconvolution exactly."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def compute_conv(f):
    """Compute full autoconvolution."""
    return np.convolve(f, f, mode='full')

def parametric_to_step(coeffs, family):
    """
    Convert parametric coefficients to a 600-dim step function.
    
    Families:
    - 'gauss': f(x) = sum_i a_i * exp(-((x - c_i)/w_i)^2)
    - 'poly': f(x) = max(0, sum_{k=0}^{K} p_k * (x - 0.25)^k * (0.25 - x)^k)
    - 'cosine': f(x) = max(0, sum_i a_i * cos(2*pi*f_i*x + phi_i))
    """
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25/N
    
    if family == 'gauss':
        # coeffs: [a1, c1, w1, a2, c2, w2, ...]
        f = np.zeros(N)
        n_gauss = len(coeffs) // 3
        for i in range(n_gauss):
            a = coeffs[3*i]
            c = coeffs[3*i+1]
            w = coeffs[3*i+2]
            if w < 1e-6:
                w = 1e-6
            f += a * np.exp(-((x - c)/w)**2)
        return np.maximum(f, 0.0)
    
    elif family == 'cosine':
        # coeffs: [a1, f1, phi1, a2, f2, phi2, ...]
        f = np.zeros(N)
        n_terms = len(coeffs) // 3
        for i in range(n_terms):
            a = coeffs[3*i]
            freq = coeffs[3*i+1]
            phi = coeffs[3*i+2]
            f += a * np.cos(2*np.pi*freq*x + phi)
        return np.maximum(f, 0.0)
    
    elif family == 'plateau':
        # coeffs: [a, w, c] -> f(x) = a * exp(-((x-c)/w)^2) with clipping
        a = coeffs[0]
        w = coeffs[1]
        c = coeffs[2]
        if w < 1e-6:
            w = 1e-6
        f = a * np.exp(-((x - c)/w)**2)
        return np.maximum(f, 0.0)
    
    return np.maximum(f, 0.0)

def objective_gauss(coeffs):
    """Objective for Gaussian mixture parametric model."""
    f = parametric_to_step(coeffs, 'gauss')
    s = np.sum(f)
    if s < 1e-12:
        return 1e10
    conv = compute_conv(f)
    return np.max(conv) / (s * s)

def objective_cosine(coeffs):
    """Objective for cosine parametric model."""
    f = parametric_to_step(coeffs, 'cosine')
    s = np.sum(f)
    if s < 1e-12:
        return 1e10
    conv = compute_conv(f)
    return np.max(conv) / (s * s)

def objective_plateau(coeffs):
    """Objective for single Gaussian plateau."""
    f = parametric_to_step(coeffs, 'plateau')
    s = np.sum(f)
    if s < 1e-12:
        return 1e10
    conv = compute_conv(f)
    return np.max(conv) / (s * s)

def construct():
    start_time = time.time()
    
    best_score = np.inf
    best_f = None
    
    # Family 1: Single Gaussian
    # coeffs: [a, c, w]
    for a0 in [1.0, 2.0, 0.5]:
        for c0 in [-0.1, 0.0, 0.1]:
            for w0 in [0.05, 0.1, 0.15, 0.2]:
                if time.time() - start_time > TIME_LIMIT - 5:
                    break
                x0 = np.array([a0, c0, w0])
                try:
                    res = minimize(objective_plateau, x0, method='Nelder-Mead',
                                   options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
                    if res.fun < best_score:
                        best_score = res.fun
                        best_f = parametric_to_step(res.x, 'plateau')
                except:
                    pass
    
    # Family 2: Two Gaussians
    # coeffs: [a1, c1, w1, a2, c2, w2]
    rng = np.random.RandomState(42)
    for trial in range(30):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        x0 = np.array([
            rng.uniform(0.5, 3.0), rng.uniform(-0.2, 0.2), rng.uniform(0.03, 0.15),
            rng.uniform(0.5, 3.0), rng.uniform(-0.2, 0.2), rng.uniform(0.03, 0.15)
        ])
        try:
            res = minimize(objective_gauss, x0, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-7})
            if res.fun < best_score:
                best_score = res.fun
                best_f = parametric_to_step(res.x, 'gauss')
        except:
            pass
    
    # Family 3: Cosine basis (3 terms)
    for trial in range(20):
        if time.time() - start_time > TIME_LIMIT - 15:
            break
        x0 = np.array([
            rng.uniform(0.5, 3.0), rng.uniform(0.5, 5.0), rng.uniform(0, 2*np.pi),
            rng.uniform(0.5, 3.0), rng.uniform(0.5, 5.0), rng.uniform(0, 2*np.pi),
            rng.uniform(0.5, 3.0), rng.uniform(0.5, 5.0), rng.uniform(0, 2*np.pi)
        ])
        try:
            res = minimize(objective_cosine, x0, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-7})
            if res.fun < best_score:
                best_score = res.fun
                best_f = parametric_to_step(res.x, 'cosine')
        except:
            pass
    
    # If best_f is None, fall back to a simple Gaussian
    if best_f is None:
        x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25/N
        best_f = np.exp(-10 * x**2)
        best_score = compute_score_exact(best_f)
    
    # Local refinement: use the best parametric solution as initial point
    # for a short gradient descent in the full 600-dim space
    if time.time() - start_time < TIME_LIMIT - 30:
        f = best_f.copy()
        s = np.sum(f)
        conv = compute_conv(f)
        current_score = np.max(conv) / (s * s)
        
        alpha = 1e-4
        for i in range(5000):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
                
            s = np.sum(f)
            if s < 1e-12:
                break
            conv = compute_conv(f)
            max_c = np.max(conv)
            current_score = max_c / (s * s)
            
            if current_score < best_score:
                best_score = current_score
                best_f = f.copy()
            
            # Gradient of the max lag
            max_idx = np.argmax(conv)
            k_max = max_idx
            C_k = conv[k_max]
            
            g_k = np.zeros(N)
            j_start = max(0, k_max - N + 1)
            j_end = min(N, k_max + 1)
            if j_start < j_end:
                idx_f = k_max - np.arange(j_start, j_end)
                g_k[j_start:j_end] += f[idx_f]
            
            j_start2 = max(0, k_max)
            j_end2 = min(N, k_max + N)
            if j_start2 < j_end2:
                idx_f2 = np.arange(j_start2, j_end2) - k_max
                g_k[j_start2:j_end2] += f[idx_f2]
            
            base = 1.0 / (s * s)
            const_term = 2.0 * C_k / (s * s * s)
            grad = base * g_k - const_term
            
            # Line search
            found_alpha = False
            current_alpha = alpha
            for ls_iter in range(15):
                f_trial = f - current_alpha * grad
                f_trial = np.maximum(f_trial, 0.0)
                s_trial = np.sum(f_trial)
                if s_trial < 1e-12:
                    current_alpha *= 0.5
                    continue
                conv_trial = compute_conv(f_trial)
                score_trial = np.max(conv_trial) / (s_trial * s_trial)
                if score_trial < current_score:
                    found_alpha = True
                    f = f_trial
                    alpha = min(current_alpha * 1.1, 1e-2)
                    break
                else:
                    current_alpha *= 0.5
                    if current_alpha < 1e-10:
                        break
            
            if not found_alpha and current_alpha < 1e-10:
                break
    
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    start = time.time()
    f = construct()
    elapsed = time.time() - start
    score = compute_score_exact(f)
    print(f"Score: {score}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
    print(f"Time: {elapsed:.2f}s")
