"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Structural change: exploit symmetry and parametric families.
The optimal function is likely symmetric about 0. We parameterize the
non-negative half (300 bins) and mirror it. Then we optimize over a
parametric family (Gaussian, cosine, polynomial) and do local coordinate
descent on the 300 free parameters.

Key insight: the autoconvolution peak is at lag 0 (center). By making the
function spread out more, we reduce the peak relative to the integral.
The optimal shape balances peak height against spread.
"""

import numpy as np
from scipy.optimize import minimize_scalar, minimize
import time

N = 600
HALF_N = N // 2  # 300
TIME_LIMIT = 140  # safety margin before 150s kill

def fft_autoconv(f):
    """Compute full autoconvolution via FFT."""
    n = len(f)
    size = 2 * n - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    F = np.fft.rfft(f, n=fft_size)
    conv = np.fft.irfft(F * F, n=fft_size)
    return conv[:size]

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    conv = fft_autoconv(f)
    return np.max(conv) / (s * s)

def make_symmetric(half_f):
    """Mirror the half function to make a full symmetric function of length N."""
    # half_f has HALF_N elements representing bins 0..HALF_N-1
    # The full function: bins 0..HALF_N-1 = half_f, bins HALF_N..N-1 = reversed half_f
    return np.concatenate([half_f, half_f[::-1]])

def symmetric_score(half_f):
    """Score for a symmetric function given its first half."""
    f = make_symmetric(half_f)
    return compute_score(f)

def gaussian_half(alpha, beta=0.0):
    """Gaussian profile on the half domain."""
    # x ranges from -1/4 to 0 (left half)
    x = np.linspace(-0.25, 0.0, HALF_N, endpoint=False) + (0.5 / N)
    # Shift so center is at 0
    x_centered = x + 0.25  # now ranges from 0 to 0.25
    f = np.exp(-alpha * x_centered ** 2)
    if beta != 0:
        f = f * np.exp(-beta * np.abs(x_centered))
    return f

def cosine_half(width, order=1):
    """Cosine bump on the half domain."""
    x = np.linspace(-0.25, 0.0, HALF_N, endpoint=False) + (0.5 / N)
    x_centered = x + 0.25  # 0 to 0.25
    f = np.zeros(HALF_N)
    mask = x_centered <= width / 2.0
    if np.sum(mask) == 0:
        return f
    t = x_centered[mask] / (width / 2.0)  # 0 to 1
    f[mask] = np.cos(np.pi * t / 2.0) ** order
    return f

def polynomial_half(width, power=2):
    """Polynomial bump on the half domain."""
    x = np.linspace(-0.25, 0.0, HALF_N, endpoint=False) + (0.5 / N)
    x_centered = x + 0.25  # 0 to 0.25
    f = np.zeros(HALF_N)
    mask = x_centered <= width / 2.0
    if np.sum(mask) == 0:
        return f
    t = x_centered[mask] / (width / 2.0)  # 0 to 1
    f[mask] = (1.0 - t ** 2) ** power
    f = np.maximum(f, 0)
    return f

def optimize_parametric(start_time, max_time):
    """Optimize over parametric families."""
    best_half = None
    best_score = np.inf
    
    # Gaussian: optimize alpha
    for beta in [0.0, 0.5, 1.0, 2.0]:
        if time.time() - start_time > max_time:
            break
        def obj_alpha(alpha):
            return symmetric_score(gaussian_half(alpha, beta))
        res = minimize_scalar(obj_alpha, bounds=(0.1, 200.0), method='bounded',
                              options={'xatol': 1e-6, 'maxiter': 100})
        if res.success:
            s = res.fun
            if s < best_score:
                best_score = s
                best_half = gaussian_half(res.x, beta).copy()
    
    # Cosine: optimize width and order
    for order in [1, 2, 3, 4]:
        if time.time() - start_time > max_time:
            break
        def obj_width(width):
            return symmetric_score(cosine_half(width, order))
        res = minimize_scalar(obj_width, bounds=(0.05, 0.5), method='bounded',
                              options={'xatol': 1e-6, 'maxiter': 100})
        if res.success:
            s = res.fun
            if s < best_score:
                best_score = s
                best_half = cosine_half(res.x, order).copy()
    
    # Polynomial: optimize width and power
    for power in [1, 2, 3, 4]:
        if time.time() - start_time > max_time:
            break
        def obj_width(width):
            return symmetric_score(polynomial_half(width, power))
        res = minimize_scalar(obj_width, bounds=(0.05, 0.5), method='bounded',
                              options={'xatol': 1e-6, 'maxiter': 100})
        if res.success:
            s = res.fun
            if s < best_score:
                best_score = s
                best_half = polynomial_half(res.x, power).copy()
    
    return best_half, best_score

def coordinate_descent_symmetric(half_f, start_time, max_time):
    """
    Coordinate descent on the half function (300 bins).
    For each bin, find the optimal value given others fixed.
    """
    half = half_f.copy()
    half = np.maximum(half, 0.0)
    s = np.sum(half)
    if s < 1e-15:
        return half_f
    
    best_half = half.copy()
    best_val = symmetric_score(half)
    
    # Precompute the full function and its autoconvolution
    full = make_symmetric(half)
    conv = fft_autoconv(full)
    current_max = np.max(conv)
    
    for sweep in range(200):
        if time.time() - start_time > max_time:
            break
        
        improved = False
        for i in range(HALF_N):
            if time.time() - start_time > max_time:
                break
            
            old_val = half[i]
            if old_val < 1e-15:
                # Try small positive values
                candidates = [1e-4, 1e-3, 0.01]
            else:
                candidates = [0.0, old_val * 0.5, old_val * 0.8, old_val * 1.2,
                              old_val * 1.5, old_val * 2.0]
            
            best_v = old_val
            best_new_max = current_max
            
            for v in candidates:
                v = max(0.0, v)
                if abs(v - old_val) < 1e-15:
                    continue
                
                # Update the full function
                new_half = half.copy()
                new_half[i] = v
                new_full = make_symmetric(new_half)
                new_conv = fft_autoconv(new_full)
                new_max = np.max(new_conv)
                
                if new_max < best_new_max - 1e-14:
                    best_new_max = new_max
                    best_v = v
            
            if best_v != old_val:
                half[i] = best_v
                improved = True
                # Update full and conv
                full = make_symmetric(half)
                conv = fft_autoconv(full)
                current_max = np.max(conv)
        
        if not improved:
            break
        
        # Check if we improved the overall score
        s = np.sum(half)
        if s > 1e-15:
            score = symmetric_score(half)
            if score < best_val - 1e-14:
                best_val = score
                best_half = half.copy()
    
    return best_half

def local_refinement_symmetric(half_f, start_time, max_time):
    """
    More aggressive local refinement using Nelder-Mead on small subsets.
    """
    half = half_f.copy()
    best_val = symmetric_score(half)
    
    # Try perturbing small groups of adjacent bins
    group_sizes = [3, 5, 8]
    for gs in group_sizes:
        if time.time() - start_time > max_time:
            break
        
        for start in range(0, HALF_N - gs, gs):
            if time.time() - start_time > max_time:
                break
            
            # Get the current values of this group
            group = half[start:start+gs].copy()
            
            def obj(params):
                new_half = half.copy()
                new_half[start:start+gs] = np.maximum(params, 0)
                return symmetric_score(new_half)
            
            res = minimize(obj, group, method='Nelder-Mead',
                          options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.success and res.fun < best_val - 1e-14:
                half[start:start+gs] = np.maximum(res.x, 0)
                best_val = res.fun
    
    return half

def construct():
    start_time = time.time()
    
    # Step 1: Optimize over parametric families
    best_half, best_score = optimize_parametric(start_time, TIME_LIMIT - 30)
    
    if best_half is None:
        # Fallback: uniform
        best_half = np.ones(HALF_N)
        best_score = symmetric_score(best_half)
    
    print(f"Parametric best: {best_score:.10f}")
    
    # Step 2: Coordinate descent polish
    if time.time() - start_time < TIME_LIMIT - 15:
        polished = coordinate_descent_symmetric(best_half, start_time, TIME_LIMIT - 10)
        polished_score = symmetric_score(polished)
        if polished_score < best_score:
            best_half = polished
            best_score = polished_score
            print(f"CD polished: {best_score:.10f}")
    
    # Step 3: Local refinement
    if time.time() - start_time < TIME_LIMIT - 5:
        refined = local_refinement_symmetric(best_half, start_time, TIME_LIMIT - 3)
        refined_score = symmetric_score(refined)
        if refined_score < best_score:
            best_half = refined
            best_score = refined_score
            print(f"Refined: {best_score:.10f}")
    
    # Final: build the full symmetric function
    best_full = make_symmetric(best_half)
    best_full = np.maximum(best_full, 0.0)
    s = np.sum(best_full)
    if s > 1e-15:
        best_full = best_full / s
    
    final_score = compute_score(best_full)
    print(f"Final score: {final_score:.10f}")
    
    return [float(x) for x in best_full]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
