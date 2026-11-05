"""DCT-spectral shaping with L-BFGS-B for min max autoconvolution."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.fftpack import dct, idct

N = 400
DEADLINE = 115.0  # safety margin before 120s limit

def _value(h):
    """Compute the objective value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e18
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _objective(coeffs, K):
    """Objective: max autoconvolution peak (before normalization).
    We minimize max(conv) / (sum(h))^2 * 2*N, but since 2*N is constant,
    we minimize max(conv) / (sum(h))^2.
    Actually, let's just minimize the full value directly for simplicity,
    but that's expensive. Instead, minimize max(conv) subject to sum(h)=1.
    Then value = 2*N*max(conv).
    """
    h = _coeffs_to_h(coeffs, K)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e18
    conv = np.convolve(h, h)
    return 2.0 * N * np.max(conv) / (s * s)

def _coeffs_to_h(coeffs, K):
    """Convert DCT coefficients to time-domain heights."""
    # coeffs[0] is the DC component, coeffs[1:K] are AC
    # We need to reconstruct using DCT-II inverse
    full_coeffs = np.zeros(N)
    full_coeffs[:K] = coeffs
    h = idct(full_coeffs, type=2, norm='ortho')
    return h

def _grad_objective(coeffs, K):
    """Gradient of the objective with respect to DCT coefficients."""
    h = _coeffs_to_h(coeffs, K)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.zeros_like(coeffs)
    
    conv = np.convolve(h, h)
    max_val = np.max(conv)
    
    # Gradient of max(conv) w.r.t. h
    # conv[k] = sum_i h[i]*h[k-i] for valid indices
    # d(conv[k])/d(h[j]) = h[k-j] + h[j-k] (with appropriate bounds)
    # For full convolution of length 2N-1:
    # conv[m] = sum_{i} h[i]*h[m-i] where 0<=i<N and 0<=m-i<N
    # d(conv[m])/d(h[j]) = h[m-j] (if 0<=m-j<N) + h[j-m] (if 0<=j-m<N)
    
    # Find index of max
    m = np.argmax(conv)
    
    # Gradient of conv[m] w.r.t. h
    grad_conv_m = np.zeros(N)
    # h[m-j] for j in range where 0<=m-j<N => j in [max(0,m-N+1), min(m, N-1)]
    j_start = max(0, m - N + 1)
    j_end = min(m, N - 1)
    if j_start <= j_end:
        grad_conv_m[j_start:j_end+1] = h[m - np.arange(j_start, j_end+1)]
    
    # h[j-m] for j in range where 0<=j-m<N => j in [max(0,m), min(m+N-1, N-1)]
    j_start2 = max(0, m)
    j_end2 = min(m + N - 1, N - 1)
    if j_start2 <= j_end2:
        idx = np.arange(j_start2, j_end2+1)
        grad_conv_m[idx] += h[idx - m]
    
    # Objective f = 2*N*max(conv)/(s^2)
    # df/d(h[j]) = 2*N * [grad_conv_m[j]*s^2 - max(conv)*2*s*1] / s^4
    #            = 2*N * [grad_conv_m[j]/s - 2*max(conv)/(s*s)] ... wait
    # f = 2*N * max_conv / s^2
    # df/dh_j = 2*N * (grad_conv_m[j] * s^2 - max_conv * 2*s * dh_sum/dh_j) / s^4
    # dh_sum/dh_j = 1
    # df/dh_j = 2*N * (grad_conv_m[j] * s^2 - 2*max_conv*s) / s^4
    #          = 2*N * (grad_conv_m[j] * s - 2*max_conv) / s^3
    
    grad_h = 2.0 * N * (grad_conv_m * s - 2.0 * max_val) / (s**3)
    
    # Now map gradient from h-space to coeffs-space
    # h = idct(coeffs_full, type=2, norm='ortho')
    # coeffs_full = dct(h, type=2, norm='ortho')
    # So grad_coeffs = dct(grad_h, type=2, norm='ortho')
    grad_coeffs_full = dct(grad_h, type=2, norm='ortho')
    return grad_coeffs_full[:K]

def _try_K(K, budget):
    """Try to optimize with K DCT coefficients within budget time."""
    start_time = time.time()
    
    # Initial guess: start with a simple pattern
    # Option 1: all zeros except DC
    # Option 2: random
    # Option 3: sinusoidal-like
    
    best_val = 1e18
    best_coeffs = None
    
    # Try a few initializations
    inits = []
    
    # Init 1: DC only (gives uniform h)
    c1 = np.zeros(K)
    c1[0] = 1.0
    inits.append(c1)
    
    # Init 2: random
    rng = np.random.default_rng(42)
    c2 = rng.standard_normal(K)
    c2[0] = 0.0  # keep DC at 0 to allow negative values
    inits.append(c2)
    
    # Init 3: first few DCT modes with alternating signs
    c3 = np.zeros(K)
    for i in range(1, min(K, 10)):
        c3[i] = (-1)**i * 0.1
    inits.append(c3)
    
    # Init 4: Zadoff-Chu inspired (constant amplitude in time domain)
    # For DCT, a constant-amplitude sequence in time has specific DCT coefficients
    # Let's just use a chirp in DCT domain
    c4 = np.zeros(K)
    for i in range(1, K):
        c4[i] = 0.05 * np.sin(2*np.pi*i/K)
    inits.append(c4)
    
    for init in inits:
        if time.time() - start_time > budget:
            break
        
        try:
            res = minimize(
                _objective,
                init,
                args=(K,),
                jac=_grad_objective,
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if time.time() - start_time > budget:
                break
            
            if res.success or res.fun < best_val:
                val = _value(_coeffs_to_h(res.x, K))
                if val < best_val:
                    best_val = val
                    best_coeffs = res.x.copy()
        except Exception:
            continue
    
    return best_val, best_coeffs

def construct():
    start_time = time.time()
    
    best_val = 2.0  # seed value
    best_h = np.ones(N)
    
    # Try increasing K values
    K_list = [5, 10, 20, 50, 100]
    
    for K in K_list:
        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 5:
            break
        
        budget = min(remaining * 0.6, 20.0)  # cap per K
        val, coeffs = _try_K(K, budget)
        
        if coeffs is not None and val < best_val:
            best_val = val
            h = _coeffs_to_h(coeffs, K)
            # Ensure sum is not zero
            if abs(np.sum(h)) > 1e-12:
                best_h = h
            else:
                # Add small DC
                coeffs2 = coeffs.copy()
                coeffs2[0] = 0.1
                h = _coeffs_to_h(coeffs2, K)
                if abs(np.sum(h)) > 1e-12:
                    best_h = h
        
        # Check deadline
        if time.time() - start_time > DEADLINE - 5:
            break
    
    # Final safety check
    if time.time() - start_time > DEADLINE - 2:
        return best_h
    
    # One more refinement on best found
    if best_val < 2.0:
        # Find which K was best
        for K in K_list:
            h_test = _coeffs_to_h(best_h, N)  # just use best_h directly
            break
        
        # Do a final L-BFGS-B on the best h converted to DCT
        h_start = best_h.copy()
        coeffs_start = dct(h_start, type=2, norm='ortho')
        K_final = min(200, N)
        
        try:
            res = minimize(
                _objective,
                coeffs_start[:K_final],
                args=(K_final,),
                jac=_grad_objective,
                method='L-BFGS-B',
                options={'maxiter': 300, 'gtol': 1e-10, 'maxls': 50}
            )
            if time.time() - start_time < DEADLINE - 2:
                val = _value(_coeffs_to_h(res.x, K_final))
                if val < best_val:
                    best_val = val
                    best_h = _coeffs_to_h(res.x, K_final)
        except Exception:
            pass
    
    # Ensure validity
    if not np.all(np.isfinite(best_h)):
        return np.ones(N)
    if abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
    
    return best_h
