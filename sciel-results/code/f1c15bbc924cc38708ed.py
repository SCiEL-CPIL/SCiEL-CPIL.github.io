"""DCT-spectral shaping with smooth-max L-BFGS-B for min max autoconvolution."""
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

def _coeffs_to_h(coeffs, K):
    """Convert DCT coefficients to time-domain heights."""
    full_coeffs = np.zeros(N)
    full_coeffs[:K] = coeffs
    h = idct(full_coeffs, type=2, norm='ortho')
    return h

def _smooth_max(conv, beta=50.0):
    """Smooth maximum using log-sum-exp."""
    max_val = np.max(conv)
    shifted = conv - max_val
    log_sum_exp = max_val + np.log(np.sum(np.exp(beta * shifted))) / beta
    return log_sum_exp

def _smooth_objective(coeffs, K, beta=50.0):
    """Smooth objective using log-sum-exp approximation of max."""
    h = _coeffs_to_h(coeffs, K)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e18
    conv = np.convolve(h, h)
    smax = _smooth_max(conv, beta)
    return 2.0 * N * smax / (s * s)

def _smooth_grad_objective(coeffs, K, beta=50.0):
    """Gradient of the smooth objective with respect to DCT coefficients."""
    h = _coeffs_to_h(coeffs, K)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.zeros_like(coeffs)
    
    conv = np.convolve(h, h)
    max_val = np.max(conv)
    shifted = conv - max_val
    weights = np.exp(beta * shifted)
    weights /= np.sum(weights)
    smax = max_val + np.log(np.sum(weights)) / beta
    
    # Gradient of smooth max w.r.t. conv
    grad_smax_conv = beta * weights
    
    # Gradient of conv w.r.t. h: for each m, grad_smax_h[m] = sum_k grad_smax_conv[k] * d(conv[k])/d(h[m])
    # d(conv[k])/d(h[j]) = h[k-j] (if 0<=k-j<N) + h[j-k] (if 0<=j-k<N)
    grad_smax_h = np.zeros(N)
    
    # Vectorized computation
    for k in range(len(conv)):
        wk = grad_smax_conv[k]
        if wk < 1e-15:
            continue
        # Contribution from h[k-j] term: j in [max(0,k-N+1), min(k, N-1)]
        j_start = max(0, k - N + 1)
        j_end = min(k, N - 1)
        if j_start <= j_end:
            idx = np.arange(j_start, j_end + 1)
            grad_smax_h[idx] += wk * h[k - idx]
        # Contribution from h[j-k] term: j in [max(0,k), min(k+N-1, N-1)]
        j_start2 = max(0, k)
        j_end2 = min(k + N - 1, N - 1)
        if j_start2 <= j_end2:
            idx = np.arange(j_start2, j_end2 + 1)
            grad_smax_h[idx] += wk * h[idx - k]
    
    # f = 2*N*smax/s^2
    # df/dh_j = 2*N*(grad_smax_h[j]*s^2 - smax*2*s) / s^4 = 2*N*(grad_smax_h[j]*s - 2*smax) / s^3
    grad_h = 2.0 * N * (grad_smax_h * s - 2.0 * smax) / (s**3)
    
    # Map to DCT coefficient space
    grad_coeffs_full = dct(grad_h, type=2, norm='ortho')
    return grad_coeffs_full[:K]

def _try_K(K, budget, beta=50.0):
    """Try to optimize with K DCT coefficients within budget time."""
    start_time = time.time()
    best_val = 1e18
    best_coeffs = None
    
    # Try a few initializations
    inits = []
    
    # Init 1: DC only
    c1 = np.zeros(K)
    c1[0] = 1.0
    inits.append(c1)
    
    # Init 2: random with zero DC
    rng = np.random.default_rng(42)
    c2 = rng.standard_normal(K)
    c2[0] = 0.0
    inits.append(c2)
    
    # Init 3: alternating low-frequency modes
    c3 = np.zeros(K)
    for i in range(1, min(K, 15)):
        c3[i] = (-1)**i * 0.15
    inits.append(c3)
    
    # Init 4: chirp-like in DCT domain
    c4 = np.zeros(K)
    for i in range(1, K):
        c4[i] = 0.08 * np.sin(2*np.pi*i*1.5/K)
    inits.append(c4)
    
    # Init 5: Golay-inspired (try to make autocorrelation flat)
    c5 = np.zeros(K)
    c5[0] = 0.5
    for i in range(1, min(K, 20)):
        c5[i] = 0.05 * np.cos(2*np.pi*i*0.7/K)
    inits.append(c5)
    
    for init in inits:
        if time.time() - start_time > budget:
            break
        
        try:
            res = minimize(
                _smooth_objective,
                init,
                args=(K, beta),
                jac=_smooth_grad_objective,
                method='L-BFGS-B',
                options={'maxiter': 800, 'gtol': 1e-9, 'maxls': 80, 'ftol': 1e-12}
            )
            
            if time.time() - start_time > budget:
                break
            
            h_test = _coeffs_to_h(res.x, K)
            if abs(np.sum(h_test)) > 1e-12:
                val = _value(h_test)
                if val < best_val:
                    best_val = val
                    best_coeffs = res.x.copy()
        except Exception:
            continue
    
    return best_val, best_coeffs

def construct():
    start_time = time.time()
    
    best_val = 2.0
    best_h = np.ones(N)
    
    # Try increasing K values with smooth objective
    K_list = [5, 10, 20, 40, 80, 120]
    
    for K in K_list:
        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 8:
            break
        
        # Allocate more budget for larger K since they're more expressive
        budget = min(remaining * 0.5, 25.0)
        val, coeffs = _try_K(K, budget, beta=50.0)
        
        if coeffs is not None and val < best_val:
            best_val = val
            h = _coeffs_to_h(coeffs, K)
            if abs(np.sum(h)) > 1e-12:
                best_h = h
            else:
                coeffs2 = coeffs.copy()
                coeffs2[0] = 0.1
                h = _coeffs_to_h(coeffs2, K)
                if abs(np.sum(h)) > 1e-12:
                    best_h = h
        
        if time.time() - start_time > DEADLINE - 10:
            break
    
    # Final safety check
    if time.time() - start_time > DEADLINE - 3:
        return best_h
    
    # Polish pass: use higher beta (closer to true max) and tighter tolerance
    if best_val < 2.0:
        h_start = best_h.copy()
        coeffs_start = dct(h_start, type=2, norm='ortho')
        K_final = min(150, N)
        
        try:
            # Use higher beta for polish (closer to exact max)
            res = minimize(
                _smooth_objective,
                coeffs_start[:K_final],
                args=(K_final, 200.0),
                jac=_smooth_grad_objective,
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-11, 'maxls': 100, 'ftol': 1e-14}
            )
            if time.time() - start_time < DEADLINE - 2:
                val = _value(_coeffs_to_h(res.x, K_final))
                if val < best_val:
                    best_val = val
                    best_h = _coeffs_to_h(res.x, K_final)
        except Exception:
            pass
        
        # Second polish with even tighter settings if time allows
        if time.time() - start_time < DEADLINE - 5 and best_val < 1.5:
            try:
                h_start2 = best_h.copy()
                coeffs_start2 = dct(h_start2, type=2, norm='ortho')
                K_final2 = min(200, N)
                
                res2 = minimize(
                    _smooth_objective,
                    coeffs_start2[:K_final2],
                    args=(K_final2, 500.0),
                    jac=_smooth_grad_objective,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'gtol': 1e-12, 'maxls': 100, 'ftol': 1e-15}
                )
                if time.time() - start_time < DEADLINE - 1:
                    val2 = _value(_coeffs_to_h(res2.x, K_final2))
                    if val2 < best_val:
                        best_val = val2
                        best_h = _coeffs_to_h(res2.x, K_final2)
            except Exception:
                pass
    
    # Ensure validity
    if not np.all(np.isfinite(best_h)):
        return np.ones(N)
    if abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
    
    return best_h
