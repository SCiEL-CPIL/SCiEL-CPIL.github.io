import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve # or manual FFT

N = 600
DEADLINE = 115

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)

    def compute_autoconv(h):
        # h is length N
        # Linear convolution
        # Use FFT for speed
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        return conv

    def smooth_max_obj_and_grad(coeffs, tau=1.0):
        # coeffs: array of cosine coefficients a_0 ... a_M
        # h_k = sum a_m cos(2 pi m k / N)
        # Note: standard DCT/FFT relation
        # If we use rfft, the coefficients are complex.
        # Here we use real cosine basis.
        # h_k = a_0 + sum_{m=1}^M a_m cos(2 pi m k / N)
        
        k = np.arange(N)
        h = np.zeros(N)
        for m in range(len(coeffs)):
            if m == 0:
                h += coeffs[0]
            else:
                h += coeffs[m] * np.cos(2 * np.pi * m * k / N)
        
        # Enforce non-negativity? 
        # If we optimize in coeff space, h might go negative.
        # We can add a penalty or project.
        # Projecting h to non-negative and then re-extracting coeffs is messy.
        # Better: use a parameterization that guarantees non-negativity?
        # e.g. h_k = exp(y_k) ? But then we lose the low-freq structure.
        # Or h_k = (sum a_m cos(...))^2 ? No, too high order.
        
        # Let's just clip h to 0 for evaluation, but gradient will be wrong at boundaries.
        # Or, use L-BFGS-B with bounds on coeffs? No, bounds on coeffs don't guarantee h>=0.
        
        # Alternative: Optimize h directly but use a smooth basis to restrict search space?
        # The inspiration used:
        # 1. coeffs_to_h
        # 2. h = max(h, 0)
        # 3. h = h / sum(h)
        # 4. Compute obj
        # 5. Gradient: computed via chain rule assuming h is smooth.
        # This is an approximation. It works well if h stays positive.
        
        h_clipped = np.maximum(h, 0)
        s = np.sum(h_clipped)
        if s < 1e-10:
            return 1e10, np.zeros(len(coeffs))
        h_norm = h_clipped / s
        
        c = compute_autoconv(h_norm)
        c_max = np.max(c)
        exp_c = np.exp((c - c_max) / tau)
        sum_exp = np.sum(exp_c)
        obj = tau * (np.log(sum_exp) + c_max / tau)
        
        # Gradient
        # w_k = exp_c[k] / sum_exp
        w = exp_c / sum_exp
        
        # G_h = 2 * correlation(w, h_norm)
        # corr[w, h]_l = sum_k w_k h_{k-l}
        # We need G_h[l] for l in 0..N-1.
        
        # Use FFT for correlation
        # corr = ifft(fft(w) * conj(fft(h)))
        # w is length 2N-1 (from conv). h is length N.
        # We need to align them.
        # The autoconv c is length 2N-1.
        # w is defined on indices 0..2N-2.
        # h is defined on 0..N-1.
        # G_h[l] = sum_{k} w_k h_{k-l}.
        # Valid k such that 0 <= k-l < N => l <= k < l+N.
        
        # Let's compute G_h via FFT.
        # Pad w to size L (e.g. 4N).
        # Pad h to size L.
        # G_circ = ifft(fft(w) * conj(fft(h)))
        # This gives circular correlation.
        # If L is large enough (L >= len(w) + len(h) - 1), the first len(h) elements correspond to the linear correlation?
        # Actually, correlation of w (len M) and h (len N) has length M+N-1.
        # We want the part corresponding to shifts l=0..N-1.
        
        L = 4 * N
        w_pad = np.zeros(L)
        w_pad[:len(w)] = w
        h_pad = np.zeros(L)
        h_pad[:N] = h_norm
        
        W = np.fft.fft(w_pad, n=L)
        H = np.fft.fft(h_pad, n=L)
        
        # Correlation: sum w_k h_{k-l}
        # This is ifft( W * conj(H) ) ?
        # Let's check.
        # Convolution: sum w_k h_{k-l} is cross-correlation.
        # Correlation is Convolution with reversed signal.
        # corr[w, h] = conv[w, flip(h)]
        # In FFT: fft(w) * fft(flip(h)) = fft(w) * conj(fft(h)) (if h is real).
        
        corr_full = np.fft.ifft(W * np.conj(H)).real
        
        # The linear correlation is embedded in corr_full.
        # We need indices corresponding to l=0..N-1.
        # In the circular correlation of size L, the result for shift l is at index l.
        # So G_h = corr_full[0:N] * 2.0
        
        G_h = 2.0 * corr_full[0:N]
        
        # Gradient w.r.t coeffs
        # grad_c[m] = sum_l G_h[l] * dh_l/da_m
        # dh_l/da_m = cos(2 pi m l / N) for m>0, 1 for m=0.
        
        grad_c = np.zeros(len(coeffs))
        for m in range(len(coeffs)):
            if m == 0:
                grad_c[m] = np.sum(G_h)
            else:
                grad_c[m] = np.dot(G_h, np.cos(2 * np.pi * m * k / N))
                
        return obj, grad_c

    # Initialization
    k = np.arange(N)
    inits = []
    
    # Uniform
    inits.append(np.ones(N))
    
    # Bimodal
    for sigma in [N/10, N/8, N/6]:
        for alpha in [0.1, 0.2, 0.3]:
            pos1 = alpha * N
            pos2 = (1 - alpha) * N
            g1 = np.exp(-((k - pos1) ** 2) / (2 * sigma ** 2))
            g2 = np.exp(-((k - pos2) ** 2) / (2 * sigma ** 2))
            inits.append(g1 + g2)
            
    # W-shape
    w_shape = 1.0 + 1.5 * np.cos(4 * np.pi * k / N)
    inits.append(np.maximum(w_shape, 0.01))
    
    # Convert h to coeffs
    M = 12 # Number of cosine terms
    def h_to_coeffs(h):
        # Project onto cosine basis
        # a_m = <h, cos_m> / <cos_m, cos_m>
        # <cos_m, cos_m> = N/2 for m>0, N for m=0
        c = np.zeros(M+1)
        c[0] = np.mean(h)
        for m in range(1, M+1):
            c[m] = 2.0 * np.dot(h, np.cos(2 * np.pi * m * k / N)) / N
        return c

    def coeffs_to_h(c):
        h = np.zeros(N)
        for m in range(len(c)):
            if m == 0:
                h += c[0]
            else:
                h += c[m] * np.cos(2 * np.pi * m * k / N)
        return h

    best_h = None
    best_val = np.inf
    
    # Evaluate inits
    for h_init in inits:
        if time_left() < 5: break
        h_norm = h_init / np.sum(h_init)
        c = compute_autoconv(h_norm)
        val = np.max(c)
        if val < best_val:
            best_val = val
            best_h = h_norm.copy()
            
    if best_h is None:
        best_h = np.ones(N) / N
        best_val = np.max(compute_autoconv(best_h))

    # Optimize each init in coeff space
    # We will run optimization for the top few inits
    
    # Sort inits by val
    # ... (logic to pick best inits)
    
    # For simplicity, just optimize the best_h found so far, and a few others.
    # Let's optimize the best one and a couple of diverse ones.
    
    candidates = []
    # Add best_h
    candidates.append(best_h)
    # Add a bimodal and a w-shape
    # (Re-generate or pick from list)
    
    for h_cand in candidates[:3]:
        if time_left() < 10: break
        
        c_init = h_to_coeffs(h_cand)
        
        def obj_func(c):
            val, grad = smooth_max_obj_and_grad(c, tau=1.0)
            return val
        
        def grad_func(c):
            val, grad = smooth_max_obj_and_grad(c, tau=1.0)
            return grad
            
        try:
            res = minimize(obj_func, c_init, jac=grad_func, method='L-BFGS-B', 
                           options={'maxiter': 50, 'maxfun': 100})
            
            h_res = coeffs_to_h(res.x)
            h_res = np.maximum(h_res, 0)
            h_res = h_res / np.sum(h_res)
            
            c_res = compute_autoconv(h_res)
            val_res = np.max(c_res)
            
            if val_res < best_val:
                best_val = val_res
                best_h = h_res.copy()
                
        except Exception as e:
            pass

    return best_h.tolist()

construct()
