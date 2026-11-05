"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import fft, ifft
from scipy.optimize import minimize

def _eval_value(h):
    """Compute the objective value using FFT-based convolution."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = _convolve_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _convolve_fft(h):
    """Compute full autoconvolution using FFT."""
    n_conv = 2 * len(h) - 1
    n_fft = 1
    while n_fft < n_conv:
        n_fft *= 2
    H = fft(h, n=n_fft)
    conv = ifft(H * H, n=n_fft).real[:n_conv]
    return conv

def _soft_max(x, beta):
    """Numerically stable soft maximum."""
    m = np.max(x)
    if not np.isfinite(m):
        return m
    exp_vals = np.exp(beta * (x - m))
    sm = m + np.log(np.sum(exp_vals)) / beta
    return sm

def _soft_max_grad(x, beta):
    """Soft maximum and its gradient."""
    m = np.max(x)
    if not np.isfinite(m):
        return m, np.zeros_like(x)
    exp_vals = np.exp(beta * (x - m))
    w = exp_vals / np.sum(exp_vals)
    sm = m + np.log(np.sum(exp_vals)) / beta
    grad = beta * w
    return sm, grad

def _objective_soft(h, beta):
    """Soft-max surrogate objective with normalization."""
    if not np.all(np.isfinite(h)):
        return 1e10
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    h_norm = h * (N / s)
    conv = _convolve_fft(h_norm)
    sm = _soft_max(conv, beta)
    return abs(2.0 * N * sm / (N * N))

def _objective_grad(h, beta):
    """Objective and gradient using soft-max surrogate."""
    if not np.all(np.isfinite(h)):
        return 1e10, np.zeros_like(h)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(h)
    
    # Normalize: h_norm = h * N/s
    scale = N / s
    h_norm = h * scale
    conv = _convolve_fft(h_norm)
    sm, sm_grad = _soft_max_grad(conv, beta)
    
    # d(sm)/d(h) = scale * d(conv)/d(h) * sm_grad
    # conv[k] = sum_i h[i]*h[k-i], so d(conv[k])/d(h[j]) = h[k-j] + h[j-k] (for valid indices)
    # This is the cross-correlation of sm_grad with h_norm
    n_conv = len(conv)
    # Gradient of conv w.r.t. h: for each lag k, d(conv[k])/d(h[j]) = h_norm[k-j] if 0<=k-j<N else h_norm[j-k] if 0<=j-k<N else 0
    # Equivalently, grad_h[j] = sum_k sm_grad[k] * d(conv[k])/d(h[j])
    # = sum_k sm_grad[k] * (h_norm[k-j] + h_norm[j-k]) for valid indices
    # This is: conv(sm_grad, h_norm)[j] + conv(sm_grad, h_norm[::-1])[N-1-j] ... let me think more carefully
    
    # Actually: conv_full[k] = sum_{i=0}^{N-1} h[i] * h[k-i] where h is zero-padded
    # d(conv_full[k])/d(h[j]) = h[k-j] (if 0<=k-j<N) + h[j-k] (if 0<=j-k<N)
    # So grad_h[j] = sum_k sm_grad[k] * (h_norm[k-j] * 1_{0<=k-j<N} + h_norm[j-k] * 1_{0<=j-k<N})
    # = (sm_grad * h_norm padded) convolved appropriately
    
    # Use FFT: the gradient is the cross-correlation of sm_grad with h_norm
    # grad_h[j] = sum_k sm_grad[k] * h_norm[|k-j|] with appropriate signs... 
    # Actually simpler: grad_h = ifft(fft(sm_grad) * fft(h_norm)) type operation
    # Let me just do it directly with np.convolve for clarity (N=400 is small)
    
    # d(conv[k])/d(h[j]) for full convolution:
    # conv[k] = sum_{i=max(0,k-N+1)}^{min(k,N-1)} h[i] * h[k-i]
    # d(conv[k])/d(h[j]) = h[k-j] if 0<=k-j<N else 0, PLUS h[j-k] if 0<=j-k<N else 0
    # But when j == k-j (i.e., k=2j), we count it once, not twice.
    # Actually for full conv of h with h: conv[k] = sum_i h[i]*h[k-i]
    # d(conv[k])/d(h[j]) = h[k-j] (if 0<=k-j<N) + h[j-k] (if 0<=j-k<N)
    # If k-j == j-k (k=2j), then h[k-j] = h[j-k], so it's 2*h[j] unless we're careful
    # But in the sum, when i=j, we get h[j]*h[k-j], and when i=k-j, we get h[k-j]*h[j].
    # If j != k-j, these are two separate terms. If j == k-j, it's one term h[j]^2.
    # So d(conv[k])/d(h[j]) = h[k-j] + h[j-k] - h[j] * 1_{k=2j}
    
    # For simplicity, let's compute gradient numerically-free using the fact that:
    # The gradient of sum_k g[k]*conv[k] w.r.t. h[j] is:
    # = sum_k g[k] * (h[k-j] * 1_{0<=k-j<N} + h[j-k] * 1_{0<=j-k<N} - h[j]*1_{k=2j})
    # The first two terms together form the circular-ish convolution of g with h
    # Let's just compute it directly since N=400
    
    grad_h = np.zeros(N)
    for j in range(N):
        # Sum over k where 0 <= k-j < N: g[k] * h[k-j]
        k_lo = j
        k_hi = j + N - 1
        k_lo = max(k_lo, 0)
        k_hi = min(k_hi, n_conv - 1)
        if k_hi >= k_lo:
            idx = np.arange(k_lo, k_hi + 1)
            grad_h[j] += np.sum(sm_grad[idx] * h_norm[idx - j])
        
        # Sum over k where 0 <= j-k < N: g[k] * h[j-k]
        k_lo2 = j - (N - 1)
        k_hi2 = j
        k_lo2 = max(k_lo2, 0)
        k_hi2 = min(k_hi2, n_conv - 1)
        if k_hi2 >= k_lo2:
            idx2 = np.arange(k_lo2, k_hi2 + 1)
            grad_h[j] += np.sum(sm_grad[idx2] * h_norm[j - idx2])
        
        # Subtract double-counting at k=2j
        k_diag = 2 * j
        if 0 <= k_diag < n_conv:
            grad_h[j] -= sm_grad[k_diag] * h_norm[j]
    
    # Chain rule: d(obj)/d(h[j]) = d(obj)/d(h_norm[j]) * d(h_norm[j])/d(h[j])
    # h_norm[j] = h[j] * N/s, so d(h_norm[j])/d(h[j]) = N/s = scale
    # Also need to account for the abs() and the 1/(s^2) factor... 
    # Actually, let's simplify: work with the normalized objective directly
    # obj = |2N * sm / N^2| = |2*sm/N| where sm is soft max of conv of h_norm
    # d(obj)/d(h[j]) = sign(2*sm/N) * 2/N * d(sm)/d(h[j])
    # And d(sm)/d(h[j]) = scale * grad_h[j] (from above, since h_norm = h*scale)
    # But we also need to account for the normalization s in the chain rule for the full objective
    # Let me just return the gradient of the soft-max objective w.r.t. h_norm, and handle scaling
    
    # Actually, let's keep it simple: return objective and gradient w.r.t. h
    # The objective is abs(2N*sm/(s^2)) where sm = soft_max(conv(h))
    # For now, let's just use the unnormalized version and let the optimizer handle it
    # Better: define the objective as 2N*sm/(s^2) without abs, assuming sm>0 (which it is for reasonable h)
    
    # Let me recompute more carefully. The value we want to minimize is:
    # V = |2N * max(conv(h)) / s^2|
    # Using soft max: V_soft = |2N * sm / s^2|
    # dV_soft/dh[j] = sign(2N*sm/s^2) * 2N/s^2 * d(sm)/dh[j] - 2*2N*sm/s^3 * ds/dh[j]
    # where ds/dh[j] = 1
    # d(sm)/dh[j] = scale * grad_h[j] (where scale = N/s)
    # So dV_soft/dh[j] = sign(...) * 2N/s^2 * (N/s) * grad_h[j] - 4N*sm/s^3
    # = sign(...) * 2N^2/s^3 * grad_h[j] - 4N*sm/s^3
    
    # For simplicity, let's just minimize the soft-max of the normalized convolution
    # and use a simpler gradient
    
    obj_val = abs(2.0 * N * sm / (s * s))
    
    # Gradient computation (simplified - use the scale factor)
    # dV/dh[j] ≈ (2N/s^2) * scale * grad_h[j] - (4N*sm/s^3) * 1
    # Let's just use the first term as an approximation for the gradient direction
    grad_out = (2.0 * N / (s * s)) * scale * grad_h - (4.0 * N * sm / (s ** 3))
    
    return obj_val, grad_out

def _coordinate_descent(h, beta, max_iters=50, step_size=0.01):
    """Coordinate descent on individual elements."""
    best_h = h.copy()
    best_val = _objective_soft(best_h, beta)
    
    for it in range(max_iters):
        improved = False
        for j in range(N):
            # Try small perturbations
            for delta in [-step_size, step_size, -step_size*0.5, step_size*0.5]:
                h_trial = best_h.copy()
                h_trial[j] += delta
                # Renormalize to keep sum constant
                s = np.sum(h_trial)
                if abs(s) < 1e-10:
                    continue
                h_trial = h_trial * (N / s)
                v = _objective_soft(h_trial, beta)
                if v < best_val - 1e-12:
                    best_h = h_trial
                    best_val = v
                    improved = True
                    break
        if not improved:
            break
    return best_h, best_val

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Multi-phase optimization: parametric families, random walks, 
    gradient-based DCT optimization, and coordinate descent polish.
    """
    deadline = time.time() + 145.0  # Safety margin before 150s limit

    # Start with all-ones (value 2.0)
    best_h = np.ones(N)
    best_val = 2.0

    # Phase 1: Quick parametric families
    # Family 1: h[i] = 1 + a*cos(2*pi*m*i/N)
    for m in range(1, 31):
        if time.time() > deadline - 90:
            break
        for a in np.linspace(-0.9, 0.9, 19):
            h = 1.0 + a * np.cos(2.0 * np.pi * m * np.arange(N) / N)
            v = _eval_value(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()

    # Family 2: Smoothed block (tanh)
    for beta in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]:
        if time.time() > deadline - 90:
            break
        h = np.tanh(beta * (np.arange(N) - N / 2.0))
        s = np.sum(h)
        if abs(s) < 1e-6:
            continue
        v = _eval_value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()

    # Family 3: Two-level step function
    for frac in np.linspace(0.1, 0.9, 17):
        if time.time() > deadline - 90:
            break
        h = np.ones(N)
        n_high = int(frac * N)
        h[:n_high] = 1.0 + 0.5
        h[n_high:] = 1.0 - 0.5
        # Make zero-mean
        h = h - np.mean(h) + 1.0
        v = _eval_value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()

    # Family 4: Alternating patterns
    for period in [2, 3, 4, 5, 7, 8, 10]:
        if time.time() > deadline - 90:
            break
        for amp in [0.3, 0.5, 0.7]:
            h = 1.0 + amp * np.sin(2.0 * np.pi * np.arange(N) / period)
            v = _eval_value(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()

    # Phase 2: Random walk with zero-sum constraint
    if time.time() < deadline - 70:
        rng = np.random.RandomState(42)
        
        for noise_scale in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]:
            if time.time() > deadline - 70:
                break
            
            for trial in range(15):
                if time.time() > deadline - 70:
                    break
                
                noise = rng.randn(N) * noise_scale
                walk = np.cumsum(noise)
                
                # Subtract linear trend for zero-sum
                sum_i = np.sum(np.arange(N))
                drift = np.sum(walk) / sum_i
                h = 1.0 + walk - drift * np.arange(N)
                
                if not np.all(np.isfinite(h)) or abs(np.sum(h) - N) > 1e-6:
                    continue
                
                v = _eval_value(h)
                if v < best_val:
                    best_val = v
                    best_h = h.copy()

    # Phase 3: Gradient-based optimization in DCT domain
    from scipy.fft import dct, idct
    
    def _h_to_dct(h):
        return dct(h, type=2, norm='ortho')
    
    def _dct_to_h(coeffs):
        return idct(coeffs, type=2, norm='ortho')
    
    c_best = _h_to_dct(best_h)
    c_dc = c_best[0].copy()

    # Multi-stage soft-max optimization
    for beta in [2.0, 10.0, 50.0, 200.0, 500.0]:
        if time.time() > deadline - 25:
            break

        def obj_dct(c_ac, beta=beta):
            c_full = np.concatenate([[c_dc], c_ac])
            h_recon = _dct_to_h(c_full)
            if not np.all(np.isfinite(h_recon)):
                return 1e10
            s = np.sum(h_recon)
            if abs(s) < 1e-10:
                return 1e10
            h_norm = h_recon * (N / s)
            conv = _convolve_fft(h_norm)
            sm = _soft_max(conv, beta)
            return abs(2.0 * N * sm / (N * N))

        rng = np.random.RandomState(42 + int(beta))
        c_ac_init = c_best[1:].copy()

        for trial in range(3):
            if time.time() > deadline - 25:
                break

            if trial == 0:
                c_ac_trial = c_ac_init.copy()
            else:
                c_ac_trial = c_ac_init + 0.05 * rng.randn(N-1)

            try:
                result = minimize(
                    obj_dct,
                    c_ac_trial,
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-13, 'gtol': 1e-11}
                )
                c_ac_opt = result.x
                c_full_opt = np.concatenate([[c_dc], c_ac_opt])
                h_opt = _dct_to_h(c_full_opt)

                if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                    v_opt = _eval_value(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
                        c_best = _h_to_dct(best_h)
                        c_dc = c_best[0].copy()
                        c_ac_init = c_best[1:].copy()
            except Exception:
                continue

    # Phase 4: Coordinate descent polish
    if time.time() < deadline - 15:
        for beta_cd in [50.0, 200.0, 500.0]:
            if time.time() > deadline - 10:
                break
            h_cd, v_cd = _coordinate_descent(best_h, beta_cd, max_iters=30, step_size=0.02)
            if v_cd < best_val:
                best_val = v_cd
                best_h = h_cd

    # Phase 5: Final DCT polish with high beta
    if time.time() < deadline - 5:
        beta_final = 1000.0
        def obj_final(c_ac, beta=beta_final):
            c_full = np.concatenate([[c_dc], c_ac])
            h_recon = _dct_to_h(c_full)
            if not np.all(np.isfinite(h_recon)):
                return 1e10
            s = np.sum(h_recon)
            if abs(s) < 1e-10:
                return 1e10
            h_norm = h_recon * (N / s)
            conv = _convolve_fft(h_norm)
            sm = _soft_max(conv, beta)
            return abs(2.0 * N * sm / (N * N))
        
        try:
            result = minimize(
                obj_final,
                c_best[1:].copy(),
                method='L-BFGS-B',
                options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            c_ac_opt = result.x
            c_full_opt = np.concatenate([[c_dc], c_ac_opt])
            h_opt = _dct_to_h(c_full_opt)
            if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                v_opt = _eval_value(h_opt)
                if v_opt < best_val:
                    best_val = v_opt
                    best_h = h_opt.copy()
        except Exception:
            pass

    # Ensure finite and sum != 0
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)

    return best_h
# EVOLVE-BLOCK-END
