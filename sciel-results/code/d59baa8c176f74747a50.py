import time
import numpy as np
from scipy.optimize import minimize

N = 600
DEADLINE = 140.0  # Safety margin before 150s kill
M_FFT = 2048  # Next power of 2 >= 2*600-1 = 1199

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution for a step function."""
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return np.inf
    heights = np.maximum(heights, 0.0)
    
    n = len(heights)
    w = 0.5 / n
    
    f = np.zeros(M_FFT, dtype=np.float64)
    f[:n] = heights
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    max_conv = np.max(conv)
    total = np.sum(heights)
    
    if total <= 0:
        return np.inf
    
    score = max_conv * n / (total * total * w)
    return score

def _smooth_max_surrogate(conv, beta):
    """Smooth approximation of max(conv) using log-sum-exp."""
    m = np.max(conv)
    shifted = conv - m
    exp_shifted = np.exp(shifted / beta)
    sum_exp = np.sum(exp_shifted)
    
    surrogate = m + beta * np.log(sum_exp)
    grad_conv = exp_shifted / sum_exp
    
    return surrogate, grad_conv

def _obj_grad(x, beta):
    """
    Objective and gradient in log-domain: h = exp(x).
    Minimize smooth-max surrogate of autoconvolution / (sum h)^2.
    """
    h = np.exp(x)
    n = len(h)
    total = np.sum(h)
    
    if total <= 0:
        return np.inf, np.zeros_like(x)
    
    w = 0.5 / n
    
    f = np.zeros(M_FFT, dtype=np.float64)
    f[:n] = h
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    surrogate, grad_conv = _smooth_max_surrogate(conv, beta)
    
    score = surrogate * n / (total * total * w)
    
    # Gradient computation
    # d(score)/d(h) = n/(total^2 * w) * (d_surr_dh - 2*surrogate*h/total)
    
    # d_surr_dh = term1 + term2
    # term1[i] = sum_j h[j] * grad_conv[j+i]
    # term2[i] = sum_j h[j] * grad_conv[i-j] (for i>=j)
    
    i_vals = np.arange(n)
    j_vals = np.arange(n)
    
    idx1_matrix = j_vals[None, :] + i_vals[:, None]
    valid1_matrix = idx1_matrix < (2*n - 1)
    
    grad_conv_padded = np.zeros(2*n)
    grad_conv_padded[:2*n-1] = grad_conv
    
    P = grad_conv_padded[idx1_matrix]
    P[~valid1_matrix] = 0
    term1 = P @ h
    
    idx2_matrix = i_vals[:, None] - j_vals[None, :]
    valid2_matrix = idx2_matrix >= 0
    
    grad_conv_ext = np.zeros(4*n)
    offset = 2*n
    grad_conv_ext[offset:offset+2*n-1] = grad_conv
    
    idx2_ext = idx2_matrix + offset
    Q = grad_conv_ext[idx2_ext]
    Q[~valid2_matrix] = 0
    term2 = Q @ h
    
    d_surr_dh = term1 + term2
    d_score_dh = (n / (total * total * w)) * (d_surr_dh - 2 * surrogate * h / total)
    
    grad_x = d_score_dh * h
    
    return score, grad_x

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def deadline_reached():
        return time_left() < 5.0

    rng = np.random.default_rng(42)
    
    # Candidate 0: Uniform (Theoretical Global Min)
    h_uniform = np.ones(N, dtype=np.float64)
    best_h = h_uniform.copy()
    best_score = compute_score(h_uniform)
    
    # Generate diverse initial candidates
    candidates = [h_uniform]
    
    x = np.linspace(0, 1, N)
    for width in [0.1, 0.2, 0.3, 0.5]:
        hg = np.exp(-((x - 0.5) / width) ** 2)
        candidates.append(hg)
        
    # Two blocks
    for frac in [0.25, 0.4, 0.45]:
        hb = np.zeros(N)
        end = int(frac * N)
        hb[:end] = 1.0
        hb[N-end:] = 1.0
        candidates.append(hb)

    # Random
    for _ in range(5):
        hr = np.random.default_rng(rng.integers(1000)).random(N)
        candidates.append(hr)

    # Evaluate initial candidates
    for hc in candidates:
        if deadline_reached():
            break
        hc = np.maximum(hc, 0.0)
        if np.sum(hc) <= 0:
            continue
        sc = compute_score(hc)
        if sc < best_score:
            best_score = sc
            best_h = hc.copy()

    # Optimization with Adaptive Beta
    # Start with large beta (smooth) and decrease
    betas = [10.0, 5.0, 2.0, 1.0, 0.5]
    
    for start_idx in range(min(5, len(candidates))):
        if deadline_reached():
            break
        
        h_start = candidates[start_idx].copy()
        h_start = np.maximum(h_start, 1e-10)
        x_start = np.log(h_start)
        
        current_beta = betas[0]
        x_opt = x_start
        
        for beta in betas:
            if deadline_reached():
                break
            
            # Refine with current beta
            try:
                result = minimize(
                    _obj_grad,
                    x_opt,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    options={
                        'maxiter': 50,
                        'ftol': 1e-12,
                        'gtol': 1e-10,
                        'maxls': 20
                    }
                )
                if result.fun < best_score:
                    x_opt = result.x
                    h_opt = np.exp(result.x)
                    h_opt = np.maximum(h_opt, 0.0)
                    sc = compute_score(h_opt)
                    if sc < best_score:
                        best_score = sc
                        best_h = h_opt.copy()
                else:
                    # If it didn't improve, keep previous x_opt
                    pass
            except Exception:
                pass

    # Final polish with very small beta
    if time_left() > 10:
        h_start = best_h.copy()
        h_start = np.maximum(h_start, 1e-10)
        x_start = np.log(h_start)
        
        try:
            result = minimize(
                _obj_grad,
                x_start,
                args=(0.1,),
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 100,
                    'ftol': 1e-14,
                    'gtol': 1e-12,
                    'maxls': 30
                }
            )
            if result.fun < best_score:
                h_opt = np.exp(result.x)
                h_opt = np.maximum(h_opt, 0.0)
                sc = compute_score(h_opt)
                if sc < best_score:
                    best_score = sc
                    best_h = h_opt.copy()
        except Exception:
            pass

    best_h = np.maximum(best_h, 0.0)
    return list(best_h)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
