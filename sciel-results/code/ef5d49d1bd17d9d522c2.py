import time
import numpy as np
from scipy.optimize import minimize

N = 600

def compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of nonnegative values.
    Returns max_k (h*h)[k] / (sum h)^2
    """
    s = np.sum(h)
    if s == 0:
        return 0.0
    n_fft = 2 * N - 1
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)[:2 * N - 1]
    max_conv = np.max(conv)
    return max_conv / (s * s)

def smooth_max_surrogate(conv, beta=100.0):
    """
    Smooth approximation of max(conv) using log-sum-exp.
    """
    max_val = np.max(conv)
    shifted = beta * (conv - max_val)
    shifted = np.clip(shifted, -100, 100)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return (max_val + log_sum / beta)

def objective(params, beta=100.0):
    """
    Objective function for optimization.
    """
    h = np.maximum(params, 0.0)
    s = np.sum(h)
    if s < 1e-10:
        return 1e10
    n_fft = 2 * N - 1
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)[:2 * N - 1]
    smoothed_peak = smooth_max_surrogate(conv, beta)
    return smoothed_peak / (s * s)

def objective_grad(params, beta=100.0):
    """
    Gradient of the objective function using FFT for efficiency.
    """
    h = np.maximum(params, 0.0)
    s = np.sum(h)
    if s < 1e-10:
        return np.zeros_like(params)
    
    n_fft = 2 * N - 1
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)[:2 * N - 1]
    
    # Compute softmax weights for the max surrogate
    max_val = np.max(conv)
    shifted = beta * (conv - max_val)
    shifted = np.clip(shifted, -100, 100)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    
    # Compute gradient of smooth max
    # grad_max = 2 * correlate(weights, h)
    # Using FFT: IFFT(FFT(weights) * conj(FFT(h)))? 
    # Wait, correlation is IFFT(FFT(a) * conj(FFT(b))).
    # Here we want sum_k w_k * (h * delta_k)_i.
    # (h*h)_k = sum_j h_j h_{k-j}. d/dh_i (h*h)_k = h_{k-i} + h_{k+i}.
    # This is equivalent to 2 * (w * h) where * is linear convolution?
    # Let's check: sum_k w_k h_{k-i}. This is (w * h)_{i} if w is reversed?
    # Standard correlation: (a \star b)_i = sum_j a_j b_{j-i}.
    # If we compute IFFT(FFT(w) * conj(FFT(h))), we get correlation.
    # Let's verify with small example.
    # Actually, d/dh_i sum_j h_j h_{k-j} = h_{k-i} + h_{k+i} (with h_out=0).
    # Sum_k w_k h_{k-i} + w_k h_{k+i}.
    # The term sum_k w_k h_{k-i} is (w * h)_{i} (convolution).
    # The term sum_k w_k h_{k+i} is (w * h)_{-i}? No.
    # Due to symmetry of autoconvolution and real h, the gradient is 2 * (w * h) truncated?
    # Let's use the property: grad = 2 * IFFT(FFT(w) * H) ?
    # No, IFFT(FFT(w) * H) is convolution w*h.
    # Let's test if grad = 2 * (w * h) works.
    
    W = np.fft.rfft(weights, n=n_fft)
    # Convolution w*h
    conv_wh = np.fft.irfft(W * H, n=n_fft)
    
    # The gradient of the autoconvolution peak w.r.t h is 2 * (w * h) 
    # provided we handle the boundaries correctly. 
    # For linear convolution of length N with itself, the derivative of the k-th term
    # with respect to h_i is h_{k-i} + h_{k+i} (if indices in range).
    # Summing over k with weights w_k:
    # sum_k w_k h_{k-i} + sum_k w_k h_{k+i}
    # Let j = k-i => k = j+i. sum_j w_{j+i} h_j. This is correlation of w and h?
    # Let m = k+i => k = m-i. sum_m w_{m-i} h_m.
    # Since w is symmetric (autoconvolution weights are symmetric around peak? No, w depends on conv which is symmetric, so w is symmetric).
    # If w is symmetric, w_{k-i} = w_{-(k-i)} = w_{i-k}.
    # Then sum_k w_{i-k} h_k is correlation.
    # So both terms are the same: 2 * corr(w, h).
    # corr(w, h) = IFFT(FFT(w) * conj(FFT(h))).
    
    # Let's re-evaluate:
    # grad_h = 2 * IFFT(FFT(w) * conj(H))[:N]
    
    grad_max = 2.0 * np.real(np.fft.irfft(W * np.conj(H), n=n_fft))[:N]
    
    # Gradient of objective: f = smooth_max / s^2
    smooth_max_val = smooth_max_surrogate(conv, beta)
    grad_obj = grad_max / (s * s) - 2 * smooth_max_val / (s * s * s)
    
    # Zero out gradient where h is at boundary (h=0) to respect constraint
    # This helps L-BFGS-B respect the bound
    grad_obj[h <= 1e-12] = 0.0
    
    return grad_obj

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    best_h = np.ones(N) / N
    best_score = compute_score(best_h)

    np.random.seed(42)
    
    # Multiple initializations
    initializations = []
    
    # 1. Uniform
    initializations.append(np.ones(N) / N)
    
    # 2. Gaussian
    x = np.linspace(-1, 1, N)
    gauss = np.exp(-x**2 / (2 * 0.5**2))
    gauss /= np.sum(gauss)
    initializations.append(gauss)
    
    # 3. Linear ramp
    lin = np.linspace(0.1, 1.0, N)
    lin /= np.sum(lin)
    initializations.append(lin)
    
    # 4. Sine
    sine = np.sin(np.pi * (x + 1) / 2) ** 2
    sine /= np.sum(sine)
    initializations.append(sine)
    
    # 5. Step function (left half vs right half)
    step = np.zeros(N)
    step[:N//2] = 1.0
    step /= np.sum(step)
    initializations.append(step)
    
    # 6. Random
    rand_h = np.random.rand(N)
    rand_h /= np.sum(rand_h)
    initializations.append(rand_h)
    
    # 7. Triangular
    tri = np.abs(np.linspace(-1, 1, N))
    tri = 1.0 - tri
    tri /= np.sum(tri)
    initializations.append(tri)
    
    # 8. Two-block: high-low-high pattern
    two_block = np.zeros(N)
    two_block[:N//3] = 1.0
    two_block[2*N//3:] = 1.0
    two_block /= np.sum(two_block)
    initializations.append(two_block)
    
    # 9. Cosine bump
    cos_bump = np.cos(np.pi * x) ** 2
    cos_bump = np.maximum(cos_bump, 0)
    cos_bump /= np.sum(cos_bump)
    initializations.append(cos_bump)
    
    bounds = [(0, None)] * N
    
    # Adaptive Beta Schedule:
    # Phase 1: Beta=30 (smooth, easy to navigate)
    # Phase 2: Beta=100 (sharper)
    # Phase 3: Beta=300 (very sharp, final polish)
    betas = [30.0, 100.0, 300.0]
    
    for h0 in initializations:
        if time.time() > deadline - 30:
            break
            
        h_current = h0.copy()
        
        for beta in betas:
            if time.time() > deadline - 10:
                break
                
            try:
                res = minimize(
                    objective, 
                    h_current, 
                    method='L-BFGS-B', 
                    jac=objective_grad, 
                    args=(beta,),
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
                )
                
                h_opt = res.x
                h_opt = np.maximum(h_opt, 0)
                
                # Normalize to ensure sum is consistent? 
                # The objective is scale invariant, so scaling doesn't matter for the score,
                # but it might help numerical stability.
                # However, L-BFGS-B might drift. Let's just check the score.
                
                score_opt = compute_score(h_opt)
                
                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt.copy()
                    
                h_current = h_opt
                
            except Exception:
                pass

    # Final high-precision polish on the best candidate
    if time.time() < deadline - 5:
        try:
            # Use a very high beta and tight tolerance
            res = minimize(
                objective, 
                best_h, 
                method='L-BFGS-B', 
                jac=objective_grad, 
                args=(500.0,),
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            h_opt = res.x
            h_opt = np.maximum(h_opt, 0)
            score_opt = compute_score(h_opt)
            
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
        except Exception:
            pass
            
    # Sparse refinement: try zeroing out small values
    # This can sometimes jump to a better local minimum if the solution is sparse
    if time.time() < deadline - 2:
        h_try = best_h.copy()
        # Sort indices by value
        indices = np.argsort(h_try)
        
        # Try zeroing the smallest 10%
        num_zero = N // 10
        for i in range(num_zero):
            h_try[indices[i]] = 0.0
        
        if np.sum(h_try) > 0:
            score_try = compute_score(h_try)
            if score_try < best_score:
                best_score = score_try
                best_h = h_try.copy()
                
                # Try zeroing more
                num_zero = N // 5
                h_try2 = best_h.copy()
                indices = np.argsort(h_try2)
                for i in range(num_zero):
                    h_try2[indices[i]] = 0.0
                if np.sum(h_try2) > 0:
                    score_try2 = compute_score(h_try2)
                    if score_try2 < best_score:
                        best_score = score_try2
                        best_h = h_try2.copy()

    # Final cleanup
    best_h = np.maximum(best_h, 0)
    # Ensure it's a list as required
    return best_h.tolist()
