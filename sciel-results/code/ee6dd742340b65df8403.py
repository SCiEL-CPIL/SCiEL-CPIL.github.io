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
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    return max_conv / (s * s)

def fft_convolve(h):
    """FFT-based autoconvolution for speed."""
    n = len(h)
    size = 2 * n
    f = np.fft.rfft(h, size)
    conv = np.fft.irfft(f * f, size)[:2*n-1]
    return conv

def smooth_max_surrogate(h, beta=100.0):
    """
    Smooth approximation of max(conv) / (sum h)^2.
    Uses log-sum-exp for smooth max.
    """
    s = np.sum(h)
    if s == 0:
        return 0.0, np.zeros_like(h)
    
    conv = fft_convolve(h)
    # Scale for numerical stability in log-sum-exp
    conv_scaled = conv / beta
    max_conv_scaled = np.max(conv_scaled)
    # log-sum-exp: max + log(sum(exp(x - max)))
    log_sum_exp = max_conv_scaled + np.log(np.sum(np.exp(conv_scaled - max_conv_scaled)))
    smooth_max = beta * log_sum_exp
    
    score = smooth_max / (s * s)
    
    # Gradient
    # d/dh_i [smooth_max / s^2] = (1/s^2) * d_smooth_max/dh_i - 2*smooth_max/s^3 * d_s/dh_i
    # d_s/dh_i = 1
    # d_smooth_max/dh_i = sum_j w_j * d_conv_j/dh_i where w_j = exp(conv_j - max) / sum(exp(...))
    # conv_j = sum_k h_k * h_{j-k} (with appropriate indexing)
    # This is complex; use numerical gradient or simplify
    
    # For simplicity, use finite differences for gradient (expensive but reliable)
    eps = 1e-7
    grad = np.zeros_like(h)
    for i in range(N):
        h_plus = h.copy()
        h_plus[i] += eps
        h_minus = h.copy()
        h_minus[i] -= eps
        if h_minus[i] < 0:
            h_minus[i] = 0
            s_plus = np.sum(h_plus)
            conv_plus = fft_convolve(h_plus)
            cp_scaled = conv_plus / beta
            max_cp = np.max(cp_scaled)
            lse_plus = max_cp + np.log(np.sum(np.exp(cp_scaled - max_cp)))
            sm_plus = beta * lse_plus
            score_plus = sm_plus / (s_plus * s_plus)
            
            s_minus = np.sum(h_minus)
            conv_minus = fft_convolve(h_minus)
            cm_scaled = conv_minus / beta
            max_cm = np.max(cm_scaled)
            lse_minus = max_cm + np.log(np.sum(np.exp(cm_scaled - max_cm)))
            sm_minus = beta * lse_minus
            score_minus = sm_minus / (s_minus * s_minus)
            
            grad[i] = (score_plus - score_minus) / (2 * eps)
        else:
            s_plus = np.sum(h_plus)
            conv_plus = fft_convolve(h_plus)
            cp_scaled = conv_plus / beta
            max_cp = np.max(cp_scaled)
            lse_plus = max_cp + np.log(np.sum(np.exp(cp_scaled - max_cp)))
            sm_plus = beta * lse_plus
            score_plus = sm_plus / (s_plus * s_plus)
            
            s_minus = np.sum(h_minus)
            conv_minus = fft_convolve(h_minus)
            cm_scaled = conv_minus / beta
            max_cm = np.max(cm_scaled)
            lse_minus = max_cm + np.log(np.sum(np.exp(cm_scaled - max_cm)))
            sm_minus = beta * lse_minus
            score_minus = sm_minus / (s_minus * s_minus)
            
            grad[i] = (score_plus - score_minus) / (2 * eps)
    
    return score, grad

def objective_and_grad(x, beta):
    """Objective and gradient for L-BFGS-B."""
    h = np.maximum(x, 0)
    score, grad = smooth_max_surrogate(h, beta)
    # Chain rule: if x_i < 0, h_i = 0, so dh_i/dx_i = 0
    mask = x >= 0
    grad_out = np.where(mask, grad, 0)
    return score, grad_out

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Initialize with uniform distribution
    h_uniform = np.full(N, 1.0)
    
    best_h = h_uniform.copy()
    best_score = compute_score(h_uniform)
    
    # Try gradient-based optimization from multiple initializations
    initializations = [
        "uniform",
        "gaussian_centered",
        "bimodal",
        "random_sparse"
    ]
    
    np.random.seed(42)
    
    for init_name in initializations:
        if time.time() > deadline - 30:
            break
        
        if init_name == "uniform":
            h0 = h_uniform.copy()
        elif init_name == "gaussian_centered":
            x = np.linspace(-1, 1, N)
            h0 = np.exp(-x**2 / 0.3)
            h0 = np.maximum(h0, 0)
            if np.sum(h0) > 0:
                h0 = h0 / np.sum(h0) * N
        elif init_name == "bimodal":
            x = np.linspace(-1, 1, N)
            h0 = np.exp(-(x-0.5)**2/0.1) + np.exp(-(x+0.5)**2/0.1)
            h0 = np.maximum(h0, 0)
            if np.sum(h0) > 0:
                h0 = h0 / np.sum(h0) * N
        else:  # random_sparse
            h0 = np.zeros(N)
            num_nonzero = N // 4
            idx = np.random.choice(N, num_nonzero, replace=False)
            h0[idx] = 1.0
            h0 = h0 / np.sum(h0) * N if np.sum(h0) > 0 else h_uniform.copy()
        
        # Optimize with decreasing beta (annealing)
        betas = [50.0, 100.0, 200.0, 500.0]
        current_h = h0.copy()
        
        for beta in betas:
            if time.time() > deadline - 15:
                break
            
            # L-BFGS-B optimization
            try:
                result = minimize(
                    lambda x: objective_and_grad(x, beta)[0],
                    current_h,
                    jac=lambda x: objective_and_grad(x, beta)[1],
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                current_h = np.maximum(result.x, 0)
            except Exception:
                pass
            
            # Evaluate actual score
            actual_score = compute_score(current_h)
            if actual_score < best_score:
                best_score = actual_score
                best_h = current_h.copy()
        
        # Also check the final point
        final_score = compute_score(current_h)
        if final_score < best_score:
            best_score = final_score
            best_h = current_h.copy()
    
    # Final check: ensure non-negativity
    best_h = np.maximum(best_h, 0)
    
    return best_h.tolist()
# EVOLVE-BLOCK-END
