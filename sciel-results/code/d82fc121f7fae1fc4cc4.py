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
    Uses log-sum-exp for smooth max with analytical gradient.
    """
    s = np.sum(h)
    if s == 0:
        return 0.0, np.zeros_like(h)
    
    conv = fft_convolve(h)
    # Scale for numerical stability in log-sum-exp
    conv_scaled = conv / beta
    max_conv_scaled = np.max(conv_scaled)
    exp_terms = np.exp(conv_scaled - max_conv_scaled)
    log_sum_exp = max_conv_scaled + np.log(np.sum(exp_terms))
    smooth_max = beta * log_sum_exp
    
    score = smooth_max / (s * s)
    
    # Gradient computation
    # d/dh_i [smooth_max / s^2] = (1/s^2) * d_smooth_max/dh_i - 2*smooth_max/s^3 * 1
    # d_smooth_max/dh_i = sum_j w_j * d_conv_j/dh_i
    # where w_j = exp(conv_j/beta - max) / sum(exp(...)) = exp_terms[j] / sum(exp_terms)
    # conv_j = sum_k h[k] * h[j-k] (with appropriate indexing for full convolution)
    # d_conv_j/dh_i = h[j-i] + h[i-j] (for the two terms in the convolution)
    # For full convolution of length 2N-1, index j ranges from 0 to 2N-2
    # conv[j] = sum_{k=max(0,j-N+1)}^{min(N-1,j)} h[k] * h[j-k]
    # d_conv[j]/dh[i] = h[j-i] if 0 <= j-i < N else 0, plus h[i-j] if 0 <= i-j < N else 0
    # But we need to be careful with the indexing.
    
    # Let's compute the gradient more carefully using the structure of the convolution.
    # w = exp_terms / sum(exp_terms) has length 2N-1
    # grad_smooth_max[i] = sum_j w[j] * (d_conv[j]/dh[i])
    # d_conv[j]/dh[i] = h[j-i] (if j-i is valid index) + h[i-j] (if i-j is valid index)
    # Actually for autoconvolution conv[j] = sum_k h[k]*h[j-k], 
    # d_conv[j]/dh[i] = h[j-i] + h[i-j] when both are valid indices in [0, N-1]
    
    w = exp_terms / np.sum(exp_terms)
    
    # For each i, grad_smooth_max[i] = sum_j w[j] * (h[j-i] if 0<=j-i<N else 0 + h[i-j] if 0<=i-j<N else 0)
    # This can be computed as a convolution of w with h, but with appropriate shifts.
    
    # Let's think of it differently. 
    # grad_smooth_max = d/dh [beta * log(sum_j exp(conv[j]/beta))]
    # = sum_j w[j] * d_conv[j]/dh
    # where w[j] = exp(conv[j]/beta) / sum_k exp(conv[k]/beta)
    
    # d_conv[j]/dh is a matrix M[j,i] = h[j-i] + h[i-j] (with boundary conditions)
    # So grad_smooth_max[i] = sum_j w[j] * M[j,i]
    
    # M[j,i] = h[j-i] if 0 <= j-i < N, else 0, plus h[i-j] if 0 <= i-j < N, else 0
    # Note that h[j-i] is nonzero only when j-i >= 0 and j-i < N, i.e., i <= j < i+N
    # And h[i-j] is nonzero only when i-j >= 0 and i-j < N, i.e., j <= i < j+N
    
    # Let's compute this using FFT or direct convolution.
    # grad_smooth_max[i] = sum_j w[j] * h[j-i] (for j >= i) + sum_j w[j] * h[i-j] (for j <= i)
    # = sum_{k=0}^{N-1} w[i+k] * h[k] (for i+k < 2N-1) + sum_{k=0}^{N-1} w[i-k] * h[k] (for i-k >= 0)
    
    # Let's compute using convolution.
    # First term: for each i, sum over k of w[i+k] * h[k] where i+k < len(w)
    # This is like a cross-correlation.
    
    # Let's use a direct approach with numpy operations.
    # Create a matrix W of shape (N, 2N-1) where W[i,j] = w[j] if |j-i| < N else 0
    # Actually, let's just compute it directly.
    
    # grad_sm[i] = sum_j w[j] * (h[j-i] if 0<=j-i<N else 0) + sum_j w[j] * (h[i-j] if 0<=i-j<N else 0)
    
    # First part: sum_j w[j] * h[j-i] for j such that 0 <= j-i < N
    # Let k = j-i, so j = i+k, k ranges from max(0, -i) to min(N-1, 2N-2-i)
    # Since i is in [0, N-1], j-i ranges from -i to 2N-2-i
    # Valid k: 0 <= k < N and 0 <= i+k < 2N-1
    # So k ranges from 0 to min(N-1, 2N-2-i)
    
    # Second part: sum_j w[j] * h[i-j] for j such that 0 <= i-j < N
    # Let k = i-j, so j = i-k, k ranges from max(0, i-(2N-2)) to min(N-1, i)
    # Since i is in [0, N-1], i-j ranges from i-(2N-2) to i
    # Valid k: 0 <= k < N and 0 <= i-k < 2N-1
    # So k ranges from max(0, i-(2N-2)) to min(N-1, i)
    
    # Let's compute using vectorized operations.
    # For the first part, we can think of it as: for each i, take w[i:i+N] (padded) and multiply by h, then sum.
    # For the second part, take w[0:i+N] reversed appropriately and multiply by h, then sum.
    
    # Actually, let's use a simpler approach. Since N=600, we can afford some O(N^2) computation.
    # But let's try to be smarter.
    
    # grad_sm[i] = sum_{k=0}^{N-1} h[k] * (w[i+k] if i+k < 2N-1 else 0 + w[i-k] if i-k >= 0 else 0)
    
    # Let's create an array W_ext of length 2N-1 + N-1 = 3N-2, centered, such that
    # W_ext[i + (N-1) + k] = w[i+k] for valid i+k
    # This is getting complicated. Let's just do it with a loop over i but vectorize over k.
    
    grad_sm = np.zeros(N)
    for i in range(N):
        # First term: sum_k h[k] * w[i+k] for k where 0 <= i+k < 2N-1
        k_range1 = np.arange(0, N)
        j_range1 = i + k_range1
        valid1 = (j_range1 >= 0) & (j_range1 < 2*N-1)
        term1 = np.sum(h[k_range1[valid1]] * w[j_range1[valid1]])
        
        # Second term: sum_k h[k] * w[i-k] for k where 0 <= i-k < 2N-1
        k_range2 = np.arange(0, N)
        j_range2 = i - k_range2
        valid2 = (j_range2 >= 0) & (j_range2 < 2*N-1)
        term2 = np.sum(h[k_range2[valid2]] * w[j_range2[valid2]])
        
        grad_sm[i] = term1 + term2
    
    # Now, score = smooth_max / s^2
    # d_score/dh_i = (1/s^2) * grad_sm[i] - 2*smooth_max/s^3 * 1
    grad_score = grad_sm / (s * s) - 2 * smooth_max / (s * s * s)
    
    return score, grad_score

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
                    lambda x, b=beta: objective_and_grad(x, b)[0],
                    current_h,
                    jac=lambda x, b=beta: objective_and_grad(x, b)[1],
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
