"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 600
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0  # seconds

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a nonnegative step function with N equal intervals on [-1/4, 1/4]
    to minimize the normalized maximum of its autoconvolution.
    
    The autoconvolution of a step function h on N equal intervals can be computed
    via FFT. The score is the maximum of the autoconvolution normalized by 
    (sum of h) to give an upper bound on C1.
    """
    start_time = time.time()
    
    # Initial guess: uniform function
    h = np.ones(N)
    
    # Compute initial score
    def compute_score(h_vec):
        """
        Compute the normalized maximum of the autoconvolution.
        h_vec: array of N nonnegative heights.
        The step function is supported on [-1/4, 1/4] with N equal intervals.
        The autoconvolution (h * h) is supported on [-1/2, 1/2].
        We need to compute the autoconvolution on the grid and find its maximum.
        """
        # Ensure nonnegativity
        h_vec = np.maximum(h_vec, 0)
        
        # Compute autoconvolution via FFT
        # The autoconvolution of h with itself: (h * h)(k) = sum_j h(j) h(k-j)
        # Using FFT: ifft(fft(h) * fft(h)) gives the circular convolution
        # For linear convolution, we need to pad to avoid wrap-around
        n_conv = 2 * N - 1
        h_padded = np.zeros(n_conv)
        h_padded[:N] = h_vec
        fft_h = np.fft.fft(h_padded, n=2 * n_conv)
        conv = np.real(np.fft.ifft(fft_h * fft_h))[:n_conv]
        
        # The autoconvolution is supported on [-1/2, 1/2]
        # The grid for the convolution has n_conv points
        # The maximum of the autoconvolution:
        max_conv = np.max(conv)
        
        # Normalize: the "normalized maximum" likely means max_conv / (sum(h)^2) 
        # or some similar normalization. Let's use max_conv / (sum(h))^2.
        total = np.sum(h_vec)
        if total == 0:
            return 0.0
        return max_conv / (total * total)
    
    current_score = compute_score(h)
    best_h = h.copy()
    best_score = current_score
    
    # Try to optimize using a simple iterative approach
    # Use a smoothed max approach: minimize soft-max of the autoconvolution
    # This is a non-convex problem, so we use local search / gradient descent
    
    # Gradient of the soft-max objective
    def soft_max_grad(h_vec, beta=10.0):
        """
        Compute the gradient of the soft-max of the autoconvolution.
        Uses a smooth approximation to the max function.
        """
        h_vec = np.maximum(h_vec, 0)
        n_conv = 2 * N - 1
        h_padded = np.zeros(n_conv)
        h_padded[:N] = h_vec
        fft_h = np.fft.fft(h_padded, n=2 * n_conv)
        conv = np.real(np.fft.ifft(fft_h * fft_h))[:n_conv]
        
        # Soft max: log(sum(exp(beta * conv)))
        # Gradient w.r.t. conv: beta * exp(beta * conv) / sum(exp(beta * conv))
        # Use log-sum-exp for numerical stability
        max_conv = np.max(conv)
        shifted = beta * (conv - max_conv)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        weights = exp_shifted / sum_exp
        
        # Gradient of conv w.r.t. h_vec
        # conv = ifft(fft(h_padded) * fft(h_padded))
        # d(conv)/d(h_j) is a shifted version of h_padded
        # Specifically, conv[k] = sum_j h_padded[j] h_padded[k-j]
        # d(conv[k])/d(h_padded[j]) = h_padded[k-j] + h_padded[j-k] (for the autocorrelation-like structure)
        # Actually, for autoconvolution h*h, d(h*h)[k]/d(h[j]) = h[k-j]
        # So the gradient of the soft-max w.r.t. h_padded is:
        # grad_h_padded[j] = sum_k weights[k] * h_padded[k-j]
        
        # Compute this efficiently using correlation
        # grad_h_padded = correlate(weights, h_padded)
        # But we need to be careful with the indexing
        
        # Use FFT for efficiency
        grad_padded = np.zeros(n_conv)
        # grad_padded[j] = sum_k weights[k] * h_padded[k-j]
        # This is the cross-correlation of weights and h_padded
        # = ifft(conj(fft(weights)) * fft(h_padded))
        fft_w = np.fft.fft(weights, n=2 * n_conv)
        fft_hp = np.fft.fft(h_padded, n=2 * n_conv)
        grad_padded = np.real(np.fft.ifft(np.conj(fft_w) * fft_hp))[:n_conv]
        
        # grad_h_vec is the first N elements of grad_padded
        grad = grad_padded[:N]
        return grad
    
    # Iterative optimization
    max_iters = 500
    lr = 0.01
    beta = 5.0  # Smoothness parameter for soft-max
    
    for it in range(max_iters):
        # Check time
        if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        # Compute gradient
        grad = soft_max_grad(h, beta)
        
        # Gradient descent step
        h = h - lr * grad
        
        # Project onto nonnegative orthant
        h = np.maximum(h, 0)
        
        # Normalize to avoid drift (optional)
        if np.sum(h) > 0:
            pass  # Keep the scale as is, since the score is normalized
        
        # Compute current score
        current_score = compute_score(h)
        
        # Track best
        if current_score < best_score:
            best_h = h.copy()
            best_score = current_score
        
        # Adaptive learning rate
        if it % 50 == 0:
            lr *= 0.9
    
    # Return the best solution found
    return list(np.maximum(best_h, 0))
# EVOLVE-BLOCK-END
