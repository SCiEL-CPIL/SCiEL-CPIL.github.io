import numpy as np
import time

N = 600
L = 0.5  # length of [-1/4, 1/4]
dx = L / N

def compute_autoconvolution_peak(f):
    """Compute max of autoconvolution of step function f."""
    # f is array of N heights, each over width dx
    # Autoconvolution: (f*f)(t) = integral f(s)f(t-s) ds
    # For step functions on equal intervals, we can compute via FFT
    # Pad to avoid circular convolution
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    autoconv = np.fft.irfft(F * F.conj(), n=n_fft)
    # Scale: each interval has width dx, so convolution integral needs dx factor
    # Actually for step functions: (f*f)(k) = sum_i f[i] * f[k-i] * dx
    # The FFT approach: if f represents values on grid with spacing dx,
    # then convolution is sum f[i]*f[j] where i+j=k, times dx
    # Using FFT: ifft(fft(f)*fft(f)) gives circular conv, need to scale by dx
    autoconv *= dx
    # The autoconvolution is defined on [0, L] for the support
    # We want the max over the valid range
    # Valid range of autoconv: indices 0 to N-1 roughly (support is [0,L])
    # Actually for two functions on [0,L], autoconv is on [0,2L]
    # But our function is on [-1/4, 1/4], so autoconv is on [-1/2, 1/2]
    # Let me reconsider: f is defined on [-1/4, 1/4] with N intervals
    # f*f is defined on [-1/2, 1/2]
    # The max of f*f is what we want
    return np.max(autoconv)

def compute_integral(f):
    """Compute integral of step function."""
    return np.sum(f) * dx

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    # Initialize with uniform
    f = np.ones(N)
    
    # Current best
    int_f = compute_integral(f)
    peak = compute_autoconvolution_peak(f)
    best_score = peak / (int_f * int_f)
    best_f = f.copy()
    
    # Use frequency domain parameterization
    # F = rfft(f), work with amplitude and phase
    # But for real symmetric f, we can work with real FFT coefficients
    
    # Alternative: direct gradient descent on f with non-negativity constraint
    # Objective: minimize max(f*f) / (int f)^2
    # Use log-sum-exp approximation for max to make it differentiable
    
    def objective_and_grad(f):
        """Compute objective and gradient using soft approximation of max."""
        int_f = np.sum(f) * dx
        F = np.fft.rfft(f, n=2*N)
        autoconv = np.fft.irfft(F * F.conj(), n=2*N) * dx
        
        # Soft max approximation
        beta = 50.0
        shifted = autoconv * beta
        # Clip for numerical stability
        shifted = np.clip(shifted, -50, 50)
        logsumexp = np.log(np.sum(np.exp(shifted))) / beta
        
        # Gradient of logsumexp w.r.t. autoconv
        weights = np.exp(shifted - np.log(np.sum(np.exp(shifted))))
        
        # Gradient of autoconv w.r.t. f
        # autoconv = ifft(fft(f)^2) * dx
        # d(autoconv)/d(f) involves convolution with f
        # Use FFT: grad_autoconv = ifft(2 * fft(f) * conj(fft(f))) ... no
        # autoconv[k] = sum_i f[i]*f[k-i]*dx
        # d(autoconv[k])/d(f[j]) = f[k-j]*dx + f[j-k]*dx (for valid indices)
        # This is 2*dx*f convolved with delta, i.e., 2*dx*f (with proper indexing)
        # Actually: d/d(f[j]) sum_i f[i]*f[i+j]*dx = f[j]*dx + f[-j]*dx
        # For the full autoconv, grad = 2 * dx * f (circularly, but we need linear)
        
        # Simpler: use finite differences or analytic via FFT
        # Let's use: grad_obj = weights @ (d autoconv / d f) / (int_f^2)
        # d autoconv / d f[j] = 2 * dx * f convolved appropriately
        
        # For linear autoconvolution: (f*f)[k] = sum_i f[i]*f[k-i]*dx for k in [0, 2N-1]
        # d/d(f[j]) (f*f)[k] = f[k-j]*dx + f[j-k]*dx (when indices valid)
        # So grad_autoconv[k] = sum over j of weights[k] * (f[k-j]*dx + f[j-k]*dx)
        # = 2*dx * (weights conv f)[j] (approximately, for linear conv)
        
        # Use FFT for efficiency
        # grad_autoconv[j] = 2*dx * sum_k weights[k] * f[k-j] (for linear conv)
        # This is 2*dx * (weights * f) convolved, but reversed
        # Actually: sum_k weights[k]*f[k-j] = (weights * flip(f))[j]
        
        # Let me just compute it directly for clarity, N=600 is small
        # But 600x600 = 360000, that's fine
        
        # grad_f[j] = sum_k weights[k] * d(autoconv[k])/d(f[j])
        # d(autoconv[k])/d(f[j]) = dx*(f[k-j] + f[j-k]) for valid k-j and j-k in [0,N-1]
        
        # Use correlation: grad_f = 2*dx * correlate(weights, f)
        # where correlate(a,b)[j] = sum_k a[k]*b[k-j]
        
        # For linear correlation, pad
        n_corr = 2 * N
        W = np.fft.rfft(weights, n=n_corr)
        Ff = np.fft.rfft(f, n=n_corr)
        # correlate(weights, f) = ifft(conj(W) * Ff) ... need to be careful
        # np.correlate(weights, f, mode='full') gives what we want
        # But that's O(N^2). For N=600, that's 360K ops, fine.
        
        # Actually let's use FFT-based correlation
        # corr[j] = sum_k weights[k] * f[k-j]
        # = ifft(fft(weights) * conj(fft(f))) ... let me verify
        # Standard: cross-correlation via FFT
        corr = np.fft.irfft(W * np.conj(Ff), n=n_corr)
        # Take the part corresponding to j in [0, N-1]
        # The full correlation has length 2N-1, centered
        # For j in [0,N-1], we want corr[N-1+j] ... this is getting complex
        
        # Let me just do direct computation for correctness
        grad_f = np.zeros(N)
        for j in range(N):
            # d(autoconv[k])/d(f[j]) = dx * (f[k-j] if 0<=k-j<N else 0) + dx*(f[j-k] if 0<=j-k<N else 0)
            # autoconv[k] for k in [0, 2N-1]
            # Term 1: f[k-j], valid when 0 <= k-j < N, i.e., j <= k < j+N
            # Term 2: f[j-k], valid when 0 <= j-k < N, i.e., j-N < k <= j
            # So for each j, sum over k in [max(0,j), min(2N-1, j+N-1)] of weights[k]*dx*f[k-j]
            # Plus sum over k in [max(0,j-N+1), min(2N-1,j)] of weights[k]*dx*f[j-k]
            
            k1_start = max(0, j)
            k1_end = min(2*N-1, j+N-1)
            if k1_start <= k1_end:
                k1 = np.arange(k1_start, k1_end+1)
                idx1 = k1 - j
                grad_f[j] += np.sum(weights[k1] * f[idx1]) * dx
            
            k2_start = max(0, j-N+1)
            k2_end = min(2*N-1, j)
            if k2_start <= k2_end:
                k2 = np.arange(k2_start, k2_end+1)
                idx2 = j - k2
                grad_f[j] += np.sum(weights[k2] * f[idx2]) * dx
        
        # Objective: logsumexp / int_f^2
        obj = logsumexp / (int_f * int_f)
        
        # Gradient of obj w.r.t. f
        # d/d(f) [logsumexp / int_f^2] = (grad_logsumexp * int_f^2 - logsumexp * 2*int_f * dx) / int_f^4
        # = grad_logsumexp / int_f^2 - 2*logsumexp*dx / int_f^3
        grad_obj = grad_f / (int_f * int_f) - 2 * logsumexp * dx / (int_f ** 3)
        
        return obj, grad_obj
    
    # Gradient descent with adaptive step size
    lr = 0.01
    best_so_far = best_score
    
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        obj, grad = objective_and_grad(f)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        
        # Take step
        f_new = f - lr * grad
        f_new = np.maximum(f_new, 0)  # enforce non-negativity
        
        # Check if improved
        int_new = compute_integral(f_new)
        peak_new = compute_autoconvolution_peak(f_new)
        score_new = peak_new / (int_new * int_new)
        
        if score_new < best_so_far:
            best_so_far = score_new
            best_f = f_new.copy()
            f = f_new.copy()
            # Decrease learning rate slightly on improvement
            lr *= 0.98
        else:
            # Rejected, decrease learning rate
            lr *= 0.95
            if lr < 1e-8:
                break
        
        # Periodically try to improve with smaller steps
        if iteration % 100 == 0 and iteration > 0:
            # Try a few random perturbations
            for _ in range(5):
                f_pert = f + 0.01 * np.random.randn(N)
                f_pert = np.maximum(f_pert, 0)
                int_pert = compute_integral(f_pert)
                peak_pert = compute_autoconvolution_peak(f_pert)
                score_pert = peak_pert / (int_pert * int_pert)
                if score_pert < best_so_far:
                    best_so_far = score_pert
                    best_f = f_pert.copy()
                    f = f_pert.copy()
    
    return list(best_f)
