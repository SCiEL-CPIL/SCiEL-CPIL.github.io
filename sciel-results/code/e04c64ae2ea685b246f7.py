import numpy as np
import time
from scipy.optimize import minimize

N = 600
dx = 0.5 / N  # interval width

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    def compute_autoconv(x):
        """Compute the autoconvolution of x (normalized)."""
        x = np.maximum(x, 0)
        s = np.sum(x)
        if s < 1e-15:
            return None, None, None
        x_norm = x / s
        n_fft = 2 * N
        x_pad = np.zeros(n_fft)
        x_pad[:N] = x_norm
        X = np.fft.fft(x_pad)
        conv = np.fft.ifft(np.abs(X)**2).real
        return conv, x_norm, X
    
    def autoconvolution_max(x):
        """Compute the normalized maximum of the autoconvolution."""
        conv, _, _ = compute_autoconv(x)
        if conv is None:
            return 1e15
        return np.max(conv)
    
    def smooth_max_logsumexp(conv, beta=100.0):
        """Smooth approximation to max using log-sum-exp."""
        conv_clipped = np.clip(conv, 1e-300, None)
        m = np.max(conv_clipped)
        return m + np.log(np.sum(np.exp(beta * (conv_clipped - m)))) / beta
    
    def objective_smooth(x, beta=100.0):
        """Smooth objective using log-sum-exp surrogate for max."""
        conv, _, _ = compute_autoconv(x)
        if conv is None:
            return 1e15
        return smooth_max_logsumexp(conv, beta)
    
    def grad_smooth(x, beta=100.0):
        """Gradient of smooth objective using log-sum-exp surrogate."""
        conv, x_norm, X = compute_autoconv(x)
        if conv is None:
            return np.zeros(N)
        
        # Gradient of log-sum-exp: (1/beta) * exp(beta*(conv - m)) / sum(exp(beta*(conv - m)))
        conv_clipped = np.clip(conv, 1e-300, None)
        m = np.max(conv_clipped)
        weights = np.exp(beta * (conv_clipped - m))
        weights /= np.sum(weights)
        
        # Gradient of conv[i] w.r.t. x_norm[j] is 2 * conv[i-j] (with appropriate indexing)
        # For normalized x, we need to account for the normalization
        # Let's use the fact that conv = ifft(|FFT(x_norm)|^2)
        # d(conv[k])/d(x_norm[j]) = 2 * Re(x_norm[j] * conj(X[k-j]) * exp(2*pi*i*k/N))
        # This is complex, so let's use finite differences or a simpler approach
        
        # Actually, let's use the property that for the autoconvolution:
        # conv = ifft(|FFT(x_norm)|^2)
        # The gradient of the log-sum-exp of conv w.r.t. x is:
        # sum_k weights[k] * d(conv[k])/d(x)
        
        # For simplicity and correctness, use the fact that:
        # d(conv[k])/d(x_norm[j]) = 2 * Re(FFT(x_norm)[k-j] * conj(x_norm[j]) * exp(2*pi*i*k*j/N))
        # This is getting complex. Let's use a different approach.
        
        # The autoconvolution can be written as:
        # conv[k] = sum_j x_norm[j] * x_norm[j-k] (with appropriate bounds)
        # So d(conv[k])/d(x_norm[j]) = x_norm[j-k] + x_norm[j+k] (with bounds handling)
        
        # But since we're using FFT, let's use the frequency domain gradient
        # d(conv)/d(X) = 2 * conj(X) (element-wise, with appropriate handling)
        # d(X)/d(x_norm) = FFT matrix
        
        # For now, let's return zeros and rely on L-BFGS-B's numerical gradient
        # Or we can compute it properly:
        
        # The gradient of the smooth max w.r.t. x_norm is:
        # grad_x_norm = sum_k weights[k] * d(conv[k])/d(x_norm)
        # where d(conv[k])/d(x_norm[j]) can be computed via FFT
        
        # conv = ifft(|X|^2)
        # d(conv[k])/d(X[m]) = 2 * conj(X[m]) * delta_{k,m} (in frequency domain)
        # But conv is in time domain, so:
        # d(conv[k])/d(X[m]) = 2 * conj(X[m]) * exp(2*pi*i*m*k/N) / N
        
        # And d(X[m])/d(x_norm[j]) = exp(-2*pi*i*m*j/N)
        
        # So d(conv[k])/d(x_norm[j]) = sum_m 2 * conj(X[m]) * exp(2*pi*i*m*k/N) / N * exp(-2*pi*i*m*j/N)
        # = 2 / N * sum_m conj(X[m]) * exp(2*pi*i*m*(k-j)/N)
        # = 2 * conj(ifft(X)[k-j]) (with appropriate indexing)
        # = 2 * conj(x_norm[k-j]) (since x_norm = ifft(X))
        
        # Wait, that's not right. Let me think again.
        
        # X = FFT(x_norm), so x_norm = ifft(X)
        # conv = ifft(|X|^2)
        
        # d(conv)/d(X) in frequency domain: d(ifft(|X|^2))/d(X)
        # |X|^2 = X * conj(X)
        # d(X * conj(X))/d(X) = conj(X) (treating conj(X) as independent)
        # But X and conj(X) are not independent.
        
        # Actually, for real x_norm, X[m] = conj(X[-m]), so we need to be careful.
        
        # Let's just use the time domain:
        # conv[k] = sum_j x_norm[j] * x_norm[j-k] (with x_norm[i] = 0 for i outside [0,N))
        # So d(conv[k])/d(x_norm[j]) = x_norm[j-k] + x_norm[j+k] (with bounds)
        
        # This is O(N^2) which is too slow for N=600.
        
        # Alternative: use the fact that the gradient of the log-sum-exp is a weighted average
        # of the gradients of conv[k], and compute it via FFT.
        
        # Let's use a simpler approach: return the gradient of the max (not smooth)
        # which is just a one-hot vector at the argmax, scaled appropriately.
        
        # For the smooth version, the gradient is approximately concentrated near the max.
        
        # Let's compute it properly using the FFT relationship:
        # grad_x_norm[k] = sum_j weights[j] * (x_norm[k-j] + x_norm[k+j])
        # This is a convolution of weights with x_norm, which can be done via FFT!
        
        # weights has length 2N-1 (for lags -N+1 to N-1)
        # Actually, conv has length 2N-1 as well (lags from -(N-1) to N-1)
        
        # Let's set up the weights properly
        # conv has indices 0 to 2N-2, corresponding to lags 0, 1, ..., N-1, -N+1, ..., -1
        # Or is it lags -(N-1) to N-1? Let's check.
        
        # When we do ifft(|FFT(x_pad)|^2), the result has length n_fft = 2N
        # The first N elements correspond to lags 0 to N-1
        # The last N elements correspond to lags -N to -1 (or -1 to -N?)
        
        # Actually, for circular convolution, the lags are 0 to 2N-1
        # But we want linear convolution, so we zero-padded.
        
        # The result conv[0:N] corresponds to lags 0 to N-1
        # The result conv[N:2N] corresponds to lags -N to -1 (wrapped around)
        
        # For the gradient, we need:
        # grad_x_norm[k] = sum_l weights[l] * d(conv[l])/d(x_norm[k])
        # where d(conv[l])/d(x_norm[k]) = x_norm[k-l] + x_norm[k+l] (with bounds)
        
        # This is equivalent to:
        # grad_x_norm = convolve(weights, x_norm) + convolve(weights, x_norm[::-1])
        # (with appropriate indexing)
        
        # Let's just use numerical gradient for now, or return zeros
        # and let L-BFGS-B handle it.
        
        return np.zeros(N)
    
    def objective(x):
        return autoconvolution_max(x)
    
    # Start with a smooth raised-cosine-like distribution
    t = np.linspace(0, 1, N)
    # Use a broad bump: cosine raised to a power to make it flatter in the middle
    x0 = np.cos(np.pi * (t - 0.5))**2  # This is a smooth bump, zero at edges
    x0 = np.maximum(x0, 0)
    
    # Ensure it's non-negative and has reasonable magnitude
    x0 = x0 / np.sum(x0) * N  # Scale so integral is ~1
    
    best_x = x0.copy()
    best_obj = objective(x0)
    
    # Try different initial shapes and optimize
    candidates = [
        x0.copy(),
        np.ones(N),  # Uniform
        np.cos(np.pi * (np.linspace(0, 1, N) - 0.5))**1.5,  # Sharper cosine
        np.exp(-((np.linspace(0, 1, N) - 0.5)**2) / 0.1),  # Gaussian-like
        np.exp(-((np.linspace(0, 1, N) - 0.5)**2) / 0.2),  # Broader Gaussian
        np.exp(-((np.linspace(0, 1, N) - 0.5)**2) / 0.05),  # Narrow Gaussian
        np.abs(np.sin(2 * np.pi * (np.linspace(0, 1, N) - 0.5)))**0.5,  # Wavy
    ]
    
    # First pass: use smooth objective for better gradient behavior
    for x_init in candidates:
        x_init = np.maximum(x_init, 0)
        if np.sum(x_init) < 1e-10:
            continue
        x_init = x_init / np.sum(x_init) * N
        
        if time.time() > deadline - 30:
            break
        
        try:
            # Use smooth objective with moderate beta
            result = minimize(lambda v: objective_smooth(v, beta=50.0), x_init.copy(), 
                            method='L-BFGS-B',
                            bounds=[(0, None)] * N,
                            options={'maxiter': 2000, 'maxfun': 20000, 'gtol': 1e-8})
            if result.fun < best_obj:
                best_x = result.x
                best_obj = objective(result.x)
        except Exception:
            pass
    
    # Second pass: refine with exact max objective
    for x_init in [best_x.copy()]:
        if time.time() > deadline - 20:
            break
        try:
            result = minimize(objective, x_init.copy(), method='L-BFGS-B',
                            bounds=[(0, None)] * N,
                            options={'maxiter': 5000, 'maxfun': 50000, 'gtol': 1e-12})
            if result.fun < best_obj:
                best_x = result.x
                best_obj = result.fun
        except Exception:
            pass
    
    # Local refinement with full coordinate descent
    if time.time() < deadline - 15:
        # Adaptive step size based on current scale
        base_step = np.mean(best_x) * 0.1
        step_sizes = [base_step * 0.1, base_step * 0.5, base_step]
        
        for step_size in step_sizes:
            if time.time() > deadline - 5:
                break
            for iteration in range(100):
                if time.time() > deadline - 5:
                    break
                improved = False
                for i in range(N):
                    if time.time() > deadline - 5:
                        break
                    
                    # Try increasing
                    x_trial = best_x.copy()
                    x_trial[i] = max(0, x_trial[i] + step_size)
                    obj_trial = objective(x_trial)
                    if obj_trial < best_obj - 1e-14:
                        best_x = x_trial
                        best_obj = obj_trial
                        improved = True
                    
                    # Try decreasing
                    if best_x[i] > step_size:
                        x_trial = best_x.copy()
                        x_trial[i] = best_x[i] - step_size
                        obj_trial = objective(x_trial)
                        if obj_trial < best_obj - 1e-14:
                            best_x = x_trial
                            best_obj = obj_trial
                            improved = True
                
                if not improved:
                    break
    
    # Final high-precision polish: very small steps
    if time.time() < deadline - 5:
        final_step = np.mean(best_x) * 0.01
        for iteration in range(200):
            if time.time() > deadline - 3:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 3:
                    break
                
                # Try small increase
                x_trial = best_x.copy()
                x_trial[i] = max(0, x_trial[i] + final_step)
                obj_trial = objective(x_trial)
                if obj_trial < best_obj - 1e-16:
                    best_x = x_trial
                    best_obj = obj_trial
                    improved = True
                
                # Try small decrease
                if best_x[i] > final_step:
                    x_trial = best_x.copy()
                    x_trial[i] = best_x[i] - final_step
                    obj_trial = objective(x_trial)
                    if obj_trial < best_obj - 1e-16:
                        best_x = x_trial
                        best_obj = obj_trial
                        improved = True
            
            if not improved:
                break
    
    # Final cleanup: ensure nonnegativity
    best_x = np.maximum(best_x, 0)
    
    return list(best_x)
