"""Rewritten program: spectral flattening approach for signed step function."""
import numpy as np
import time

N = 400

def construct():
    """
    Construct a signed step function minimizing max|autoconv| / (sum(x))^2.
    
    Strategy: Flatten the power spectrum |X(f)|^2 to minimize sidelobes in
    autoconvolution. Use gradient descent on log-power spectrum, then
    reconstruct time-domain signal.
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    def autoconv_linear(x):
        """Compute linear autoconvolution via FFT."""
        n = len(x)
        size = 1
        while size < 2 * n - 1:
            size *= 2
        fx = np.fft.rfft(x, size)
        power = fx * np.conj(fx)
        ac = np.fft.irfft(power, size)[:2 * n - 1]
        return ac
    
    def metric(x):
        """Compute the objective: max|c_k| / (sum(x))^2 for k != 0."""
        s = np.sum(x)
        if abs(s) < 1e-15:
            return float('inf')
        c = autoconv_linear(x)
        if len(c) <= 1:
            return 0.0
        return np.max(np.abs(c[1:])) / (s * s)
    
    # Phase 1: Spectral flattening via gradient descent on log-power spectrum
    # We want |X(f)|^2 to be as flat as possible (minimize variance of log |X|^2)
    
    # Initialize with a complex exponential (chirp)
    np.random.seed(123)
    # Try several initial frequencies
    best_x = None
    best_score = float('inf')
    
    for init_seed in range(5):
        if time.time() > deadline - 5:
            break
        
        np.random.seed(init_seed * 7 + 3)
        
        # Initialize with a real chirp: x_i = cos(2*pi*(f0*i + 0.5*f1*i^2))
        f0 = np.random.uniform(0.1, 0.4)
        f1 = np.random.uniform(0.01, 0.1)
        t = np.arange(N)
        x = np.cos(2 * np.pi * (f0 * t + 0.5 * f1 * t * t))
        
        # Also add a DC offset to ensure sum is not zero
        x += 0.5
        
        # Gradient descent to flatten the power spectrum
        # Objective: minimize variance of log|X(f)|^2
        # This should minimize sidelobes in autoconv
        
        lr = 0.05
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        
        best_local = x.copy()
        best_local_score = metric(x)
        
        for it in range(500):
            if time.time() > deadline - 10:
                break
            
            # Compute current power spectrum
            size = 1
            while size < 2 * N - 1:
                size *= 2
            fx = np.fft.rfft(x, size)
            power = np.abs(fx) ** 2
            # Only consider non-zero frequency bins (exclude DC and Nyquist for stability)
            valid = np.arange(1, size // 2)
            log_power = np.log(power[valid] + 1e-10)
            
            # Target: flat spectrum
            target = np.mean(log_power)
            # Gradient of variance w.r.t. x
            # d/dx Var(log|X|^2) = 2 * E[(log|X|^2 - mean) * d(log|X|^2)/dx]
            # d log|X(f)|^2 / dx = 2 * Re(conj(X(f)) * dX(f)/dx) / |X(f)|^2
            # dX(f)/dx_j = exp(-2*pi*i*f*j/size) for bin f
            # This is complex; use numerical gradient for simplicity
            
            # Numerical gradient
            grad = np.zeros_like(x)
            step = 1e-4
            for j in range(N):
                if time.time() > deadline - 10:
                    break
                x_plus = x.copy()
                x_plus[j] += step
                x_minus = x.copy()
                x_minus[j] -= step
                
                fx_plus = np.fft.rfft(x_plus, size)
                power_plus = np.abs(fx_plus) ** 2
                log_power_plus = np.log(power_plus[valid] + 1e-10)
                var_plus = np.var(log_power_plus)
                
                fx_minus = np.fft.rfft(x_minus, size)
                power_minus = np.abs(fx_minus) ** 2
                log_power_minus = np.log(power_minus[valid] + 1e-10)
                var_minus = np.var(log_power_minus)
                
                grad[j] = (var_plus - var_minus) / (2 * step)
            
            # Adam update
            t_adam = it + 1
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** t_adam)
            v_hat = v / (1 - beta2 ** t_adam)
            x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
            
            # Ensure sum is not too small
            s = np.sum(x)
            if abs(s) < 1.0:
                x += (1.0 - s) / N
            
            # Track best
            sc = metric(x)
            if sc < best_local_score:
                best_local_score = sc
                best_local = x.copy()
        
        if best_local_score < best_score:
            best_score = best_local_score
            best_x = best_local.copy()
    
    # Phase 2: Local refinement with random perturbations
    x = best_x.copy()
    current_score = best_score
    
    for it in range(2000):
        if time.time() > deadline - 5:
            break
        
        # Random perturbation
        scale = 0.05 * (1 - it / 2000)
        mask = np.random.rand(N) < 0.1
        perturb = np.random.randn(N) * scale
        x_new = x + mask * perturb
        
        s_new = np.sum(x_new)
        if abs(s_new) < 1e-10:
            continue
        
        sc_new = metric(x_new)
        if sc_new < current_score:
            x = x_new
            current_score = sc_new
            if current_score < best_score:
                best_score = current_score
                best_x = x.copy()
    
    # Phase 3: Try thresholding to ±1 and see if it helps
    # The task says "signed step heights" which could be ±1
    x_thresh = np.sign(best_x)
    # Ensure not all same sign
    if np.all(x_thresh == 1):
        x_thresh[::2] = -1
    elif np.all(x_thresh == -1):
        x_thresh[::2] = 1
    
    s_thresh = np.sum(x_thresh)
    if abs(s_thresh) > 1e-10:
        c_thresh = autoconv_linear(x_thresh)
        score_thresh = np.max(np.abs(c_thresh[1:])) / (s_thresh * s_thresh)
        if score_thresh < best_score:
            best_score = score_thresh
            best_x = x_thresh.copy()
    
    # Return the best solution
    return best_x
