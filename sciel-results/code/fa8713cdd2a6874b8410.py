import numpy as np
import time

N = 400
DEADLINE = 120.0

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    The metric is: max_k |conv(h, h)[k]| / (sum_i |h_i|)^2
    """
    conv = np.convolve(signed_step, signed_step, mode='full')
    sum_abs_h = np.sum(np.abs(signed_step))
    if sum_abs_h < 1e-12:
        return float('inf')
    return np.max(np.abs(conv)) / (sum_abs_h ** 2)

def _fast_c3(signed_step):
    """
    Faster C3 computation using FFT for discrete step functions.
    """
    M = 4 * N
    dx = (0.5) / M
    
    f_fine = np.zeros(M)
    for i in range(N):
        start = int(i * M / N)
        end = int((i + 1) * M / N)
        f_fine[start:end] = signed_step[i]
    
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    sum_abs = np.sum(np.abs(signed_step))
    if sum_abs < 1e-12:
        return float('inf')
    
    return np.max(np.abs(autoconv)) / (sum_abs ** 2)

def _soft_c3_continuous(x, beta=20.0):
    """
    Compute soft C3 for continuous variables x in [-1, 1].
    Uses the metric: soft_max(|autoconv|) / (sum|x|)^2
    """
    M = 4 * N
    dx = (0.5) / M
    
    f_fine = np.zeros(M)
    for i in range(N):
        start = int(i * M / N)
        end = int((i + 1) * M / N)
        f_fine[start:end] = x[i]
    
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    # Soft max of |autoconv|
    ax = np.abs(autoconv)
    max_ax = np.max(ax)
    log_sum_exp = np.log(np.sum(np.exp(beta * (ax - max_ax)))) + beta * max_ax
    soft_max = log_sum_exp / beta
    
    # Sum of absolute values (smooth)
    sum_abs = np.sum(np.sqrt(x**2 + 1e-10))
    
    return soft_max / (sum_abs ** 2)

def _soft_c3_grad(x, beta=20.0):
    """
    Compute gradient of soft C3 with respect to x.
    """
    M = 4 * N
    dx = (0.5) / M
    
    f_fine = np.zeros(M)
    for i in range(N):
        start = int(i * M / N)
        end = int((i + 1) * M / N)
        f_fine[start:end] = x[i]
    
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    # Gradient of soft max w.r.t. autoconv
    ax = np.abs(autoconv)
    max_ax = np.max(ax)
    weights = np.exp(beta * (ax - max_ax))
    weights /= np.sum(weights)
    d_autoconv = weights * np.sign(autoconv)
    
    # d(autoconv)/d(f) via structure
    # autoconv[t] = sum_s f[s] f[s+t] * dx (with appropriate indexing)
    # For step function, use the discrete structure
    grad_x = np.zeros(N)
    interval_width = 0.5 / N
    
    # Use FFT-based gradient computation
    # autoconv = ifft(|F|^2) * dx
    # d(autoconv)/d(f) = 2 * ifft(F * conj(F) * something)
    # Actually, for autocorrelation: R[k] = sum_i f[i] f[i+k]
    # dR[k]/df[j] = f[j+k] + f[j-k] (with bounds)
    
    # Efficient computation using the fact that we only need the gradient
    # at the N step positions
    for k in range(-N+1, N):
        weight = d_autoconv[k + N - 1] if (k + N - 1) < len(d_autoconv) else 0.0
        if weight == 0.0:
            continue
        if k >= 0:
            for j in range(N - k):
                grad_x[j] += weight * f_fine[j + k] * interval_width
                grad_x[j + k] += weight * f_fine[j] * interval_width
        else:
            kk = -k
            for j in range(N - kk):
                grad_x[j] += weight * f_fine[j + kk] * interval_width
                grad_x[j + kk] += weight * f_fine[j] * interval_width
    
    # Gradient of 1/(sum_abs)^2 term
    sum_abs = np.sum(np.sqrt(x**2 + 1e-10))
    d_sum_abs = x / np.sqrt(x**2 + 1e-10)
    grad_x -= 2 * _soft_c3_continuous(x, beta) * d_sum_abs / (sum_abs ** 2)
    
    return grad_x

def _continuous_optimize(x0, beta=20.0, max_iter=80):
    """
    Run projected gradient descent on continuous variables in [-1, 1].
    """
    start_time = time.time()
    x = x0.copy()
    lr = 0.1
    
    for it in range(max_iter):
        if time.time() - start_time > 25.0:
            break
        
        obj = _soft_c3_continuous(x, beta)
        grad = _soft_c3_grad(x, beta)
        
        # Gradient step
        x_new = x - lr * grad
        
        # Project to [-1, 1]
        x_new = np.clip(x_new, -1.0, 1.0)
        
        # Check if we improved
        obj_new = _soft_c3_continuous(x_new, beta)
        
        if obj_new < obj:
            x = x_new
            lr = min(lr * 1.05, 0.5)
        else:
            lr = max(lr * 0.8, 0.01)
    
    return x, _soft_c3_continuous(x, beta)

def _discrete_from_continuous(x):
    """Convert continuous values to discrete ±1."""
    return np.where(x >= 0, 1.0, -1.0)

def _hill_climb(step, start_time, max_time=20.0):
    """
    Discrete hill-climbing refinement with simulated annealing.
    """
    best_step = step.copy()
    best_c3 = _compute_c3(best_step)
    no_improve = 0
    patience = 5000
    T = 0.1
    
    while time.time() - start_time < max_time:
        idx = np.random.randint(N)
        old_val = best_step[idx]
        best_step[idx] = -old_val
        
        new_c3 = _compute_c3(best_step)
        
        if new_c3 < best_c3:
            best_c3 = new_c3
            no_improve = 0
            T = max(T * 0.999, 0.01)
        else:
            # Accept with probability exp(-delta/T)
            delta = new_c3 - best_c3
            if np.random.rand() < np.exp(-delta / max(T, 1e-10)):
                best_c3 = new_c3
                no_improve = 0
            else:
                best_step[idx] = old_val
                no_improve += 1
        
        if no_improve >= patience:
            break
    
    return best_step, best_c3

def construct():
    start_time = time.time()
    
    np.random.seed(42)
    
    # Best result so far
    best_step = np.ones(N)
    best_c3 = _compute_c3(best_step)
    
    # Phase 1: Continuous optimization with diverse initializations
    initializations = [
        np.ones(N),
        -np.ones(N),
        np.random.choice([-1.0, 1.0], size=N),
        np.tile([-1.0, 1.0], N // 2),
        np.tile([1.0, 1.0, -1.0, -1.0], N // 4),
        np.tile([1.0, -1.0, -1.0, 1.0], N // 4),
        np.sin(2 * np.pi * np.arange(N) / N),  # sinusoidal
        np.cos(2 * np.pi * np.arange(N) / N),  # cosine
    ]
    
    for x0 in initializations:
        if time.time() - start_time > DEADLINE - 40.0:
            break
        
        try:
            x_opt, fun_val = _continuous_optimize(x0, beta=20.0, max_iter=60)
            
            # Convert to discrete
            step_candidate = _discrete_from_continuous(x_opt)
            c3_candidate = _compute_c3(step_candidate)
            
            if c3_candidate < best_c3:
                best_c3 = c3_candidate
                best_step = step_candidate.copy()
        except Exception:
            continue
    
    # Phase 2: Hill climbing refinement
    if time.time() - start_time < DEADLINE - 10.0:
        best_step, best_c3 = _hill_climb(best_step, start_time, max_time=DEADLINE - 5.0)
    
    # Final check
    final_c3 = _compute_c3(best_step)
    if final_c3 < best_c3:
        best_c3 = final_c3
    
    return best_step
