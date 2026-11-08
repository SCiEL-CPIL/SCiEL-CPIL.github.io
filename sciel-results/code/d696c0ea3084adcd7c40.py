import numpy as np
import time

N = 400
DEADLINE = 145.0

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    C3 = max_t |autoconv(t)| where autoconv is the convolution of f with itself.
    """
    M = 2 * N  # fine grid points
    dx = 0.5 / M
    
    # Vectorized construction of fine grid
    m_indices = np.arange(M)
    i_indices = (m_indices * N // M)
    i_indices = np.clip(i_indices, 0, N-1)
    f_fine = signed_step[i_indices]
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    c3 = np.max(np.abs(autoconv))
    
    return c3

def _compute_c3_soft(signed_step, beta=50.0):
    """
    Compute a smooth surrogate of C3 using log-sum-exp.
    This is differentiable and helps gradient-based optimization.
    """
    M = 2 * N
    dx = 0.5 / M
    
    m_indices = np.arange(M)
    i_indices = (m_indices * N // M)
    i_indices = np.clip(i_indices, 0, N-1)
    f_fine = signed_step[i_indices]
    
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    # Soft max of |autoconv|
    abs_autoconv = np.abs(autoconv)
    scaled = beta * abs_autoconv
    max_val = np.max(scaled)
    log_sum_exp = max_val + np.log(np.sum(np.exp(scaled - max_val)))
    soft_c3 = log_sum_exp / beta
    
    return soft_c3

def _soft_c3_from_autoconv(autoconv, beta=50.0):
    """Compute soft C3 from precomputed autoconvolution."""
    abs_autoconv = np.abs(autoconv)
    scaled = beta * abs_autoconv
    max_val = np.max(scaled)
    log_sum_exp = max_val + np.log(np.sum(np.exp(scaled - max_val)))
    return log_sum_exp / beta

def _compute_autoconv(signed_step):
    """Compute autoconvolution for a signed step function."""
    M = 2 * N
    dx = 0.5 / M
    
    m_indices = np.arange(M)
    i_indices = (m_indices * N // M)
    i_indices = np.clip(i_indices, 0, N-1)
    f_fine = signed_step[i_indices]
    
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    return autoconv

def _gradient_soft_c3(x, beta=50.0):
    """
    Compute analytic gradient of soft C3 with respect to continuous vector x.
    x is in [-1, 1]^N.
    """
    M = 2 * N
    dx = 0.5 / M
    
    # Map x to fine grid
    m_indices = np.arange(M)
    i_indices = (m_indices * N // M)
    i_indices = np.clip(i_indices, 0, N-1)
    f_fine = x[i_indices]
    
    # Autoconvolution
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    # Gradient of log-sum-exp of |autoconv|
    abs_autoconv = np.abs(autoconv)
    scaled = beta * abs_autoconv
    max_val = np.max(scaled)
    
    # Stable log-sum-exp and weights
    exp_scaled = np.exp(scaled - max_val)
    sum_exp = np.sum(exp_scaled)
    weights = exp_scaled / sum_exp
    
    # d(soft_c3)/d(autoconv_k) = beta * weights_k * sign(autoconv_k)
    # Note: sign(0) is 0, which is fine.
    grad_autoconv = beta * weights * np.sign(autoconv)
    
    # Chain rule: d(autoconv)/d(f_fine)
    # autoconv is circular convolution (via FFT).
    # autoconv[k] = sum_j f[j] f[(k-j) mod M] * dx
    # Actually, fft_f * conj(fft_f) is the FFT of the circular autocorrelation.
    # Let h be the impulse response of the operator that maps f to autoconv.
    # autoconv = f * f (circular).
    # d(autoconv)/d(f_j) = 2 * (f * delta_j) = 2 * shifted_f.
    # Specifically, grad_f[j] = sum_k grad_autoconv[k] * d(autoconv[k])/d(f[j])
    # d(autoconv[k])/d(f[j]) = 2 * f[(k-j) mod M] * dx
    
    # We can compute this efficiently using FFT or direct correlation.
    # Let's use the property that grad_f = 2 * dx * (grad_autoconv * f) (circular correlation)
    # Wait, autoconv[k] = sum_i f[i] f[k-i].
    # d autoconv[k] / d f[j] = f[k-j] + f[j-k] = 2 f[k-j] (since f is real, f[j-k]=f[k-j] in circular sense? No, f is not necessarily symmetric, but autoconv is symmetric).
    # Actually, for real f, autoconv[k] = sum_i f[i] f[i-k].
    # d autoconv[k] / d f[j] = f[j-k] + f[j+k] ? No.
    # autoconv[k] = sum_i f[i] f[i-k].
    # If i=j, term is f[j] f[j-k]. Derivative wrt f[j] is f[j-k].
    # If i-k=j => i=j+k, term is f[j+k] f[j]. Derivative wrt f[j] is f[j+k].
    # So d autoconv[k] / d f[j] = f[j-k] + f[j+k].
    # This is 2 * (f * grad_autoconv) at index j?
    # Let's check: sum_k grad_autoconv[k] (f[j-k] + f[j+k])
    # = sum_k grad_autoconv[k] f[j-k] + sum_k grad_autoconv[k] f[j+k]
    # Let l = j-k => k = j-l. Sum_l grad_autoconv[j-l] f[l]. This is correlation.
    # Let m = j+k => k = m-j. Sum_m grad_autoconv[m-j] f[m].
    # Since grad_autoconv is symmetric (autoconv is symmetric, weights are symmetric), grad_autoconv[k] = grad_autoconv[-k].
    # So both sums are the same.
    # Grad_f[j] = 2 * dx * sum_k grad_autoconv[k] f[j-k]
    # This is 2 * dx * (grad_autoconv * f) (circular correlation).
    
    # Using FFT for speed:
    # Correlation of A and B is IFFT(FFT(A) * conj(FFT(B)))
    fft_grad_auto = np.fft.fft(grad_autoconv, n=M)
    fft_f = np.fft.fft(f_fine, n=M) # f_fine is length M, but we need to map back to N
    # Wait, f_fine is length M. We need gradient wrt x (length N).
    # grad_f_fine = 2 * dx * ifft(fft(grad_autoconv) * conj(fft(f_fine))).real
    
    # Note: autoconv was computed with n=2*M FFT to avoid wrap-around?
    # In _compute_autoconv, we used n=2*M. This implies linear convolution?
    # If we use n=2*M, it's linear convolution.
    # Linear convolution gradient:
    # autoconv[k] = sum_{i=0}^{M-1} f[i] f[k-i] (with zero padding).
    # d autoconv[k] / d f[j] = f[k-j] (if 0<=k-j<M) + f[j-k] (if 0<=j-k<M).
    # This is equivalent to linear correlation.
    
    # Let's stick to the discrete gradient definition for safety, but vectorized.
    # grad_x[i] = sum_{k} grad_autoconv[k] * d(autoconv[k])/d(x[i])
    # Since x is mapped to f_fine via i_indices, d(x[i])/d(f_fine[m]) is 1 if i_indices[m]==i else 0.
    # So grad_x[i] = sum_{m: i_indices[m]==i} grad_f_fine[m]
    
    # Compute grad_f_fine (length M)
    # Using FFT for linear correlation requires zero padding.
    # We already have f_fine length M.
    # grad_f_fine = 2 * dx * ifft(fft(grad_autoconv, n=2*M) * conj(fft(f_fine, n=2*M))).real[:M]
    
    fft_grad_auto_2M = np.fft.fft(grad_autoconv, n=2*M)
    fft_f_2M = np.fft.fft(f_fine, n=2*M)
    grad_f_fine = np.fft.ifft(fft_grad_auto_2M * np.conj(fft_f_2M)).real[:M] * (2 * dx)
    
    # Map back to x
    grad_x = np.zeros(N)
    np.add.at(grad_x, i_indices, grad_f_fine)
    
    return grad_x

def construct():
    start_time = time.time()
    
    # Initialize with all ones
    best_step = np.ones(N, dtype=np.int8)
    best_c3 = _compute_c3(best_step)
    
    np.random.seed(42)
    
    # Try a few random initializations to find a good starting point for continuous opt
    for trial in range(10):
        if time.time() - start_time > DEADLINE - 10.0:
            break
        random_step = np.random.choice([-1, 1], size=N).astype(np.int8)
        c3_val = _compute_c3(random_step)
        if c3_val < best_c3:
            best_step = random_step.copy()
            best_c3 = c3_val
    
    # Continuous Optimization Phase
    # Start from best discrete solution
    x = best_step.astype(np.float64)
    
    beta = 50.0
    step_size = 0.1
    
    # Adaptive step size gradient descent
    for grad_iter in range(1000):
        if time.time() - start_time > DEADLINE - 20.0:
            break
        
        # Compute gradient
        grad = _gradient_soft_c3(x, beta=beta)
        
        # Normalize gradient to avoid huge steps
        norm_grad = np.linalg.norm(grad)
        if norm_grad < 1e-8:
            break
            
        # Backtracking line search or fixed step with decay
        # Simple decay
        current_step = step_size / (1.0 + 0.01 * grad_iter)
        
        # Projected Gradient Step
        x_new = x - current_step * grad
        x_new = np.clip(x_new, -1.0, 1.0)
        
        # Evaluate soft C3 to check improvement (optional, can be expensive)
        # For speed, we can just take the step.
        # But to ensure convergence, let's check every 10 steps.
        if grad_iter % 10 == 0:
             soft_val_new = _compute_c3_soft(x_new, beta=beta)
             soft_val_old = _compute_c3_soft(x, beta=beta)
             if soft_val_new > soft_val_old:
                 # Step too large, reduce step size
                 step_size *= 0.5
                 # Don't update x
                 continue
             else:
                 # Accept step
                 x = x_new
                 # Update best discrete if thresholding gives better result
                 # We threshold at the end, but checking periodically helps
                 # Thresholding: sign(x)
                 cand = np.sign(x)
                 cand[cand == 0] = 1
                 cand_c3 = _compute_c3(cand)
                 if cand_c3 < best_c3:
                     best_c3 = cand_c3
                     best_step = cand.astype(np.int8)
                     # Reset step size on improvement
                     step_size = 0.1
        
        # Update x if we didn't check or if we accepted
        if grad_iter % 10 != 0:
             x = x_new
             
    # Final Hard Thresholding
    final_step = np.sign(x)
    final_step[final_step == 0] = 1
    final_step = final_step.astype(np.int8)
    final_c3 = _compute_c3(final_step)
    
    if final_c3 < best_c3:
        best_c3 = final_c3
        best_step = final_step
    
    # Discrete Polish Phase
    # Use the remaining time for local search on the discrete solution
    # This fixes any local errors introduced by thresholding
    
    no_improve_count = 0
    patience = 1000
    current_step = best_step.copy()
    current_c3 = best_c3
    
    while time.time() - start_time < DEADLINE - 5.0:
        if no_improve_count >= patience:
            break
            
        # Random flip
        idx = np.random.randint(N)
        old_val = current_step[idx]
        current_step[idx] = -old_val
        
        new_c3 = _compute_c3(current_step)
        
        if new_c3 < current_c3:
            current_c3 = new_c3
            if current_c3 < best_c3:
                best_c3 = current_c3
                best_step = current_step.copy()
            no_improve_count = 0
        else:
            current_step[idx] = old_val
            no_improve_count += 1
            
    return best_step
