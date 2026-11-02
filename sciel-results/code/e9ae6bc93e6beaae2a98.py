import numpy as np
import time

N = 400
L = 2 * N - 1

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    # Use next power of 2 for FFT efficiency
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return np.inf
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _smooth_obj_grad(x, tau=0.1):
    """
    Compute the log-sum-exp approximation of max(|ac|) / (sum x)^2 
    and its gradient.
    
    Returns:
    val: the smooth objective value
    grad: the gradient w.r.t. x
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return np.inf, np.zeros_like(x)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    # Soft-max approximation of max(abs_ac)
    # L_tau = tau * log(sum(exp(abs_ac / tau)))
    # To avoid overflow, subtract max
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0, np.zeros_like(x)
        
    shifted = (abs_ac - m) / tau
    shifted = np.clip(shifted, -50, 0) # Numerical stability
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    if sum_exp < 1e-15:
        return m / (s * s), np.zeros_like(x)
        
    # Weights for the soft-max
    weights = exp_shifted / sum_exp
    
    # Soft max value
    soft_max = m + tau * np.log(sum_exp)
    
    # Objective value
    val = soft_max / (s * s)
    
    # Gradient calculation
    # d/dx [ soft_max / s^2 ] = (grad_soft_max * s^2 - soft_max * 2 * s * grad_s) / s^4
    # grad_s = 1 (since s = sum x)
    # So, grad_obj = grad_soft_max / s^2 - 2 * soft_max / s^3
    
    # grad_soft_max = sum_k weights_k * d/dx |ac_k|
    # d/dx |ac_k| = sign(ac_k) * d/dx ac_k
    # d/dx ac_k = convolve(x, e_k) + correlate(x, e_k) ... essentially
    # Let v_k = weights_k * sign(ac_k)
    # Then grad_soft_max_j = sum_k v_k * (x_{k-j} + x_{j-k})
    
    v = weights * np.sign(ac)
    
    # Compute gradient using FFT
    # Term 1: sum_k v_k x_{k-j} (Cross-correlation of v and x)
    # Term 2: sum_k v_k x_{j-k} (Convolution of v and x)
    
    L_fft = 1
    while L_fft < L + N:
        L_fft *= 2
        
    v_pad = np.zeros(L_fft)
    v_pad[:L] = v
    
    x_pad = np.zeros(L_fft)
    x_pad[:N] = x
    
    V = np.fft.rfft(v_pad, n=L_fft)
    X = np.fft.rfft(x_pad, n=L_fft)
    
    # corr1[j] = sum_k v_k x_{k-j}
    # This is cross-correlation. In FFT: IFFT(V * conj(X))
    corr1 = np.fft.irfft(V * np.conj(X), n=L_fft)
    
    # conv2[j] = sum_k v_k x_{j-k}
    # This is convolution. In FFT: IFFT(V * X)
    conv2 = np.fft.irfft(V * X, n=L_fft)
    
    grad_soft_max = corr1[:N] + conv2[:N]
    
    grad_obj = grad_soft_max / (s * s) - 2.0 * soft_max / (s * s * s)
    
    return val, grad_obj

def _solve_optimal_psd(N_fft, max_iters=1000, lr=0.05, tau_init=1.0):
    """
    Solve for the optimal PSD S minimizing the peak of its inverse FFT.
    S is non-negative and sums to 1.
    """
    S = np.ones(N_fft) / N_fft
    tau = tau_init
    best_S = S.copy()
    best_val = np.inf
    
    for i in range(max_iters):
        c = np.fft.ifft(S).real
        
        abs_c = np.abs(c)
        m = np.max(abs_c)
        if m < 1e-15:
            break
            
        shifted = (abs_c - m) / tau
        shifted = np.clip(shifted, -50, 0)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        
        if sum_exp < 1e-15:
            break
            
        weights = exp_shifted / sum_exp
        soft_max = m + tau * np.log(sum_exp)
        
        g = weights * np.sign(c)
        grad_S = np.real(np.fft.ifft(g)) / N_fft
        
        S_new = S - lr * grad_S
        
        # Projection onto probability simplex
        v = np.sort(S_new)[::-1]
        cssv = np.cumsum(v)
        rho = np.nonzero(v * np.arange(1, N_fft + 1) > (cssv - 1))[0]
        if len(rho) == 0:
            theta = 0
        else:
            rho_idx = rho[-1]
            theta = (cssv[rho_idx] - 1) / (rho_idx + 1.0)
        S_new = np.maximum(S_new - theta, 0)
        
        if soft_max < best_val:
            best_val = soft_max
            best_S = S_new.copy()
            
        S = S_new
        tau *= 0.99
        if tau < 0.01:
            tau = 0.01

    return best_S

def _phase_retrieval(S_target, N, max_iters=200):
    """
    Find x such that |FFT(x)|^2 approx S_target.
    """
    N_fft = 1
    while N_fft < 2 * N:
        N_fft *= 2
        
    # Ensure S_target is compatible with N_fft
    # If S_target length is different, we need to resize it.
    # But we call it with the same N_fft.
    
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, size=N)
    
    x_pad = np.zeros(N_fft)
    x_pad[:N] = x
    
    for i in range(max_iters):
        X = np.fft.fft(x_pad)
        mag_target = np.sqrt(S_target)
        phase = np.exp(1j * np.angle(X))
        X_new = mag_target * phase
        x_pad = np.fft.ifft(X_new).real
        x = x_pad[:N]
        
        s = np.sum(x)
        if abs(s) > 1e-6:
            x = x / s

    return x

def construct():
    START_TIME = time.time()
    TIME_LIMIT = 145
    
    def _time_ok():
        return (time.time() - START_TIME) < TIME_LIMIT

    N_fft = 1
    while N_fft < 2 * N:
        N_fft *= 2
        
    S_opt = _solve_optimal_psd(N_fft, max_iters=500, lr=0.05, tau_init=1.0)
    
    if not _time_ok():
        return np.ones(N)

    x_candidate = _phase_retrieval(S_opt, N, max_iters=200)
    
    best_x = x_candidate
    best_val = _objective(best_x)
    
    if _time_ok():
        m = np.zeros_like(best_x)
        v = np.zeros_like(best_x)
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        t_adam = 0
        lr = 0.01
        tau = 0.1
        patience = 0
        max_patience = 100
        
        for step in range(2000):
            if not _time_ok():
                break
                
            val, grad = _smooth_obj_grad(best_x, tau)
            
            if not np.isfinite(val):
                break
                
            if val < best_val:
                best_val = val
                patience = 0
            else:
                patience += 1
                if patience > max_patience:
                    break
                    
            t_adam += 1
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** t_adam)
            v_hat = v / (1 - beta2 ** t_adam)
            
            best_x = best_x - lr * m_hat / (np.sqrt(v_hat) + epsilon)
            
            s = np.sum(best_x)
            if abs(s) > 1e-6:
                best_x = best_x / s

    return best_x
