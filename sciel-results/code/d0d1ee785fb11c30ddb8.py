import numpy as np
import time
import scipy.optimize as opt

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s/150s kill

    def compute_metric(h):
        """Compute normalized max of autoconvolution: max(f*f) / (sum(h))^2"""
        h = np.asarray(h, dtype=np.float64)
        total = np.sum(h)
        if total == 0:
            return float('inf')
        # Autoconvolution via FFT
        size = 2 * N
        H = np.fft.rfft(h, n=size)
        acorr = np.fft.irfft(H * np.conj(H), n=size)
        peak = np.max(acorr)
        return peak / (total * total)

    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def smooth_objective(x, alpha):
        """
        Objective function for optimization.
        x: log-heights
        alpha: sharpness parameter for max approximation
        Returns: (value, gradient)
        """
        h = softmax(x)
        # Compute auto-correlation
        size = 2 * N
        H = np.fft.rfft(h, n=size)
        power = np.abs(H) ** 2
        acorr = np.fft.irfft(power, n=size)
        
        # Smooth max: (1/alpha) * log(sum(exp(alpha * acorr)))
        # To avoid overflow, shift by max
        max_acorr = np.max(acorr)
        log_sum_exp = max_acorr + (1.0 / alpha) * np.log(np.sum(np.exp(alpha * (acorr - max_acorr))))
        
        value = log_sum_exp
        
        # Gradient calculation
        w = np.exp(alpha * (acorr - max_acorr))
        w /= np.sum(w)
        
        # Gradient of sum(w_k * acorr_k) wrt h
        # grad_h = 2 * (w * h) (circular convolution)
        W = np.fft.rfft(w, n=size)
        H_h = np.fft.rfft(h, n=size)
        conv_wh = np.fft.irfft(W * H_h, n=size)[:N]
        
        grad_h = 2.0 * conv_wh
        
        # Gradient wrt x: h = softmax(x)
        S = np.dot(grad_h, h)
        grad_x = h * (grad_h - S)
        
        return value, grad_x

    def optimize_single(alpha_seq, init_x):
        best_h = None
        best_val = float('inf')
        
        x = init_x.copy()
        
        for alpha in alpha_seq:
            if time.time() > deadline:
                break
            
            # Optimize with current alpha
            res = opt.minimize(
                smooth_objective, 
                x, 
                args=(alpha,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 50, 'ftol': 1e-12}
            )
            x = res.x
            h = softmax(x)
            val = compute_metric(h)
            
            if val < best_val:
                best_val = val
                best_h = h.copy()
                
        return best_h, best_val

    # Multiple initializations
    np.random.seed(42)
    inits = [
        np.zeros(N), # Uniform
        np.random.randn(N),
        np.random.rand(N),
        np.sin(np.linspace(0, 2*np.pi, N)),
        np.exp(-np.linspace(0, 1, N)**2)
    ]
    
    best_h = None
    best_m = float('inf')
    
    alphas = [10, 50, 200, 1000]
    
    for init in inits:
        if time.time() > deadline:
            break
        h_opt, m_opt = optimize_single(alphas, init)
        if h_opt is not None and m_opt < best_m:
            best_m = m_opt
            best_h = h_opt
    
    # Fallback to uniform if optimization fails or is worse
    uniform_h = np.ones(N) / N
    uniform_m = compute_metric(uniform_h)
    
    if best_h is None or uniform_m < best_m:
        return list(uniform_h)
        
    return list(best_h)

# EVOLVE-BLOCK-END
