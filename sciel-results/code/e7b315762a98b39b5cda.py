"""Optimized program for signed autocorrelation minimum peak-to-sum-ratio."""
import numpy as np
import time

N = 400

def compute_objective(x):
    """Compute max(|autoconv|) / (sum(x))^2"""
    n_fft = 2 * N
    X = np.fft.rfft(x, n=n_fft)
    autoconv = np.fft.irfft(X * np.conj(X), n=n_fft)[:N]
    peak = np.max(np.abs(autoconv))
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    return peak / (s * s)

def construct():
    start_time = time.time()
    deadline = 140.0
    
    best_x = None
    best_val = 1e10
    rng = np.random.default_rng(42)
    
    def try_candidate(x):
        nonlocal best_x, best_val
        val = compute_objective(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
        return val
    
    # All ones baseline
    try_candidate(np.ones(N))
    
    # Continuous optimization via projected gradient descent
    # Strategy: minimize max|autoconv| with fixed sum constraint
    # Use soft-max (log-sum-exp) to smooth the max
    
    def soft_max_penalty(x, temperature):
        """Smooth approximation of max|autoconv| using log-sum-exp"""
        n_fft = 2 * N
        X = np.fft.rfft(x, n=n_fft)
        autoconv = np.fft.irfft(X * np.conj(X), n=n_fft)[:N]
        abs_ac = np.abs(autoconv)
        # Log-sum-exp for smooth max
        max_ac = np.max(abs_ac)
        if max_ac < 1e-10:
            return 0.0
        shifted = abs_ac - max_ac
        log_sum_exp = max_ac + np.log(np.sum(np.exp(shifted / temperature)))
        return log_sum_exp * temperature
    
    def gradient_soft_max(x, temperature):
        """Gradient of soft-max penalty w.r.t. x"""
        n_fft = 2 * N
        X = np.fft.rfft(x, n=n_fft)
        autoconv = np.fft.irfft(X * np.conj(X), n=n_fft)[:N]
        abs_ac = np.abs(autoconv)
        sign_ac = np.sign(autoconv)
        
        max_ac = np.max(abs_ac)
        if max_ac < 1e-10:
            return np.zeros_like(x)
        
        shifted = abs_ac - max_ac
        weights = np.exp(shifted / temperature)
        weights /= np.sum(weights)
        
        # Gradient of max|ac| w.r.t. x: sum over lag k of w_k * sign(ac_k) * d(ac_k)/dx
        # ac_k = sum_i x_i * x_{i+k}
        # d(ac_k)/dx_j = x_{j+k} + x_{j-k} (with boundary handling)
        
        grad = np.zeros_like(x)
        for k in range(N):
            if weights[k] < 1e-10:
                continue
            # ac_k = sum_{i=0}^{N-1-k} x_i * x_{i+k}
            # d(ac_k)/dx_j = x_{j+k} (if j+k < N) + x_{j-k} (if j-k >= 0)
            if k == 0:
                # ac_0 = sum x_i^2, d/dx_j = 2*x_j
                grad += 2.0 * weights[k] * sign_ac[k] * x
            else:
                # For j from 0 to N-1-k: contribution from x_j * x_{j+k}
                # d/dx_j = x_{j+k}
                grad[:N-k] += weights[k] * sign_ac[k] * x[k:]
                # For j from k to N-1: contribution from x_{j-k} * x_j
                # d/dx_j = x_{j-k}
                grad[k:] += weights[k] * sign_ac[k] * x[:N-k]
        
        return grad
    
    def projected_gradient_descent(x_init, target_sum, max_iters, lr):
        """PGD with projection onto sum constraint"""
        x = x_init.copy()
        current_val = compute_objective(x)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute gradient of soft-max penalty
            temp = max(0.1, 1.0 / (it + 1))  # Annealing temperature
            grad = gradient_soft_max(x, temp)
            
            # Project gradient: remove component along all-ones direction
            grad_mean = np.mean(grad)
            grad_proj = grad - grad_mean
            
            # Gradient step
            x_new = x - lr * grad_proj
            
            # Project onto sum constraint
            x_new = x_new + (target_sum - np.sum(x_new)) / N
            
            # Evaluate
            new_val = compute_objective(x_new)
            if new_val < current_val:
                x = x_new
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_x = x.copy()
        
        return x, current_val
    
    # Try continuous optimization from different starting points
    for trial in range(5):
        if time.time() > deadline:
            break
        
        # Start from biased binary or continuous initialization
        if trial == 0:
            x_init = np.ones(N)
        elif trial == 1:
            x_init = rng.choice([-1.0, 1.0], size=N, p=[0.3, 0.7])
        elif trial == 2:
            x_init = rng.normal(0.5, 0.3, size=N)
        elif trial == 3:
            x_init = np.sin(np.linspace(0, 2*np.pi, N)) * 0.5 + 0.5
        else:
            x_init = rng.uniform(0, 1, size=N)
        
        target_sum = N  # Fix sum to isolate peak minimization
        
        # Adaptive learning rate
        lr = 0.1
        max_iters = 200
        
        x_opt, val_opt = projected_gradient_descent(x_init, target_sum, max_iters, lr)
        try_candidate(x_opt)
        
        # Also try binary local search from the continuous solution
        x_bin = np.sign(x_opt)
        x_bin[x_bin == 0] = 1.0
        val_bin = try_candidate(x_bin)
        
        # Short binary polish
        for it in range(100):
            if time.time() > deadline:
                break
            idx = rng.integers(0, N)
            old = x_bin[idx]
            x_bin[idx] = -old
            new_val = compute_objective(x_bin)
            if new_val < val_bin:
                val_bin = new_val
                if val_bin < best_val:
                    best_val = val_bin
                    best_x = x_bin.copy()
            else:
                x_bin[idx] = old
    
    # Also keep the original binary random search for comparison
    for trial in range(100):
        if time.time() > deadline:
            break
        
        p_plus = rng.uniform(0.6, 0.98)
        x = rng.choice([-1.0, 1.0], size=N, p=[1-p_plus, p_plus])
        
        val = try_candidate(x)
        
        for it in range(200):
            if time.time() > deadline:
                break
            idx = rng.integers(0, N)
            old = x[idx]
            x[idx] = -old
            new_val = compute_objective(x)
            if new_val < val:
                val = new_val
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
            else:
                x[idx] = old
    
    if best_x is None:
        best_x = np.ones(N)
    
    return best_x

# EVOLVE-BLOCK-START
# (construct function defined above)
# EVOLVE-BLOCK-END
