"""Model-and-optimize bootstrap: parameterize step heights and optimize the objective."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Construct a sequence of N step heights minimizing:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)
    
    Strategy:
    1. Generate diverse initial points.
    2. Use smooth surrogate (log-sum-exp) + L-BFGS-B gradient optimization.
    3. Final exact evaluation.
    """
    start_time = time.time()
    DEADLINE = 145  # 5s safety margin before 150s kill
    
    best_value = np.inf
    best_h = np.ones(N).astype(float)
    
    def evaluate(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf, False
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        valid = np.all(np.isfinite(h)) and abs(s) > 1e-12
        return val, valid
    
    # First evaluate the seed
    val_seed, _ = evaluate(best_h)
    best_value = val_seed
    
    rng = np.random.RandomState(42)
    
    def smooth_objective_and_grad(h, beta):
        """
        Smooth surrogate for |2N * max(conv) / sum(h)^2|.
        Vectorized gradient computation.
        """
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10, np.zeros_like(h)
        
        conv = np.convolve(h, h)
        
        # Log-sum-exp for numerical stability
        max_conv = np.max(conv)
        shifted = beta * (conv - max_conv)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        
        if sum_exp == 0:
            soft_max = max_conv
            weights = np.zeros_like(conv)
            weights[np.argmax(conv)] = 1.0
        else:
            logsumexp = max_conv + np.log(sum_exp) / beta
            soft_max = logsumexp
            weights = exp_shifted / sum_exp
        
        # f = 2N * soft_max / s^2
        f = 2.0 * N * soft_max / (s * s)
        
        # Vectorized gradient of soft_max w.r.t. h
        # grad_soft_max[k] = 2 * sum_i weights[i] * h[i-k] for valid i,k
        # This is a cross-correlation-like operation
        # Use FFT for efficiency: grad_soft_max = 2 * ifft(fft(weights) * conj(fft(h)))
        # But we need only the first N entries of the full convolution
        
        # Alternative: use np.convolve with 'valid' mode won't work directly
        # Let's use the fact that grad_soft_max[k] = 2 * (weights * h)[k] where * is correlation
        # Actually: grad_soft_max[k] = 2 * sum_{i=k}^{k+N-1} weights[i] * h[i-k]
        # This is the cross-correlation of weights and h at lag k
        
        # Efficient approach: pad and use FFT
        n_fft = 1 << (2 * N - 1).bit_length()
        W = np.fft.rfft(weights, n=n_fft)
        H = np.fft.rfft(h, n=n_fft)
        # Cross-correlation: ifft(conj(W) * H) gives sum_i W[i] * H[i+k]
        # We want sum_i weights[i] * h[i-k] which is ifft(W * conj(H))
        corr = np.fft.irfft(W * np.conj(H), n=n_fft)
        # corr[k] = sum_i weights[i] * h[i-k] for k=0..N-1
        grad_soft_max = 2.0 * corr[:N]
        
        # Gradient of f
        grad_f = (2.0 * N / (s * s)) * grad_soft_max - (4.0 * N * soft_max / (s ** 3))
        
        return f, grad_f
    
    # Generate initial points
    init_points = []
    
    # Seed
    init_points.append(np.ones(N).astype(float))
    
    # Random uniform
    for trial in range(200):
        if time.time() - start_time > DEADLINE:
            break
        h = rng.uniform(-1, 1, N)
        init_points.append(h)
    
    # Random normal
    for scale in [0.5, 1.0, 2.0, 3.0]:
        for trial in range(80):
            if time.time() - start_time > DEADLINE:
                break
            h = rng.normal(0, scale, N)
            init_points.append(h)
    
    # Alternating
    h_alt = np.ones(N)
    h_alt[1::2] = -1
    init_points.append(h_alt)
    
    # Cosine
    t = np.linspace(0, 1, N, endpoint=False)
    for freq in [1, 2, 3, 5, 7, 10, 15, 20]:
        init_points.append(np.cos(2 * np.pi * freq * t))
    
    # Sine
    for freq in [1, 2, 3, 5, 7, 10, 15, 20]:
        init_points.append(np.sin(2 * np.pi * freq * t))
    
    # Block structures: positive block, negative block
    for block_size in [N//4, N//3, N//2]:
        h_block = np.zeros(N)
        h_block[:block_size] = 1.0
        h_block[N-block_size:] = -1.0
        init_points.append(h_block)
        
        h_block2 = np.zeros(N)
        h_block2[N//4:3*N//4] = 1.0
        init_points.append(h_block2)
    
    # Gaussian-like bumps
    for center in [N//4, N//2, 3*N//4]:
        for width in [N//8, N//4, N//6]:
            x = np.arange(N)
            h_gauss = np.exp(-0.5 * ((x - center) / width) ** 2)
            init_points.append(h_gauss)
            h_gauss_neg = -h_gauss
            init_points.append(h_gauss_neg)
    
    # Two-level: random blocks
    for n_blocks in [2, 3, 4, 5]:
        block_len = N // n_blocks
        h_two = np.zeros(N)
        for b in range(n_blocks):
            sign = 1.0 if b % 2 == 0 else -1.0
            h_two[b*block_len:(b+1)*block_len] = sign
        init_points.append(h_two)
    
    # Evaluate all init points and keep top candidates
    evals = []
    for h in init_points:
        if time.time() - start_time > DEADLINE:
            break
        val, valid = evaluate(h)
        if valid:
            evals.append((val, h))
    
    evals.sort(key=lambda x: x[0])
    
    # Take top candidates for optimization
    n_optimize = min(15, len(evals))
    
    # For each top candidate, run L-BFGS with increasing beta
    for beta in [10, 50, 100, 200, 500, 1000]:
        if time.time() - start_time > DEADLINE:
            break
        
        for idx in range(n_optimize):
            if time.time() - start_time > DEADLINE:
                break
            
            h0 = evals[idx][1].copy()
            
            try:
                result = minimize(
                    lambda h, b=beta: smooth_objective_and_grad(h, b)[0],
                    h0,
                    jac=lambda h, b=beta: smooth_objective_and_grad(h, b)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                )
                
                if result.success or result.fun < 1e10:
                    h_opt = result.x
                    val, valid = evaluate(h_opt)
                    if valid and val < best_value:
                        best_value = val
                        best_h = h_opt.copy()
                        
            except Exception:
                pass
    
    # Final polish: coordinate descent with adaptive step size
    if time.time() - start_time < DEADLINE:
        h_curr = best_h.copy()
        lr = 0.05
        stagnation = 0
        
        for iteration in range(500):
            if time.time() - start_time > DEADLINE:
                break
            improved = False
            for coord in rng.permutation(N):
                if time.time() - start_time > DEADLINE:
                    break
                # Try several step sizes
                for step_mult in [1.0, 0.5, 0.2, 2.0]:
                    h_try = h_curr.copy()
                    h_try[coord] += lr * step_mult * rng.uniform(-1, 1)
                    val_try, valid = evaluate(h_try)
                    if valid and val_try < best_value:
                        best_value = val_try
                        best_h = h_try.copy()
                        h_curr = h_try
                        improved = True
                        break
                if best_value < 0.8:
                    break
            if best_value < 0.8:
                break
            if not improved:
                stagnation += 1
                lr *= 0.7
                if stagnation > 10:
                    break
            else:
                stagnation = 0
    
    return best_h
# EVOLVE-BLOCK-END
