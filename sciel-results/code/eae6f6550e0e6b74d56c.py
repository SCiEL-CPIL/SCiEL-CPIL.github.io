import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin

    def project_to_polytope(h, S):
        """Project h onto { x : sum(x) = S, 0 <= x_i <= 1 }."""
        # h is clipped to [0,1] initially in the loop, but we handle general case
        # We need to find lambda such that sum(clip(h - lambda, 0, 1)) = S
        
        # If S is outside [0, N], it's infeasible, but S=N/2 is always feasible.
        
        # Binary search for lambda
        lo = -1.0
        hi = 2.0
        # Check if clipping is needed
        # If sum(h) == S, return h
        if abs(np.sum(h) - S) < 1e-12:
             return np.clip(h, 0.0, 1.0)
             
        # Determine bounds for lambda
        # If sum(h) > S, we need to decrease h, so lambda > 0
        # If sum(h) < S, we need to increase h, so lambda < 0
        
        for _ in range(50):
            mid = (lo + hi) / 2.0
            proj = np.clip(h - mid, 0.0, 1.0)
            val = np.sum(proj)
            if val > S:
                lo = mid
            else:
                hi = mid
        
        lambda_star = (lo + hi) / 2.0
        return np.clip(h - lambda_star, 0.0, 1.0)

    def compute_corr_fft(h):
        """Compute circular autocorrelation using FFT."""
        fft_h = np.fft.fft(h, n=2 * N)
        corr = np.fft.ifft(fft_h * np.conj(fft_h)).real[:N]
        return corr

    def max_overlap(h):
        """Compute the actual maximum overlap."""
        corr = compute_corr_fft(h)
        # Overlap at shift k = N/2 - corr[k]
        # We want to minimize max_k overlap
        overlaps = N / 2.0 - corr
        return np.max(overlaps)

    def smooth_min_corr_value_and_grad(h, beta):
        """
        Returns (value, grad) for the smooth approximation of -min(Corr).
        Value is L_beta = (1/beta) * ln(sum(exp(-beta * C)))
        Grad is -conv(w_prime, h)
        """
        corr = compute_corr_fft(h)
        
        # Stable log-sum-exp for sum(exp(-beta * C))
        min_c = np.min(corr)
        # We want to minimize -min(C). 
        # Let f(C) = -min(C). 
        # Smooth approx: L_beta = (1/beta) ln sum exp(-beta C_k).
        # As beta -> inf, this -> -min(C_k).
        
        # To avoid overflow/underflow, shift by min_c? 
        # exp(-beta * C_k) = exp(-beta * (C_k - min_c) + beta * min_c)
        # = exp(beta * min_c) * exp(-beta * (C_k - min_c))
        # Since C_k >= min_c, C_k - min_c >= 0, so -beta(C_k - min_c) <= 0.
        # Max exponent is 0 (when C_k = min_c).
        
        exps = np.exp(-beta * (corr - min_c))
        sum_exps = np.sum(exps)
        
        # Value
        val = min_c - np.log(sum_exps) / beta
        
        # Weights w_k proportional to exp(-beta * C_k)
        # w_k = exps / sum_exps
        weights = exps / sum_exps
        
        # Construct w_prime for gradient
        # grad_j = - [ 2 w_0 h_j + 2 sum_{k!=0} w_k h_{j-k} ]
        # Let w_prime[0] = 2 * weights[0]
        # w_prime[k] = 2 * weights[k] for k != 0
        # Then grad = - conv(w_prime, h)
        
        w_prime = 2.0 * weights
        
        # Compute convolution via FFT
        fft_wp = np.fft.fft(w_prime, n=2 * N)
        fft_h = np.fft.fft(h, n=2 * N)
        conv = np.fft.ifft(fft_wp * fft_h).real[:N]
        
        grad = -conv
        
        return val, grad

    def optimize_from_init(h0):
        h = h0.copy()
        h = project_to_polytope(h, N / 2.0)
        
        best_local_score = max_overlap(h)
        best_local_h = h.copy()
        
        beta = 10.0
        alpha = 0.05
        max_steps = 500
        patience = 50
        no_improve = 0
        
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
            
            val, grad = smooth_min_corr_value_and_grad(h, beta)
            
            # Gradient step
            # We are minimizing val.
            h_new = h - alpha * grad
            h_new = project_to_polytope(h_new, N / 2.0)
            
            # Check improvement in actual score
            new_actual_score = max_overlap(h_new)
            
            if new_actual_score < best_local_score:
                best_local_score = new_actual_score
                best_local_h = h_new.copy()
                no_improve = 0
            else:
                no_improve += 1
            
            # Check if smooth objective improved to decide on step
            new_val, _ = smooth_min_corr_value_and_grad(h_new, beta)
            
            if new_val < val:
                h = h_new
                # Adaptive beta and alpha
                if step % 20 == 0:
                    beta *= 1.2
                alpha *= 0.99
            else:
                alpha *= 0.5
                if alpha < 1e-8:
                    break
            
            if no_improve > patience:
                break
                
        return best_local_h, best_local_score

    best_h_global = np.full(N, 0.5)
    best_score_global = max_overlap(best_h_global)
    
    rng = np.random.default_rng(42)
    
    # Generate initializations
    inits = []
    
    # 1. Uniform
    inits.append(np.full(N, 0.5))
    
    # 2. Random
    for _ in range(20):
        h = rng.random(N)
        inits.append(h)
        
    # 3. Sinusoidal
    for freq in [1, 2, 3, 5, 7, 11]:
        for amp in [0.2, 0.4, 0.45]:
            h = 0.5 + amp * np.sin(2 * np.pi * freq * np.arange(N) / N)
            inits.append(h)
            
    # 4. Binary-like
    for _ in range(10):
        h = (rng.random(N) > 0.5).astype(float)
        inits.append(h)
        
    # 5. Sawtooth
    for period in [3, 4, 5, 7, 10, 13, 17]:
        for phase in range(period):
            h = np.zeros(N)
            for i in range(N):
                pos = (i - phase) % period
                if pos < period // 2:
                    h[i] = 1.0
                else:
                    h[i] = 0.0
            inits.append(h)

    # Optimize from each initialization
    for h0 in inits:
        if time.time() > deadline - 10.0:
            break
        
        h_opt, score_opt = optimize_from_init(h0)
        
        if score_opt < best_score_global:
            best_score_global = score_opt
            best_h_global = h_opt.copy()

    # Final cleanup
    final_h = project_to_polytope(best_h_global, N / 2.0)
    
    # Ensure sum is exactly N/2
    diff = N / 2.0 - np.sum(final_h)
    if abs(diff) > 1e-10:
        mask = (final_h > 1e-6) & (final_h < 1 - 1e-6)
        if np.any(mask):
            final_h[mask] += diff / np.sum(mask)
            final_h = np.clip(final_h, 0.0, 1.0)
    
    return final_h
