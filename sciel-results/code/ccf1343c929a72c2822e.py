"""Minimize the normalized maximum of the autoconvolution of a nonnegative
step function on 600 equal intervals over [-1/4, 1/4].

Strategy:
We minimize the L1-normalized peak of the autoconvolution: max(c) / (sum h)^2.
For a uniform function with sum h = S, autoconv peak = S^2/N, giving ratio 1/N.
We search for profiles that have a flatter autoconvolution than uniform.

The key insight: functions with broader support and smoother shape tend to have
more spread-out autoconvolutions, reducing the peak relative to the total mass.

We use:
1. Gradient descent in log-space (to maintain non-negativity)
2. Simulated annealing for global exploration
3. Multiple restarts from different initializations
"""

import time
import math
import numpy as np

N = 600
DEADLINE = 145.0  # safety margin before 150s kill

def construct():
    t0 = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - t0)
    
    def autoconv_fft(h):
        """Fast autoconvolution using FFT."""
        n = len(h)
        size = 2 * n
        f = np.fft.rfft(h, n=size)
        c = np.fft.irfft(f * f, n=size)
        # For non-negative h, the relevant part is c[0:n]
        # But full convolution has 2n-1 elements. The peak could be anywhere.
        # Actually for non-negative functions, the peak of autoconv is at lag 0
        # (the center), so c[n-1:n] region. Let me think...
        # Full convolution of h with h has length 2n-1.
        # With FFT padding to size=2n, we get circular convolution.
        # c[0:n-1] is the "wrap-around" part, c[n-1:2n-1] is the direct part.
        # For non-negative h, the maximum is at the center (lag 0), which is c[n-1].
        # But to be safe, let's compute the full linear convolution.
        return np.convolve(h, h, mode='full')
    
    def objective(h):
        """L1-normalized peak of autoconvolution."""
        s = np.sum(h)
        if s <= 1e-12:
            return 1e12
        c = autoconv_fft(h)
        peak = np.max(c)
        return peak / (s * s)
    
    def log_objective(log_h):
        """Objective in log-space."""
        h = np.exp(log_h)
        return objective(h)
    
    # Initializations
    def init_uniform():
        return np.ones(N) / N  # normalized to sum=1
    
    def init_gaussian(sigma):
        x = np.linspace(-0.25, 0.25, N)
        h = np.exp(-x**2 / (2 * sigma**2))
        s = np.sum(h)
        return h / s
    
    def init_cosine(p):
        x = np.linspace(-0.25, 0.25, N)
        theta = np.pi * (x + 0.25) / 0.25  # 0 to pi
        h = np.cos(theta / 2) ** p
        h = np.clip(h, 0, None)
        s = np.sum(h)
        if s > 1e-12:
            h = h / s
        else:
            h = np.ones(N) / N
        return h
    
    def init_random(rng):
        h = rng.random(N)
        s = np.sum(h)
        if s > 1e-12:
            h = h / s
        else:
            h = np.ones(N) / N
        return h
    
    def gradient_descent(h, lr=0.01, n_steps=500, log_space=True):
        """Gradient descent in log-space or direct space."""
        if log_space:
            # Work in log-space to maintain non-negativity
            # Add small epsilon to avoid log(0)
            eps = 1e-10
            h_safe = np.maximum(h, eps)
            log_h = np.log(h_safe)
            
            best_log_h = log_h.copy()
            best_obj = log_objective(log_h)
            
            for step in range(n_steps):
                if time_left() < 5.0:
                    break
                
                # Compute gradient numerically in log-space
                # Use central differences for a few coordinates
                # For speed, use forward differences with small step
                dx = 0.01
                grad = np.zeros_like(log_h)
                
                # Sample a subset of coordinates for gradient estimation
                idx = np.random.choice(N, size=min(N, 100), replace=False)
                for i in idx:
                    lp = log_h.copy()
                    lm = log_h.copy()
                    lp[i] += dx
                    lm[i] -= dx
                    grad[i] = (log_objective(lp) - log_objective(lm)) / (2 * dx)
                
                # Extend gradient estimate to all coordinates using the pattern
                # This is a rough estimate but should work
                # Actually let's just do coordinate descent style
                # Simpler: just do a few full gradient steps with random coordinates
                
                # Update
                lr_step = lr / (1 + step * 0.01)  # decay learning rate
                log_h = log_h - lr_step * grad
                log_h = np.clip(log_h, -20, 20)  # keep values reasonable
                
                h_new = np.exp(log_h)
                obj_new = log_objective(log_h)
                
                if obj_new < best_obj:
                    best_obj = obj_new
                    best_log_h = log_h.copy()
            
            h_out = np.exp(best_log_h)
            s = np.sum(h_out)
            if s > 1e-12:
                h_out = h_out / s
            return h_out, best_obj
        
        else:
            # Direct space gradient descent with projection
            best_h = h.copy()
            best_obj = objective(h)
            
            for step in range(n_steps):
                if time_left() < 5.0:
                    break
                
                # Numerical gradient
                dx = 1e-4
                grad = np.zeros(N)
                for i in range(0, N, max(1, N // 50)):
                    hp = h.copy()
                    hm = h.copy()
                    hp[i] += dx
                    hm[i] = max(hm[i] - dx, 0)
                    grad[i] = (objective(hp) - objective(hm)) / dx
                
                lr_step = lr / (1 + step * 0.01)
                h_new = h - lr_step * grad
                h_new = np.maximum(h_new, 0)
                
                obj_new = objective(h_new)
                if obj_new < best_obj:
                    best_obj = obj_new
                    best_h = h_new.copy()
                
                h = h_new
            
            s = np.sum(best_h)
            if s > 1e-12:
                best_h = best_h / s
            return best_h, best_obj
    
    def simulated_annealing(h, n_iter=2000, T_start=0.01, T_end=0.0001):
        """Simulated annealing in log-space."""
        eps = 1e-10
        h_safe = np.maximum(h, eps)
        log_h = np.log(h_safe).copy()
        
        best_log_h = log_h.copy()
        best_obj = log_objective(log_h)
        current_obj = best_obj
        
        for i in range(n_iter):
            if time_left() < 5.0:
                break
            
            T = T_start * (T_end / T_start) ** (i / n_iter)
            
            # Random perturbation
            idx = np.random.randint(0, N, size=10)
            delta = np.random.normal(0, 0.1, size=10)
            
            log_h_new = log_h.copy()
            log_h_new[idx] += delta
            log_h_new = np.clip(log_h_new, -20, 20)
            
            obj_new = log_objective(log_h_new)
            
            # Metropolis criterion
            if obj_new < current_obj or np.random.random() < math.exp(-(obj_new - current_obj) / T):
                log_h = log_h_new
                current_obj = obj_new
                
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_log_h = log_h_new.copy()
        
        h_out = np.exp(best_log_h)
        s = np.sum(h_out)
        if s > 1e-12:
            h_out = h_out / s
        return h_out, best_obj
    
    # Main search
    rng = np.random.default_rng(12345)
    
    best_h = np.ones(N) / N
    best_score = objective(best_h)
    
    # Try various initializations
    initializations = []
    initializations.append(init_uniform())
    for sigma in [0.05, 0.1, 0.15, 0.2, 0.25]:
        initializations.append(init_gaussian(sigma))
    for p in [0.5, 1.0, 2.0, 4.0]:
        initializations.append(init_cosine(p))
    for _ in range(5):
        initializations.append(init_random(rng))
    
    for h0 in initializations:
        if time_left() < 10.0:
            break
        
        # Evaluate initial
        obj0 = objective(h0)
        
        # Gradient descent in log-space
        h1, obj1 = gradient_descent(h0, lr=0.05, n_steps=300, log_space=True)
        if obj1 < best_score:
            best_score = obj1
            best_h = h1.copy()
        
        # Simulated annealing from the best of initial and GD result
        h_start = h1 if obj1 < obj0 else h0
        h2, obj2 = simulated_annealing(h_start, n_iter=1500)
        if obj2 < best_score:
            best_score = obj2
            best_h = h2.copy()
        
        # Another GD pass from the best
        if time_left() > 10.0:
            h3, obj3 = gradient_descent(h2 if obj2 < obj1 else h1, lr=0.03, n_steps=200, log_space=True)
            if obj3 < best_score:
                best_score = obj3
                best_h = h3.copy()
    
    # Final refinement
    if time_left() > 5.0:
        h_final, obj_final = gradient_descent(best_h, lr=0.01, n_steps=100, log_space=True)
        if obj_final < best_score:
            best_score = obj_final
            best_h = h_final.copy()
    
    # Normalize to sum = 1
    s = np.sum(best_h)
    if s > 1e-12:
        best_h = best_h / s
    
    return [float(v) for v in best_h]
