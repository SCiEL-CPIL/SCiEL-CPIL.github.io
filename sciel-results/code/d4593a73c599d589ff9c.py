"""
Strategy: Direct Multi-Start Local Optimization with Parametric Initialization and Fast Gradient-Free Refinement.

Rationale:
1. The objective function is smooth and well-behaved (ratio of sums of polynomials).
2. Genetic algorithms are often inefficient for low-dimensional (N=50) continuous optimization compared to direct local methods, especially when good initialization is available.
3. We use a set of diverse parametric initializers (Gaussian, Triangular, Sine, etc.) to cover the search space.
4. For each initializer, we perform a robust local optimization using a Nelder-Mead style simplex method or a custom coordinate descent with line search, which is faster and more reliable for this specific structure than GA.
5. We prioritize speed by using vectorized operations where possible and keeping the local search tight.
"""

import numpy as np
import time
import random

N = 50

def evaluate(h):
    """
    Compute the value for a given set of step heights.
    """
    h = np.asarray(h, dtype=np.float64)
    h = np.clip(h, 0, None)
    
    g = np.convolve(h, h)
    m = len(g)
    w = 1.0 / (m + 1)
    
    # y = [0, g..., 0]
    # Instead of constructing y, we compute the sum directly.
    # Sum over i=0..m of (w/3) * (y[i]^2 + y[i]y[i+1] + y[i+1]^2)
    # y[0]=0, y[1]=g[0], ..., y[m]=g[m-1], y[m+1]=0
    
    # Terms:
    # i=0: 0 + 0 + g[0]^2 = g[0]^2
    # i=m: g[m-1]^2 + 0 + 0 = g[m-1]^2
    # i=1..m-1: let k = i-1. y[i]=g[k], y[i+1]=g[k+1].
    # Term: g[k]^2 + g[k]g[k+1] + g[k+1]^2 for k=0..m-2
    
    g_sq = g**2
    cross = g[:-1] * g[1:]
    
    # Sum of g[k]^2 + g[k]g[k+1] + g[k+1]^2 for k=0..m-2
    # = (g[0]^2 + ... + g[m-2]^2) + cross + (g[1]^2 + ... + g[m-1]^2)
    # = (sum(g_sq) - g[m-1]^2) + cross + (sum(g_sq) - g[0]^2)
    
    sum_g_sq = np.sum(g_sq)
    sum_terms = (sum_g_sq - g[-1]**2) + np.sum(cross) + (sum_g_sq - g[0]**2)
    
    # Add boundary terms: g[0]^2 + g[-1]^2
    total_sum = sum_terms + g[0]**2 + g[-1]**2
    
    l2 = (w / 3.0) * total_sum
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 * norm_inf == 0:
        return -1.0
    return l2 / (norm_1 * norm_inf)

def local_optimize(h0, time_limit):
    """
    Perform local optimization starting from h0.
    Uses a custom coordinate descent with adaptive step size and line search.
    """
    h = np.array(h0, dtype=np.float64)
    best_h = h.copy()
    best_val = evaluate(h)
    
    t0 = time.time()
    step_sizes = [0.1, 0.05, 0.01, 0.005, 0.001]
    current_step_idx = 0
    
    # Simplex-like or Coordinate Descent
    # Coordinate descent is simple and effective for this smooth function.
    # We cycle through coordinates.
    
    n_iter = 0
    max_iter = 10000
    
    while time.time() - t0 < time_limit and n_iter < max_iter:
        n_iter += 1
        improved = False
        
        # Randomly permute coordinates for this iteration
        coords = np.random.permutation(N)
        
        for i in coords:
            if time.time() - t0 > time_limit:
                break
            
            # Try increasing and decreasing
            for sign in [1, -1]:
                # Adaptive step size
                step = step_sizes[current_step_idx]
                new_h = h.copy()
                new_h[i] = max(0, h[i] + sign * step)
                
                new_val = evaluate(new_h)
                
                if new_val > best_val:
                    h = new_h
                    best_h = h.copy()
                    best_val = new_val
                    improved = True
                    # Reset step size index if we improved significantly?
                    # Or just keep going.
                else:
                    # If no improvement, try smaller step?
                    # For now, just try the other sign.
                    pass
            
            # If no improvement in this coordinate with current step,
            # maybe reduce step size globally?
            # Simple strategy: if we didn't improve in this coordinate,
            # it's fine. We'll come back to it.
        
        # If no improvement in the entire cycle, reduce step size
        if not improved:
            current_step_idx = min(current_step_idx + 1, len(step_sizes) - 1)
            if current_step_idx == len(step_sizes) - 1:
                # Try a few more iterations with smallest step before breaking
                # Or break if we've been stuck for a while.
                # Let's just break if we're at the smallest step and no improvement.
                # Actually, let's continue for a bit to allow for noise.
                pass
    
    return best_h, best_val

def construct():
    deadline = time.time() + 115.0  # Safety margin
    
    # Generate diverse initial points
    x = np.linspace(-1, 1, N)
    
    initializers = []
    
    # 1. Uniform
    initializers.append(np.ones(N))
    
    # 2. Gaussian centered
    initializers.append(np.exp(-4 * x**2))
    
    # 3. Triangular
    initializers.append(np.abs(1 - 2 * np.linspace(0, 1, N)))
    
    # 4. Sine bump
    initializers.append(np.sin(np.pi * np.linspace(0, 1, N)))
    
    # 5. Bimodal
    initializers.append(np.exp(-10*(x-0.4)**2) + np.exp(-10*(x+0.4)**2))
    
    # 6. Flat top
    h = np.zeros(N)
    h[10:40] = 1.0
    initializers.append(h)
    
    # 7. Random smooth
    for _ in range(10):
        h = np.random.uniform(0, 1, N)
        # Smooth it
        kernel = np.ones(5)/5
        h = np.convolve(h, kernel, mode='same')
        initializers.append(h)
    
    # 8. Random sparse
    for _ in range(10):
        h = np.zeros(N)
        idx = np.random.choice(N, size=10, replace=False)
        h[idx] = np.random.uniform(0.5, 1.0, 10)
        initializers.append(h)
    
    best_h = None
    best_val = -np.inf
    
    # Allocate time for each initializer
    # We have ~115s. 20 initializers. ~5s each?
    # But local optimization can be fast. Let's do a time-budgeted loop.
    
    for h0 in initializers:
        if time.time() > deadline - 10:
            break
            
        # Give each a fair chance, but prioritize early ones if they are good?
        # No, just run them all with a fixed time budget per run, or until deadline.
        # Let's use a dynamic time budget: remaining_time / remaining_initializers
        remaining_initializers = len(initializers) - initializers.index(h0)
        time_left = deadline - time.time()
        time_per_run = time_left / max(1, remaining_initializers)
        
        # Cap time per run to avoid one bad run eating all time
        time_per_run = min(time_per_run, 5.0)
        
        h_opt, val_opt = local_optimize(h0, time_per_run)
        
        if val_opt > best_val:
            best_val = val_opt
            best_h = h_opt
    
    # If we have time left, refine the best one further
    if time.time() < deadline - 5 and best_h is not None:
        remaining_time = deadline - time.time() - 2
        if remaining_time > 0:
            h_opt, val_opt = local_optimize(best_h, remaining_time)
            if val_opt > best_val:
                best_val = val_opt
                best_h = h_opt
    
    if best_h is None:
        return list(np.ones(N))
        
    return list(best_h)

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
