import numpy as np
import time
import sys

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin before 150s kill
    
    # Precompute constants
    # g = convolve(h, h) has length 2*N - 1 = 99
    # m = 99
    # y has length m + 2 = 101
    # Points: np.linspace(-0.5, 0.5, m + 2) -> spacing w = 1.0 / (m + 1) = 1/100
    # Wait, linspace(-0.5, 0.5, 101) has spacing (0.5 - (-0.5)) / (101 - 1) = 1.0 / 100 = 0.01
    m = 2 * N - 1
    w = 1.0 / (m + 1)
    
    def evaluate(h):
        """
        h: array of N nonnegative heights
        Returns: value = l2 / (norm_1 * norm_inf)
        """
        h = np.asarray(h, dtype=np.float64)
        # Ensure non-negative
        h = np.maximum(h, 0.0)
        
        g = np.convolve(h, h)
        m_local = len(g)
        # y = [0, g[0], ..., g[m-1], 0]
        y = np.zeros(m_local + 2)
        y[1:m_local+1] = g
        
        # l2 = sum_{i=0}^{m} (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
        # Note: the sum is over i in 0..m, where y has indices 0..m+1
        # So i goes from 0 to m, which is m+1 terms
        y_sq = y**2
        y_prod = y * np.roll(y, -1)  # y[i] * y[i+1]
        # But careful: roll shifts, so y_prod[i] = y[i] * y[i+1] for i < len(y)-1
        # For i = m, y[i] * y[i+1] = y[m] * y[m+1]
        # np.roll(y, -1) gives [y[1], y[2], ..., y[m+1], y[0]]
        # So y_prod[i] = y[i] * y[i+1] for i in 0..m-1, and y_prod[m] = y[m] * y[0] = y[m] * 0 = 0
        # Wait, y[m+1] = 0, so y[m] * y[m+1] = 0.
        # Let's just compute directly to be safe
        l2 = 0.0
        for i in range(m_local + 1):
            yi = y[i]
            yip1 = y[i+1]
            l2 += (w / 3.0) * (yi*yi + yi*yip1 + yip1*yip1)
        
        norm_1 = np.sum(np.abs(g)) / (m_local + 1)
        norm_inf = np.max(np.abs(g))
        
        if norm_1 * norm_inf == 0:
            return 0.0
        
        return l2 / (norm_1 * norm_inf)
    
    # Seed: 50 equal heights
    h_seed = np.ones(N)
    best_h = h_seed.copy()
    best_val = evaluate(best_h)
    
    # Simple local search / gradient-free optimization
    # Try a few random perturbations and keep the best
    # Use a coarse-to-fine grid or random restarts
    
    # Strategy: 
    # 1. Try symmetric profiles
    # 2. Try random perturbations with annealing
    # 3. Local refinement
    
    rng = np.random.RandomState(42)
    
    # Try some structured profiles first
    # Triangular: h[i] = 1 - |i - (N-1)/2| / ((N-1)/2)
    x = np.linspace(0, 1, N)
    profiles = [
        np.ones(N),
        1 - np.abs(x - 0.5) * 2,  # Triangular
        (1 - x**2),  # Parabolic
        np.cos(np.pi * x)**2,  # Cosine squared
        np.exp(-((x - 0.5)**2) / (2 * 0.1**2)),  # Gaussian
        np.exp(-((x - 0.5)**2) / (2 * 0.2**2)),  # Wider Gaussian
        np.exp(-((x - 0.5)**2) / (2 * 0.05**2)),  # Narrow Gaussian
    ]
    
    for p in profiles:
        p = np.maximum(p, 0)
        if np.max(p) == 0:
            continue
        p = p / np.max(p)  # Scale to max 1
        val = evaluate(p)
        if val > best_val:
            best_val = val
            best_h = p.copy()
    
    # Random search with local refinement
    # Use a simple hill-climbing approach
    step_size = 0.1
    num_iterations = 5000
    
    for it in range(num_iterations):
        if time.time() > deadline:
            break
        
        # Random perturbation
        h_new = best_h.copy()
        # Perturb a random subset
        num_perturb = rng.randint(1, 5)
        indices = rng.choice(N, num_perturb, replace=False)
        h_new[indices] += rng.uniform(-step_size, step_size, num_perturb)
        h_new = np.maximum(h_new, 0)
        
        val = evaluate(h_new)
        if val > best_val:
            best_val = val
            best_h = h_new.copy()
            # Decrease step size slightly
            step_size *= 0.98
        else:
            # Occasionally increase step size to escape local minima
            if it % 100 == 0:
                step_size *= 1.05
            step_size = min(step_size, 0.5)
    
    # Final refinement with smaller steps
    step_size = 0.01
    for it in range(2000):
        if time.time() > deadline:
            break
        
        h_new = best_h.copy()
        num_perturb = rng.randint(1, 3)
        indices = rng.choice(N, num_perturb, replace=False)
        h_new[indices] += rng.uniform(-step_size, step_size, num_perturb)
        h_new = np.maximum(h_new, 0)
        
        val = evaluate(h_new)
        if val > best_val:
            best_val = val
            best_h = h_new.copy()
            step_size *= 0.99
        else:
            if it % 200 == 0:
                step_size *= 1.02
            step_size = min(step_size, 0.1)
    
    # Convert to list of floats
    return [float(h) for h in best_h]
# EVOLVE-BLOCK-END
