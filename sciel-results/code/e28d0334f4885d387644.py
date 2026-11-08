"""
Structural change: Replace local gradient-based optimization and population-based
DE with a Simulated Annealing (SA) search.
SA uses a probabilistic acceptance criterion to escape local optima, which is
effective for the non-convex landscape of this problem. It is a single-trajectory
method, unlike DE, and does not require gradients, unlike L-BFGS.
"""
import numpy as np
import time

N = 50
M = 2 * N - 1          # len(g) = 99
W = 1.0 / (M + 1)      # 1/100
TIME_LIMIT = 150.0
RNG = np.random.RandomState(50)

def _build_y(g):
    y = np.zeros(M + 2)
    y[1:-1] = g
    return y

def compute_value(h):
    """Exact evaluator metric for given heights (defensive)."""
    h = np.asarray(h, dtype=float).ravel()
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if not np.all(np.isfinite(h)) or np.any(h < -1e-12):
        return 0.0
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def construct():
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 10.0)  # 10s safety margin

    # Initialize with a good heuristic profile
    x = np.linspace(0, N - 1, N)
    # Gaussian profile centered
    h_curr = np.exp(-0.5 * ((x - N / 2) / (N / 4)) ** 2)
    h_curr = np.maximum(0.0, h_curr)
    
    curr_val = compute_value(h_curr)
    best_h = h_curr.copy()
    best_val = curr_val

    # Simulated Annealing Parameters
    T_start = 1.0
    T_end = 1e-4
    
    # We will run until time limit.
    # Estimate steps: 
    # If we do 100,000 steps in 140s, that's ~1.4ms per step.
    # Python overhead for compute_value is likely higher.
    # Let's aim for 50,000 steps to be safe, but break on time.
    steps = 50000
    alpha = (T_end / T_start) ** (1.0 / steps)
    
    T = T_start
    
    # Track stagnation for potential restarts? 
    # Simple approach: just run.
    
    for i in range(steps):
        if time.time() > deadline:
            break
            
        # Generate neighbor
        # Perturb a random subset of indices
        n_perturb = RNG.randint(3, 8)
        indices = RNG.choice(N, n_perturb, replace=False)
        
        h_new = h_curr.copy()
        # Additive Gaussian noise
        # Scale: 0.5 is a reasonable step size for heights around 1
        noise = RNG.normal(0, 0.5, n_perturb)
        h_new[indices] += noise
        h_new = np.maximum(0.0, h_new)
        
        new_val = compute_value(h_new)
        
        # Acceptance criterion
        if new_val > curr_val:
            accept = True
        else:
            delta = new_val - curr_val
            if T > 0:
                prob = np.exp(delta / T)
            else:
                prob = 0.0
            accept = (RNG.random_sample() < prob)
            
        if accept:
            h_curr = h_new
            curr_val = new_val
            if curr_val > best_val:
                best_val = curr_val
                best_h = h_curr.copy()
        
        # Cool down
        T *= alpha
        
    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()

if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
