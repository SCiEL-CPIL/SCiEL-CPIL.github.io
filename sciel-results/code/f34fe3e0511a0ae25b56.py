import numpy as np
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    n = 24
    
    # Precompute roots of unity for fast evaluation
    # We use a sufficiently dense grid to approximate the sup norm.
    # 1024 points is a good balance for speed and accuracy in local search.
    # For the final check, we might use more, but for search, 1024 is fast.
    m = 2048
    theta = np.linspace(0, 2 * np.pi, m, endpoint=False)
    z = np.exp(1j * theta)
    Z = np.vstack([z ** k for k in range(n)])
    
    def evaluate(coeffs):
        # coeffs is a numpy array of shape (n,) with values +/-1
        # Compute P(z) for all z in the grid
        vals = Z @ coeffs
        return np.max(np.abs(vals))
    
    # Initial best
    # Start with a random configuration or a known good one (e.g., Rudin-Shapiro)
    # Let's try a few random starts and keep the best.
    
    best_coeffs = np.ones(n, dtype=np.int8)
    best_score = evaluate(best_coeffs)
    
    # Try a few deterministic starts
    starts = []
    # All ones
    starts.append(np.ones(n, dtype=np.int8))
    # Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < n:
        p, q = p + q, p + [-c for c in q]
    starts.append(np.array(p[:n], dtype=np.int8))
    # Alternating
    starts.append(np.array([1 if i % 2 == 0 else -1 for i in range(n)], dtype=np.int8))
    
    for s in starts:
        sc = evaluate(s)
        if sc < best_score:
            best_score = sc
            best_coeffs = s.copy()
    
    # Local Search with Simulated Annealing
    # We will perform random restarts if time permits, or just one long annealing run.
    # Given the small size (24 bits), a single good annealing run from a good start is often enough.
    # Let's do multiple short runs or one long run.
    
    current_coeffs = best_coeffs.copy()
    current_score = best_score
    
    # Annealing parameters
    T_start = 0.5
    T_end = 0.001
    alpha = 0.995  # Cooling rate
    
    T = T_start
    steps_per_temp = 1000
    
    # Precompute indices for random flipping
    indices = np.arange(n)
    
    while time.time() < deadline:
        for _ in range(steps_per_temp):
            if time.time() > deadline:
                break
                
            # Pick a random index to flip
            idx = np.random.randint(n)
            
            # Flip
            old_val = current_coeffs[idx]
            current_coeffs[idx] = -old_val
            
            new_score = evaluate(current_coeffs)
            
            # Acceptance criterion
            if new_score < current_score or np.random.random() < math.exp((current_score - new_score) / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_coeffs = current_coeffs.copy()
            else:
                # Revert
                current_coeffs[idx] = old_val
                
        T *= alpha
        if T < T_end:
            # Restart with a new random configuration or keep current?
            # Let's try a few random restarts
            current_coeffs = np.random.choice([1, -1], size=n).astype(np.int8)
            current_score = evaluate(current_coeffs)
            T = T_start * 0.5  # Lower start temp for restarts
            
    return list(best_coeffs)
