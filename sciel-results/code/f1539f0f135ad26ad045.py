import numpy as np
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin for 120s limit
    
    # Initial candidate: Uniform
    x = np.ones(N)
    best_x = x.copy()
    
    # Helper to calculate score (raw value to minimize)
    def get_score(vec):
        # Normalize to sum=1
        s = np.sum(vec)
        if s == 0: return float('inf')
        v = vec / s
        # Convolution
        conv = np.convolve(v, v, mode='full')
        return np.max(conv)
        
    best_score = get_score(best_x)
    
    # Try a few deterministic shapes
    candidates = []
    
    # 1. Uniform (already in best_x)
    
    # 2. Cosine window
    t = np.linspace(0, np.pi, N)
    x_cos = np.cos(t/2)**2
    x_cos[x_cos < 0] = 0
    if np.sum(x_cos) > 0:
        candidates.append(x_cos)
        
    # 3. Gaussian
    t = np.linspace(-3, 3, N)
    x_gauss = np.exp(-t**2)
    candidates.append(x_gauss)
    
    # 4. Triangular
    t = np.linspace(-1, 1, N)
    x_tri = np.maximum(0, 1 - np.abs(t))
    candidates.append(x_tri)
    
    for c in candidates:
        if time.time() - start_time > timeout: break
        s = get_score(c)
        if s < best_score:
            best_score = s
            best_x = c.copy()
            
    # Local Search / Hill Climbing
    # We perturb a few random indices
    # To be efficient, we can use a small number of iterations
    # Since N=600, full convolution is O(N^2) or O(N log N).
    # np.convolve is O(N^2) for small N, but 600 is small.
    # 600^2 = 360,000 ops. Very fast.
    
    # Let's do a simple random restart hill climb
    np.random.seed(42)
    max_iter = 5000
    step_size = 0.1
    
    for i in range(max_iter):
        if time.time() - start_time > timeout:
            break
            
        # Create a perturbed version
        x_curr = best_x.copy()
        
        # Pick a few random indices to perturb
        num_perturb = 5
        idxs = np.random.choice(N, num_perturb, replace=False)
        
        # Add random noise
        for idx in idxs:
            x_curr[idx] += np.random.normal(0, step_size * best_x[idx] + 1e-6)
            if x_curr[idx] < 0:
                x_curr[idx] = 0
                
        s = get_score(x_curr)
        if s < best_score:
            best_score = s
            best_x = x_curr.copy()
            
    return list(best_x)
# EVOLVE-BLOCK-END
