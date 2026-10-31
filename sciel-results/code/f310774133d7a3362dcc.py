import numpy as np
import time

N = 50
DEADLINE = 150.0
START_TIME = time.time()

def construct():
    # Return a sequence of 50 nonnegative heights
    # We will search for a shape that maximizes the ratio
    # Ratio = sum(h^2) / (sum(h) * max(conv(h)))
    
    def score(h):
        h = np.array(h)
        l1 = np.sum(h)
        if l1 == 0:
            return 0.0
        l2_sq = np.sum(h**2)
        # Compute autoconvolution
        # Using FFT for speed, or direct since N=50 is small
        # Direct:
        # c_k = sum h_i h_{i+k}
        # Max is at k=0 for non-negative h
        # But let's be safe and compute max
        # Since N=50, O(N^2) is fine
        max_conv = 0.0
        for k in range(-N, N):
            s = 0.0
            for i in range(N):
                j = i + k
                if 0 <= j < N:
                    s += h[i] * h[j]
            if s > max_conv:
                max_conv = s
        if max_conv == 0:
            return 0.0
        return l2_sq / (l1 * max_conv)

    best_h = [1.0] * N
    best_score = score(best_h)
    
    # Try some parametric shapes
    # 1. Gaussian
    # 2. Triangle
    # 3. Cosine bump
    
    shapes = []
    
    # Generate Gaussians with different widths
    for sigma in np.linspace(1.0, 10.0, 20):
        x = np.arange(N) - N/2
        h = np.exp(-x**2 / (2*sigma**2))
        h = h / np.max(h) # Normalize max to 1
        shapes.append(h)
        
    # Generate Triangles
    for base in np.linspace(5, N, 20):
        center = N/2
        h = np.zeros(N)
        half = base/2
        for i in range(N):
            dist = abs(i - center)
            if dist <= half:
                h[i] = 1 - dist/half
        if np.max(h) > 0:
            h = h / np.max(h)
            shapes.append(h)
            
    # Generate Cosine bumps
    for base in np.linspace(5, N, 20):
        center = N/2
        h = np.zeros(N)
        half = base/2
        for i in range(N):
            dist = abs(i - center)
            if dist <= half:
                h[i] = 0.5 * (1 + np.cos(np.pi * dist / half))
        if np.max(h) > 0:
            h = h / np.max(h)
            shapes.append(h)

    # Evaluate all
    for h in shapes:
        s = score(h)
        if s > best_score:
            best_score = s
            best_h = h

    # Local refinement on the best shape found
    # Simple hill climbing on the values
    h = np.array(best_h)
    current_score = best_score
    
    # If time permits, do a few steps of optimization
    # Since N=50, we can try random perturbations
    if time.time() - START_TIME < DEADLINE - 5:
        for _ in range(500):
            if time.time() - START_TIME > DEADLINE - 5:
                break
            # Perturb a few elements
            h_new = h.copy()
            idx = np.random.randint(0, N, 5)
            h_new[idx] += np.random.normal(0, 0.1, 5)
            h_new = np.maximum(h_new, 0)
            if np.max(h_new) > 0:
                h_new = h_new / np.max(h_new)
            s_new = score(h_new)
            if s_new > current_score:
                h = h_new
                current_score = s_new

    return list(h)
