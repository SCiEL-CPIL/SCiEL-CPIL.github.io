import numpy as np
import time

N = 50

def _ratio(h):
    """
    Compute the ratio (sum h^2) / (sum h * max autoconv)
    This is proportional to the continuous ratio.
    """
    h = np.asarray(h, dtype=float)
    if np.any(h < 0):
        return 0.0
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-12:
        return 0.0
    # Discrete autoconvolution max
    # c[k] = sum_i h[i] h[i-k]
    # Using correlate
    c = np.correlate(h, h, mode='full')
    ac_max = c.max()
    if ac_max < 1e-12:
        return 0.0
    return l2_sq / (l1 * ac_max)

def construct():
    # Generate candidate patterns
    candidates = []
    
    # 1. Single Spike (Theoretical Max)
    spike = np.zeros(N)
    spike[0] = 1.0
    candidates.append(spike)
    
    # 2. Uniform (Seed)
    uniform = np.ones(N)
    candidates.append(uniform)
    
    # 3. Two adjacent spikes
    two_spikes = np.zeros(N)
    two_spikes[0] = 1.0
    two_spikes[1] = 1.0
    candidates.append(two_spikes)
    
    # 4. Alternating spikes
    alt = np.zeros(N)
    alt[::2] = 1.0
    candidates.append(alt)
    
    # 5. Gaussian-like
    x = np.linspace(0, N-1, N)
    gauss = np.exp(-0.5 * ((x - N//2) / 5)**2)
    candidates.append(gauss)
    
    # Evaluate all candidates
    best_h = None
    best_score = -1.0
    
    for h in candidates:
        score = _ratio(h)
        if score > best_score:
            best_score = score
            best_h = h.copy()
            
    # Ensure nonnegative
    if best_h is not None:
        best_h = np.maximum(best_h, 0.0)
    else:
        best_h = np.ones(N)
        
    # Structural change: Use a local optimization approach with coordinate descent
    # from multiple starting points, focusing on sparse structures
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    def score(h):
        return _ratio(h)
    
    # Try coordinate descent from the best candidate
    h = best_h.copy()
    current_score = score(h)
    
    for iteration in range(200):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            old_val = h[i]
            # Try several perturbations
            for delta in [-0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 1.0, -1.0]:
                new_val = max(0.0, old_val + delta)
                h[i] = new_val
                s = score(h)
                if s > current_score + 1e-10:
                    current_score = s
                    improved = True
                else:
                    h[i] = old_val
        if not improved:
            break
        if current_score > best_score:
            best_score = current_score
            best_h = h.copy()
    
    # Try random sparse initialization with local refinement
    np.random.seed(42)
    for trial in range(50):
        if time.time() > deadline:
            break
        # Sparse random initialization
        k = np.random.randint(1, 8)
        idx = np.random.choice(N, k, replace=False)
        h = np.zeros(N)
        h[idx] = np.random.exponential(1.0, k)
        s = score(h)
        
        if s > best_score:
            best_score = s
            best_h = h.copy()
        
        # Local refinement from this point
        current_score = s
        for iteration in range(50):
            if time.time() > deadline:
                break
            improved = False
            for i in range(N):
                old_val = h[i]
                for delta in [-0.5, -0.1, 0.1, 0.5, 1.0, -1.0]:
                    new_val = max(0.0, old_val + delta)
                    h[i] = new_val
                    s = score(h)
                    if s > current_score + 1e-10:
                        current_score = s
                        improved = True
                    else:
                        h[i] = old_val
            if not improved:
                break
            if current_score > best_score:
                best_score = current_score
                best_h = h.copy()
    
    # Try structured patterns with optimization
    patterns = [
        # Geometric sequences
        lambda r, s: np.array([s * (r ** i) for i in range(N)]),
        # Triangular
        lambda r, s: np.array([s * (1 - abs(i - N//2)/(N//2)) for i in range(N)]),
        # Exponential decay
        lambda r, s: np.array([s * np.exp(-r * i) for i in range(N)]),
        # Cosine window
        lambda r, s: np.array([s * (0.5 + 0.5 * np.cos(2 * np.pi * i / N)) for i in range(N)]),
    ]
    
    for pat_func in patterns:
        for r in np.linspace(0.1, 2.0, 10):
            for s in np.linspace(0.1, 2.0, 10):
                if time.time() > deadline:
                    break
                h = pat_func(r, s)
                h = np.maximum(h, 0.0)
                s_val = score(h)
                if s_val > best_score:
                    best_score = s_val
                    best_h = h.copy()
    
    # Final: ensure nonnegative and return
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)
