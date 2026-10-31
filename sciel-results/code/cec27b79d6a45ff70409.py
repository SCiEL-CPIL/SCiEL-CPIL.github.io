import numpy as np
import time

N = 50

def compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) for nonnegative h."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    # Normalize to L1=1
    h_norm = h / l1
    h2_sq = np.sum(h_norm ** 2)
    conv = np.convolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-30:
        return 0.0
    return h2_sq / conv_inf

def _optimize_weights(weights, support):
    """
    Given a fixed support (indices) and initial weights, optimize weights
    to maximize the ratio. Uses a simple iterative refinement approach.
    """
    k = len(support)
    if k == 0:
        return 0.0, np.zeros(N)
    if k == 1:
        # Single spike: ratio = 1.0 (under L1=1 normalization)
        h = np.zeros(N)
        h[support[0]] = 1.0
        return 1.0, h
    
    # Start with the given weights
    w = np.array(weights, dtype=np.float64).copy()
    w = np.abs(w)
    total = np.sum(w)
    if total < 1e-15:
        w = np.ones(k) / k
    else:
        w = w / total
    
    best_w = w.copy()
    best_score = compute_ratio(np.zeros(N) + 0)  # placeholder, will compute properly
    # Actually, let's just compute the score of the current w on the support
    def eval_weights(w_local):
        h = np.zeros(N)
        h[support] = w_local
        return compute_ratio(h)
    
    best_score = eval_weights(best_w)
    
    # Simple local search on weights
    max_iter = 5000
    for it in range(max_iter):
        improved = False
        # Try moving mass from one index to another
        for i in range(k):
            if w[i] < 1e-6:
                continue
            for j in range(k):
                if i == j:
                    continue
                # Try several deltas
                for delta_frac in [0.01, 0.05, 0.1, 0.2, 0.5]:
                    delta = w[i] * delta_frac
                    if delta > 1e-15:
                        w_new = w.copy()
                        w_new[i] -= delta
                        w_new[j] += delta
                        if w_new[i] < -1e-12:
                            w_new[i] = 0.0
                        # Don't need to renormalize if we're just moving mass
                        # But let's keep L1=1 for consistency
                        s = np.sum(w_new)
                        if s > 1e-15:
                            w_new = w_new / s
                        score = eval_weights(w_new)
                        if score > best_score + 1e-10:
                            best_score = score
                            best_w = w_new.copy()
                            w = w_new.copy()
                            improved = True
        if not improved:
            break
    
    h = np.zeros(N)
    h[support] = best_w
    final_score = compute_ratio(h)
    return final_score, h

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin: 150s limit, 20s buffer
    
    best_h = np.ones(N) / N
    best_score = compute_ratio(best_h)
    
    # Phase 1: Search over sparse supports
    # The insight from island 8 is that k=1 gives 1.0, and the current best is ~1.09,
    # suggesting optimal k is small but > 1. Let's focus on k=2 to k=12.
    
    def random_support(k):
        return np.random.choice(N, k, replace=False)
    
    # Grid search for small k
    for k in range(2, 13):
        if time.time() > deadline:
            break
        
        # Number of random support samples for this k
        n_samples = max(10, 200 // k)
        
        for s in range(n_samples):
            if time.time() > deadline:
                break
            
            support = random_support(k)
            
            # Initialize weights: try a few different initializations
            for init_type in range(3):
                if time.time() > deadline:
                    break
                
                if init_type == 0:
                    # Uniform
                    w_init = np.ones(k) / k
                elif init_type == 1:
                    # Random Dirichlet
                    w_init = np.random.dirichlet(np.ones(k))
                else:
                    # Decaying
                    x = np.arange(k)
                    w_init = np.exp(-x / (k * 0.5))
                    w_init = w_init / np.sum(w_init)
                
                score, h = _optimize_weights(w_init, support)
                if score > best_score:
                    best_score = score
                    best_h = h.copy()
    
    # Phase 2: Refine the best sparse solution with dense local search
    if time.time() < deadline:
        h = best_h.copy()
        for it in range(20000):
            if time.time() > deadline:
                break
            
            # Random perturbation
            i = np.random.randint(N)
            j = np.random.randint(N)
            if i == j:
                continue
            
            h_new = h.copy()
            # Move a small fraction of mass
            delta = h[i] * np.random.uniform(0.01, 0.2)
            h_new[i] -= delta
            h_new[j] += delta
            
            if h_new[i] < -1e-12 or h_new[j] < -1e-12:
                continue
            
            h_new[h_new < 0] = 0
            s = np.sum(h_new)
            if s < 1e-15:
                continue
            h_new = h_new / s
            
            new_score = compute_ratio(h_new)
            if new_score > best_score:
                best_score = new_score
                best_h = h_new.copy()
                h = h_new.copy()
    
    # Phase 3: Try structured parametric families
    if time.time() < deadline:
        # Triangular / tent shapes
        for k in range(1, N):
            if time.time() > deadline:
                break
            for center in range(N):
                if time.time() > deadline:
                    break
                # Tent: h[i] = max(0, 1 - |i - center| / k)
                x = np.arange(N)
                h = np.maximum(0, 1 - np.abs(x - center) / k)
                s = np.sum(h)
                if s < 1e-15:
                    continue
                h = h / s
                score = compute_ratio(h)
                if score > best_score:
                    best_score = score
                    best_h = h.copy()
        
        # Parabolic shapes
        for k in range(1, N):
            if time.time() > deadline:
                break
            for center in range(N):
                if time.time() > deadline:
                    break
                x = np.arange(N)
                h = np.maximum(0, 1 - (x - center) ** 2 / (k ** 2))
                s = np.sum(h)
                if s < 1e-15:
                    continue
                h = h / s
                score = compute_ratio(h)
                if score > best_score:
                    best_score = score
                    best_h = h.copy()
    
    # Final safety check
    if best_score < compute_ratio(np.ones(N) / N):
        best_h = np.ones(N) / N
        best_score = compute_ratio(best_h)
    
    return list(best_h)

# EVOLVE-BLOCK-START
# (The function construct() above is the entry point)
# EVOLVE-BLOCK-END
