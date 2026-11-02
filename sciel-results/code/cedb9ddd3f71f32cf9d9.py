import numpy as np
import time
import random

N = 600

def compute_score(h, exclude_lag0=True):
    """
    Compute the normalized max autoconvolution.
    If exclude_lag0 is True, max is over k >= 1.
    Normalization: (sum h)^2.
    """
    S_h = np.sum(h)
    if S_h < 1e-6:
        return np.inf
    
    # Use FFT for speed, though N=600 is small enough for direct correlate.
    # np.correlate is O(N^2) = 360k ops, very fast.
    # FFT is O(N log N), also fast. Let's stick to correlate for simplicity and precision.
    S = np.correlate(h, h, mode='full')
    
    # S is length 2N-1. Center is N-1 (lag 0).
    if exclude_lag0:
        # Max of S[0:N-1] and S[N:2N-1]
        # Due to symmetry S[k] = S[-k], max(S[0:N-1]) == max(S[N:2N-1])
        # We can just check one side to save half the time, but max is cheap.
        max_side = np.max(np.concatenate([S[:N-1], S[N:2*N-1]]))
    else:
        max_side = np.max(S)
    
    return max_side / (S_h ** 2)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin for 150s limit
    
    best_score = np.inf
    best_h = np.ones(N) / N  # Uniform baseline
    
    # 1. Evaluate Uniform
    s = compute_score(best_h, exclude_lag0=True)
    if s < best_score:
        best_score = s
        best_h = best_h.copy()
        
    # 2. Strategy: Search over a broader family of parametric shapes
    #    - Two-block (existing, but we'll re-search with randomization)
    #    - Raised Cosine (smoothed block)
    #    - Gaussian (smoothed bell)
    #    - Triangle (linear taper)
    #    
    #    For each shape, we perform a local random search on its parameters.
    
    def evaluate_and_update(h):
        nonlocal best_score, best_h
        h_sum = np.sum(h)
        if h_sum < 1e-6:
            return np.inf
        h_norm = h / h_sum
        s = compute_score(h_norm, exclude_lag0=True)
        if s < best_score:
            best_score = s
            best_h = h_norm.copy()
        return s

    def search_two_block():
        # Randomized search for two disjoint blocks
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            W1 = random.randint(10, N//3)
            W2 = random.randint(10, N//3)
            c1 = random.randint(W1//2, N//2 - W1//2)
            c2 = random.randint(N//2 + W2//2, N - W2//2)
            A2 = random.uniform(0.1, 2.0)
            
            h = np.zeros(N)
            s1 = c1 - W1//2
            e1 = c1 + W1//2 + 1
            s2 = c2 - W2//2
            e2 = c2 + W2//2 + 1
            
            if s1 < 0 or e1 > N or s2 < 0 or e2 > N:
                continue
            
            h[s1:e1] = 1.0
            h[s2:e2] = A2
            
            s = evaluate_and_update(h)
            
            # Simple local refinement
            for _ in range(5):
                if time.time() > deadline - 5:
                    break
                # Perturb parameters
                new_W1 = max(1, W1 + random.randint(-5, 5))
                new_W2 = max(1, W2 + random.randint(-5, 5))
                new_c1 = min(N-1, max(0, c1 + random.randint(-5, 5)))
                new_c2 = min(N-1, max(0, c2 + random.randint(-5, 5)))
                new_A2 = max(0.01, A2 + random.uniform(-0.2, 0.2))
                
                h_new = np.zeros(N)
                ns1 = new_c1 - new_W1//2
                ne1 = new_c1 + new_W1//2 + 1
                ns2 = new_c2 - new_W2//2
                ne2 = new_c2 + new_W2//2 + 1
                
                if ns1 < 0 or ne1 > N or ns2 < 0 or ne2 > N:
                    continue
                if ns1 > ne1 or ns2 > ne2:
                    continue
                
                h_new[ns1:ne1] = 1.0
                h_new[ns2:ne2] = new_A2
                
                s_new = evaluate_and_update(h_new)
                if s_new < s:
                    s = s_new
                    W1, W2, c1, c2, A2 = new_W1, new_W2, new_c1, new_c2, new_A2

    def search_raised_cosine():
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            W = random.randint(50, N//2)
            center = N//2
            # Symmetric raised cosine
            x = np.arange(-W//2, W//2+1)
            vals = 0.5 * (1 + np.cos(2*np.pi*x/W))
            start = center - W//2
            end = center + W//2 + 1
            if start < 0 or end > N:
                continue
            
            h = np.zeros(N)
            h[start:end] = vals
            s = evaluate_and_update(h)
            
            # Local refinement on W
            for _ in range(5):
                if time.time() > deadline - 5:
                    break
                new_W = max(10, W + random.randint(-10, 10))
                x = np.arange(-new_W//2, new_W//2+1)
                vals = 0.5 * (1 + np.cos(2*np.pi*x/new_W))
                start = center - new_W//2
                end = center + new_W//2 + 1
                if start < 0 or end > N:
                    continue
                h_new = np.zeros(N)
                h_new[start:end] = vals
                s_new = evaluate_and_update(h_new)
                if s_new < s:
                    s = s_new
                    W = new_W

    def search_gaussian():
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            sigma = random.uniform(10, N//4)
            center = N//2
            x = np.arange(N)
            h = np.exp(-0.5 * ((x - center) / sigma) ** 2)
            s = evaluate_and_update(h)
            
            # Local refinement on sigma
            for _ in range(5):
                if time.time() > deadline - 5:
                    break
                new_sigma = max(1.0, sigma + random.uniform(-5, 5))
                h_new = np.exp(-0.5 * ((x - center) / new_sigma) ** 2)
                s_new = evaluate_and_update(h_new)
                if s_new < s:
                    s = s_new
                    sigma = new_sigma

    def search_triangle():
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            W = random.randint(50, N//2)
            center = N//2
            start = center - W//2
            end = center + W//2 + 1
            if start < 0 or end > N:
                continue
            
            h = np.zeros(N)
            # Triangle peak at center
            h[start:center] = np.linspace(0, 1, center - start)
            h[center:end] = np.linspace(1, 0, end - center)
            s = evaluate_and_update(h)
            
            # Local refinement on W
            for _ in range(5):
                if time.time() > deadline - 5:
                    break
                new_W = max(10, W + random.randint(-10, 10))
                start = center - new_W//2
                end = center + new_W//2 + 1
                if start < 0 or end > N:
                    continue
                h_new = np.zeros(N)
                h_new[start:center] = np.linspace(0, 1, center - start)
                h_new[center:end] = np.linspace(1, 0, end - center)
                s_new = evaluate_and_update(h_new)
                if s_new < s:
                    s = s_new
                    W = new_W

    # Execute searches
    search_two_block()
    search_raised_cosine()
    search_gaussian()
    search_triangle()
    
    # If time remains, do a few more random restarts of the best shape type
    # For simplicity, we just return the best found.
    
    return list(best_h)
