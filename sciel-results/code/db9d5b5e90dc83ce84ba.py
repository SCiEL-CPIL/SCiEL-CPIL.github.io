import numpy as np
import time

N = 600

def compute_score(h, exclude_lag0=True):
    """
    Compute the normalized max autoconvolution.
    If exclude_lag0 is True, max is over k >= 1.
    Normalization: (sum h)^2.
    """
    if np.sum(h) < 1e-6:
        return np.inf
    S = np.correlate(h, h, mode='full')
    # S is length 2N-1. Center is N-1.
    # Lags: -N+1 to N-1.
    # We want max over k != 0.
    # k=0 is at index N-1.
    # k>=1 corresponds to indices N to 2N-2 and 0 to N-2.
    # Due to symmetry S[k] = S[-k], we can just check one side.
    if exclude_lag0:
        # Max of S[0:N-1] and S[N:2N-1]
        max_side = np.max(np.concatenate([S[:N-1], S[N:2*N-1]]))
    else:
        max_side = np.max(S)
    
    return max_side / (np.sum(h) ** 2)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_score = np.inf
    best_h = np.ones(N) / N # Uniform baseline
    
    # 1. Evaluate Uniform
    s = compute_score(best_h, exclude_lag0=True)
    if s < best_score:
        best_score = s
        best_h = best_h.copy()
        
    # 2. Try Random Binary
    h_rand = np.random.randint(0, 2, N).astype(float)
    if np.sum(h_rand) > 0:
        s = compute_score(h_rand, exclude_lag0=True)
        if s < best_score:
            best_score = s
            best_h = h_rand.copy()
            
    # 3. Try Two Disjoint Blocks
    # Optimize width W and separation D
    for W in [N//10, N//5, N//4, N//3]:
        for D in [N//2, N//2 + W//2]:
            h = np.zeros(N)
            # Center blocks
            start1 = int(N/2 - D)
            start2 = int(N/2 + D)
            # Ensure within bounds
            if start1 < 0: start1 = 0
            if start2 + W >= N: start2 = N - W
            if start1 + W > start2: continue # Overlap
            
            h[start1:start1+W] = 1.0
            h[start2:start2+W] = 1.0
            
            s = compute_score(h, exclude_lag0=True)
            if s < best_score:
                best_score = s
                best_h = h.copy()

    # 4. Local Search (Bit Flipping) on the best candidate
    h_curr = best_h.copy()
    # If best_h is continuous, round to binary for local search? 
    # Or optimize continuous values?
    # Bit flipping is fast. Let's try it on a binary version of the best candidate.
    # If best_h is uniform, it's all 1s. Flipping to 0 creates a hole.
    # Let's start from a random binary if the current best is not binary.
    if not np.allclose(h_curr, np.round(h_curr)):
         h_curr = np.random.randint(0, 2, N).astype(float)
         s = compute_score(h_curr, exclude_lag0=True)
         if s < best_score:
             best_score = s
             best_h = h_curr.copy()
             
    # Simulated Annealing / Hill Climbing
    T = 0.01
    for i in range(50000):
        if time.time() > deadline - 10:
            break
        
        # Pick a random index to flip
        idx = np.random.randint(N)
        old_val = h_curr[idx]
        new_val = 1.0 - old_val # Flip 0<->1
        
        h_curr[idx] = new_val
        s_new = compute_score(h_curr, exclude_lag0=True)
        
        delta = s_new - best_score # Note: best_score is global best, not current
        # We should compare s_new with the score of h_curr before flip
        # Let's keep a variable for current score
        # Actually, just accept if s_new < best_score (global)
        # Or use SA: accept if s_new < s_curr or exp(-(s_new-s_curr)/T) > rand
        # For simplicity and speed, let's do Greedy Hill Climbing first.
        if s_new < best_score:
            best_score = s_new
            best_h = h_curr.copy()
            T = max(0.001, T * 0.99)
        else:
            # Revert if not better (Greedy)
            h_curr[idx] = old_val
            T = max(0.001, T * 0.99)
            
    return list(best_h)
