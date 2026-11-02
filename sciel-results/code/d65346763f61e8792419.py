import numpy as np
import time

N = 600
h = (1/4 - (-1/4)) / N  # 1/1200

def _compute_score(p):
    """
    Compute the normalized maximum autoconvolution score for a probability vector p.
    Score = max(c) / h, where c is the discrete autoconvolution of p.
    Note: The seed's score formula was max(c)*h / (sum(f)*h)^2.
    If p sums to 1, sum(f)=1 (if we scale f=p).
    Then score = max(c)*h / 1^2? No.
    Let's re-verify the scaling.
    Seed:
    s = sum(f)*h
    max_conv = max(c)*h
    score = max_conv / s^2
    
    If we return p (sum 1), let f = p.
    s = 1 * h = h.
    max_conv = max(c) * h.
    score = (max(c) * h) / h^2 = max(c) / h.
    
    For uniform p (1/N):
    c_max = N * (1/N)^2 = 1/N.
    score = (1/N) / h = 1/(N*h) = 1/(600 * 1/1200) = 1/0.5 = 2.
    This matches the baseline.
    """
    # Use FFT for convolution
    L = 1 << (2*N).bit_length()
    F_p = np.fft.rfft(p, L)
    c = np.fft.irfft(F_p * F_p, L)[:2*N-1]
    # c is the autoconvolution
    return np.max(c) / h

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_score = np.inf
    best_p = None
    
    np.random.seed(42)
    
    # Define initial configurations
    # 1. Uniform
    p_uniform = np.ones(N) / N
    
    # 2. Edge-heavy (Bipolar)
    # High at ends, low in middle
    x = np.linspace(0, 1, N)
    # Use a function that peaks at 0 and 1
    p_edge = np.exp(-((x - 0.5)**2) * 0) # Flat? No.
    # Let's try: p_i proportional to 1 if i < N/4 or i > 3N/4, else 0.1
    p_edge = np.ones(N) * 0.01
    p_edge[:N//4] = 1.0
    p_edge[3*N//4:] = 1.0
    p_edge /= np.sum(p_edge)
    
    # 3. Extreme edges (Deltas)
    p_delta = np.zeros(N)
    p_delta[0] = 0.5
    p_delta[N-1] = 0.5
    
    # 4. Random
    p_rand = np.random.rand(N)
    p_rand /= np.sum(p_rand)
    
    starts = [p_uniform, p_edge, p_delta, p_rand]
    
    # Simulated Annealing
    def anneal(p_init, time_limit):
        p = p_init.copy()
        current_score = _compute_score(p)
        best_local_score = current_score
        best_local_p = p.copy()
        
        T = 0.01  # Initial temperature
        sigma = 0.1 # Noise scale
        alpha = 0.99 # Cooling rate
        min_T = 1e-6
        min_sigma = 1e-6
        
        start = time.time()
        iter_count = 0
        
        while time.time() < time_limit and T > min_T:
            # Perturb
            noise = np.random.normal(0, sigma, N)
            p_new = p + noise
            p_new = np.clip(p_new, 0, None)
            sum_new = np.sum(p_new)
            if sum_new == 0:
                p_new = p.copy()
            else:
                p_new /= sum_new
            
            new_score = _compute_score(p_new)
            delta = new_score - current_score
            
            # Acceptance
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                p = p_new
                current_score = new_score
                
                if current_score < best_local_score:
                    best_local_score = current_score
                    best_local_p = p.copy()
                    
                # Adjust sigma based on recent acceptance? 
                # Simple adaptive: if we just accepted a bad move (delta>0), we might want to increase sigma?
                # Or just cool T.
                T *= alpha
                # Decay sigma
                sigma *= 0.999 
                
            else:
                # Rejected
                T *= alpha
                sigma *= 0.999
                
            iter_count += 1
            
            # Check time periodically to avoid long stalls in FFT? FFT is fast.
            if iter_count % 100 == 0 and time.time() > start + time_limit * 0.9:
                break
                
        return best_local_p, best_local_score

    # Run SA for each start, allocating time
    # Total time 140s. 4 starts. ~30s each.
    # But we should prioritize promising starts.
    # Let's run all for a fixed duration, then continue the best ones if time remains.
    
    time_per_start = 20.0
    results = []
    
    for i, p_init in enumerate(starts):
        if time.time() > deadline - 5: # Stop if close to deadline
            break
            
        limit = time.time() + time_per_start
        p_best, s_best = anneal(p_init, limit)
        results.append((s_best, p_best))
        
        if s_best < best_score:
            best_score = s_best
            best_p = p_best

    # If time remains, refine the best solution found so far with a fresh SA run from best_p
    remaining_time = deadline - time.time()
    if remaining_time > 10 and best_p is not None:
        # Reheat or just run SA again from best_p
        p_best, s_best = anneal(best_p, remaining_time - 5)
        if s_best < best_score:
            best_score = s_best
            best_p = p_best

    if best_p is None:
        return p_uniform.tolist()
        
    return best_p.tolist()
