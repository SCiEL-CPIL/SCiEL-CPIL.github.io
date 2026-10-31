import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Candidate 1: Seed (Uniform)
    h_uniform = np.ones(N)
    
    # Candidate 2: Delta
    h_delta = np.zeros(N)
    h_delta[N//2] = 1.0
    
    # Candidate 3: Gaussian
    best_h = h_uniform
    best_score = 0.0
    
    # Helper to calculate score
    def calc_score(h):
        h = np.array(h)
        if np.sum(h) == 0:
            return 0.0
        l1 = np.sum(h)
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        l_inf = np.max(conv)
        if l1 * l_inf == 0:
            return 0.0
        return l2_sq / (l1 * l_inf)
    
    current_best_score = calc_score(h_uniform)
    best_h = h_uniform.copy()
    
    # Test Delta
    s_delta = calc_score(h_delta)
    if s_delta > current_best_score:
        current_best_score = s_delta
        best_h = h_delta.copy()
        
    # Test Gaussians
    centers = [N//2]
    widths = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    for c in centers:
        for w in widths:
            x = np.arange(N)
            h_gauss = np.exp(-((x-c)**2) / (2*w**2))
            s_gauss = calc_score(h_gauss)
            if s_gauss > current_best_score:
                current_best_score = s_gauss
                best_h = h_gauss.copy()
                
    # Local search around best_h
    # Use random perturbations
    for _ in range(5000):
        if time.time() > deadline:
            break
        h_trial = best_h.copy()
        # Randomly perturb a few elements
        idx = np.random.choice(N, size=3, replace=False)
        h_trial[idx] += np.random.normal(0, 0.1, size=3)
        h_trial = np.maximum(h_trial, 0)
        
        s_trial = calc_score(h_trial)
        if s_trial > current_best_score:
            current_best_score = s_trial
            best_h = h_trial.copy()
            
    return list(best_h)
