"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    # New Approach: Hybrid Simulated Annealing + Greedy Polish
    # 1. Stochastic global search (SA) to escape local minima
    # 2. Deterministic local refinement (Greedy) to polish the result
    
    rng = np.random.default_rng(42)
    h = rng.choice([-1.0, 1.0], size=N)
    
    fft_size = 2 * N
    
    def calc_cost(seq):
        f = np.fft.fft(seq, fft_size)
        acf = np.fft.ifft(f * np.conj(f)).real[:N]
        return np.sum(acf[1:] ** 2)
        
    current_cost = calc_cost(h)
    best_h = h.copy()
    best_cost = current_cost
    
    # --- Phase 1: Simulated Annealing ---
    start_time = time.time()
    time_limit = 90.0  # Leave 30s for polish and safety
    
    T_init = max(current_cost * 0.05, 1.0)
    T_curr = T_init
    T_final = 1e-8
    
    max_iters = 5000000 
    
    for i in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        idx = rng.integers(0, N)
        h[idx] = -h[idx]
        
        new_cost = calc_cost(h)
        delta = new_cost - current_cost
        
        if delta < 0 or (T_curr > T_final and rng.random() < np.exp(-delta / T_curr)):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_h = h.copy()
        else:
            h[idx] = -h[idx]
            
        T_curr *= 0.99999
        
    # --- Phase 2: Greedy Polish ---
    h = best_h.copy()
    current_cost = best_cost
    
    for _ in range(5):
        improved = False
        for i in range(N):
            h[i] = -h[i]
            new_cost = calc_cost(h)
            if new_cost < current_cost:
                current_cost = new_cost
                improved = True
            else:
                h[i] = -h[i]
        if not improved:
            break
            
    return h
# EVOLVE-BLOCK-END
