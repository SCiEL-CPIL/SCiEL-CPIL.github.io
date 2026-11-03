# EVOLVE-BLOCK-START
import numpy as np
import random

N = 400
N_PAD = 2048

def compute_c3_bound(s):
    s_pad = np.zeros(N_PAD)
    s_pad[:N] = s
    S = np.fft.fft(s_pad)
    C3 = np.fft.ifft(S ** 3).real
    return np.max(np.abs(C3))

def greedy_search(signs):
    current_bound = compute_c3_bound(signs)
    improved = True
    # Limit passes to prevent infinite loops or excessive time if stuck
    max_passes = 10
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        indices = list(range(N))
        random.shuffle(indices)
        for i in indices:
            signs[i] *= -1.0
            new_bound = compute_c3_bound(signs)
            if new_bound < current_bound:
                current_bound = new_bound
                improved = True
            else:
                signs[i] *= -1.0
    return current_bound, signs

def local_search(signs):
    best_signs = signs.copy()
    best_bound = compute_c3_bound(best_signs)
    
    # Multiple restarts with perturbations
    for _ in range(50):
        candidate = best_signs.copy()
        num_flips = random.randint(5, 20)
        for _ in range(num_flips):
            i = random.randint(0, N - 1)
            candidate[i] *= -1.0
        
        bound, polished = greedy_search(candidate)
        
        if bound < best_bound:
            best_bound = bound
            best_signs = polished
            
    return best_signs, best_bound

def construct():
    random.seed(42)
    
    # Generate candidates from Logistic Map
    candidates = []
    # Use Golden Ratio conjugate for systematic exploration of the unit interval
    golden_ratio_conj = 0.6180339887
    for i in range(500):
        seed = (i * golden_ratio_conj) % 1.0
        # Avoid exact 0 or 1 if they cause issues (though logistic map handles them)
        if seed < 1e-9:
            seed = 1e-9
        if seed > 1 - 1e-9:
            seed = 1 - 1e-9
            
        x = seed
        signs = np.zeros(N)
        for j in range(N):
            x = 4.0 * x * (1.0 - x)
            signs[j] = 1.0 if x > 0.5 else -1.0
            
        bound = compute_c3_bound(signs)
        candidates.append((bound, signs))
        
    # Sort and pick top 10
    candidates.sort(key=lambda x: x[0])
    
    best_overall_signs = None
    best_overall_bound = float('inf')
    
    for bound, signs in candidates[:10]:
        polished_signs, polished_bound = local_search(signs)
        
        if polished_bound < best_overall_bound:
            best_overall_bound = polished_bound
            best_overall_signs = polished_signs
            
    # Final polish on the global best
    if best_overall_signs is not None:
        final_signs, final_bound = greedy_search(best_overall_signs)
        if final_bound < best_overall_bound:
            best_overall_signs = final_signs
            best_overall_bound = final_bound

    return best_overall_signs
# EVOLVE-BLOCK-END
