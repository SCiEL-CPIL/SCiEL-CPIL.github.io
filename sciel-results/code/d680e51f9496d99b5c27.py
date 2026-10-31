import numpy as np
import time
import itertools

N = 50

def compute_ratio(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf) for h >= 0.
    """
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h)
    if l1 < 1e-300:
        return 0.0
    
    l2_sq = np.sum(h * h)
    
    # Efficient autoconvolution using FFT
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h, n=n_fft)
    hh = np.fft.irfft(H * H, n=n_fft)
    hh_max = np.max(hh)
    
    if hh_max < 1e-300:
        return 0.0
        
    return l2_sq / (l1 * hh_max)

def _refine_weights(support, initial_weights, time_limit, rng):
    """
    Deterministic-ish local optimization of weights for a fixed support.
    Uses a fast coordinate ascent / hill climbing with adaptive step sizes.
    """
    K = len(support)
    if K == 0:
        return initial_weights, 0.0
    
    w = np.array(initial_weights, dtype=np.float64)
    w = np.maximum(w, 0)
    s = np.sum(w)
    if s < 1e-10:
        w = np.ones(K) / K
    else:
        w /= s
        
    current_score = compute_ratio(np.concatenate([np.zeros(support[0]), w, np.zeros(N - support[-1] - 1)]) if len(support) > 1 else np.zeros(N))
    # Actually, let's just build the full h vector properly
    def build_h(w_vec):
        h = np.zeros(N)
        h[support] = w_vec
        return h
    
    current_h = build_h(w)
    current_score = compute_ratio(current_h)
    
    start_time = time.time()
    step_size = 0.1
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        improved = False
        # Cycle through coordinates
        for i in range(K):
            if time.time() - start_time > time_limit:
                break
            
            # Try to increase w[i]
            for delta in [step_size, step_size * 0.5, step_size * 0.1]:
                w_new = w.copy()
                w_new[i] += delta
                w_new = np.maximum(w_new, 0)
                s_new = np.sum(w_new)
                if s_new < 1e-10: continue
                w_new /= s_new
                
                h_new = build_h(w_new)
                score_new = compute_ratio(h_new)
                
                if score_new > current_score:
                    w = w_new
                    current_score = score_new
                    improved = True
                    break
            
            # Try to decrease w[i]
            for delta in [step_size, step_size * 0.5, step_size * 0.1]:
                w_new = w.copy()
                w_new[i] -= delta
                w_new = np.maximum(w_new, 0)
                s_new = np.sum(w_new)
                if s_new < 1e-10: continue
                w_new /= s_new
                
                h_new = build_h(w_new)
                score_new = compute_ratio(h_new)
                
                if score_new > current_score:
                    w = w_new
                    current_score = score_new
                    improved = True
                    break
        
        if improved:
            no_improve_count = 0
            step_size = max(0.001, step_size * 0.8)
        else:
            no_improve_count += 1
            step_size = max(0.0001, step_size * 0.5)
            if step_size < 0.0005 and no_improve_count > 5:
                break
                
    return w, current_score

def _greedy_sidon_search(max_K=15, time_limit=60.0):
    """
    Greedily construct a Sidon-like set by adding elements that minimize
    the increase in autoconvolution peak.
    """
    start_time = time.time()
    best_support = []
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Start with a single element
    current_support = [0]
    
    # We will try to build up to max_K
    for target_K in range(1, max_K + 1):
        if time.time() - start_time > time_limit:
            break
            
        # Find the best element to add
        best_candidate = -1
        best_candidate_score = -np.inf
        best_candidate_h = None
        
        # Candidate elements: 0 to N-1
        # We prioritize elements far from existing ones to reduce collisions
        # But let's just check all available
        available = [i for i in range(N) if i not in current_support]
        
        # To speed up, we can shuffle and check a subset if N is large, but N=50 is small
        # Let's check all
        for cand in available:
            new_support = current_support + [cand]
            new_support.sort()
            
            # Initial weights: uniform
            w_init = np.ones(len(new_support)) / len(new_support)
            h_cand = np.zeros(N)
            h_cand[new_support] = w_init
            
            score_cand = compute_ratio(h_cand)
            
            if score_cand > best_candidate_score:
                best_candidate_score = score_cand
                best_candidate = cand
                best_candidate_h = h_cand
                best_candidate_support = new_support
        
        if best_candidate != -1:
            current_support = best_candidate_support
            best_h = best_candidate_h
            best_score = best_candidate_score
            
            # Refine weights for this specific support
            # Allocate time based on remaining budget
            remaining = time_limit - (time.time() - start_time)
            refine_time = min(remaining * 0.5, 2.0) # Spend up to 2s per step or half remaining
            if refine_time > 0.1:
                w_refined, score_refined = _refine_weights(current_support, best_candidate_h[current_support], refine_time, None)
                if score_refined > best_score:
                    best_score = score_refined
                    best_h = np.zeros(N)
                    best_h[current_support] = w_refined
    
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Phase 1: Greedy Sidon Construction
    # This finds a structurally good support set
    h_greedy, score_greedy = _greedy_sidon_search(max_K=20, time_limit=30.0)
    if score_greedy > best_score:
        best_score = score_greedy
        best_h = h_greedy.copy()
        
    # Phase 2: Multi-start refinement on the best support and some variants
    # Extract support from best_h
    support = np.where(best_h > 1e-8)[0]
    if len(support) > 0:
        # Refine this support heavily
        remaining_time = deadline - time.time()
        if remaining_time > 10:
            w_init = best_h[support]
            w_refined, score_refined = _refine_weights(support, w_init, remaining_time * 0.6, None)
            if score_refined > best_score:
                best_score = score_refined
                best_h = np.zeros(N)
                best_h[support] = w_refined
                
        # Try a few random perturbations of the support
        rng = np.random.default_rng(42)
        for _ in range(5):
            if time.time() > deadline - 5:
                break
                
            # Mutate support: remove one, add one
            if len(support) > 3:
                new_support = list(support)
                idx_to_remove = rng.integers(0, len(new_support))
                removed = new_support.pop(idx_to_remove)
                
                # Add a new random element
                available = [i for i in range(N) if i not in new_support]
                new_elem = rng.choice(available)
                new_support.append(new_elem)
                new_support.sort()
                
                w_init = np.ones(len(new_support)) / len(new_support)
                h_cand = np.zeros(N)
                h_cand[new_support] = w_init
                
                score_cand = compute_ratio(h_cand)
                
                # Quick refine
                refine_time = min(5.0, (deadline - time.time()) * 0.5)
                if refine_time > 1.0:
                    w_ref, score_ref = _refine_weights(new_support, w_init, refine_time, None)
                    if score_ref > score_cand:
                        score_cand = score_ref
                        h_cand = np.zeros(N)
                        h_cand[new_support] = w_ref
                        
                if score_cand > best_score:
                    best_score = score_cand
                    best_h = h_cand.copy()
                    support = np.array(new_support)
    
    # Final check
    final_score = compute_ratio(best_h)
    if final_score > best_score:
        best_score = final_score
        
    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(compute_ratio(h))
