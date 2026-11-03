import numpy as np
import time

N = 100

def _compute_max_overlap(h):
    """
    Compute the maximum linear overlap C(k) = sum h[i] * (1 - h[i+k]) for k != 0.
    """
    g = 1.0 - h
    # np.correlate(h, g, 'full')
    # Length 2N-1.
    # Center is N-1.
    c = np.correlate(h, g, mode='full')
    # Exclude center (shift 0)
    # c[:N-1] and c[N:]
    # Note: c is symmetric if h is real? No, h and g are different.
    # But we take max over all non-zero shifts.
    c_nonzero = np.concatenate([c[:N-1], c[N:]])
    return np.max(c_nonzero)

def _random_binary_seq(n_ones=50, seed=None):
    """
    Generate a random binary sequence of length N with exactly n_ones ones.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    h = np.zeros(N)
    indices = rng.choice(N, size=n_ones, replace=False)
    h[indices] = 1.0
    return h

def _swap_move(h, rng):
    """
    Perform a random swap move: pick i where h[i]=1 and j where h[j]=0, swap them.
    Returns new h and the indices swapped.
    """
    ones = np.where(h == 1.0)[0]
    zeros = np.where(h == 0.0)[0]
    
    if len(ones) == 0 or len(zeros) == 0:
        return h.copy(), -1, -1
        
    i = ones[rng.integers(0, len(ones))]
    j = zeros[rng.integers(0, len(zeros))]
    
    h_new = h.copy()
    h_new[i] = 0.0
    h_new[j] = 1.0
    return h_new, i, j

def _local_search(h, max_iters=100):
    """
    Greedy local search: try all possible swaps and keep the best improvement.
    Repeat until no improvement or max_iters.
    """
    current_score = _compute_max_overlap(h)
    
    for _ in range(max_iters):
        best_swap = None
        best_score = current_score
        best_h = h
        
        ones = np.where(h == 1.0)[0]
        zeros = np.where(h == 0.0)[0]
        
        # To speed up, we can limit the number of swaps tested per iteration
        # or test all. N=100, 50*50=2500 swaps.
        # _compute_max_overlap is fast. 2500 * 100 ops = 250k ops.
        # 100 iters = 25M ops. Should be < 1s.
        
        for i in ones:
            for j in zeros:
                if i == j: continue
                h_new = h.copy()
                h_new[i] = 0.0
                h_new[j] = 1.0
                score = _compute_max_overlap(h_new)
                if score < best_score:
                    best_score = score
                    best_swap = (i, j)
                    best_h = h_new
        
        if best_swap is None:
            break
            
        h = best_h
        current_score = best_score
        
    return h, current_score

def construct():
    start_time = time.time()
    time_limit = 145.0
    
    best_h_global = None
    best_score_global = np.inf
    
    # Multiple restarts
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit - 10:
            break
            
        seed = 42 + restart
        rng = np.random.default_rng(seed)
        
        # Initialize
        h_curr = _random_binary_seq(50, seed=seed)
        score_curr = _compute_max_overlap(h_curr)
        
        if score_curr < best_score_global:
            best_score_global = score_curr
            best_h_global = h_curr.copy()
        
        # Simulated Annealing
        T_init = 5.0
        T_final = 0.1
        n_sa_steps = 5000
        
        for it in range(n_sa_steps):
            if time.time() - start_time > time_limit - 10:
                break
                
            # Cooling schedule
            progress = it / n_sa_steps
            T = T_init * (T_final / T_init) ** progress
            
            h_prop, i, j = _swap_move(h_curr, rng)
            if i == -1:
                continue
                
            score_prop = _compute_max_overlap(h_prop)
            
            delta = score_prop - score_curr
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                h_curr = h_prop
                score_curr = score_prop
                
                if score_curr < best_score_global:
                    best_score_global = score_curr
                    best_h_global = h_curr.copy()
                    
        # Local Search Refinement
        if time.time() - start_time < time_limit - 5:
            h_refined, score_refined = _local_search(h_curr, max_iters=50)
            
            if score_refined < best_score_global:
                best_score_global = score_refined
                best_h_global = h_refined.copy()
                
    # If we still have time, try a continuous relaxation on the best binary solution
    # This is a structural addition: using the binary solution as a seed for continuous optimization
    if time.time() - start_time < time_limit - 2 and best_h_global is not None:
        h_cont = best_h_global.copy()
        # Simple gradient step on continuous space?
        # Or just project? The binary solution is already on the boundary.
        # Let's try a few random continuous perturbations and project.
        for _ in range(100):
            if time.time() - start_time > time_limit - 2:
                break
            h_pert = h_cont + np.random.normal(0, 0.1, N)
            # Project to [0,1] and sum=50
            h_pert = np.clip(h_pert, 0, 1)
            diff = 50.0 - np.sum(h_pert)
            h_pert += diff / N
            h_pert = np.clip(h_pert, 0, 1)
            # Re-normalize sum exactly?
            # The problem requires sum exactly len/2.
            # If we clip, sum might drift.
            # Let's use the projection from inspiration 1 for continuous.
            for _ in range(20):
                diff = 50.0 - np.sum(h_pert)
                if abs(diff) < 1e-9: break
                h_pert += diff / N
                h_pert = np.clip(h_pert, 0, 1)
                
            score_pert = _compute_max_overlap(h_pert)
            if score_pert < best_score_global:
                best_score_global = score_pert
                best_h_global = h_pert.copy()

    return best_h_global
