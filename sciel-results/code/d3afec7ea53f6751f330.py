import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Simulated Annealing on the Score Vector
    1. Initialize scores using a parallel exclusion process heuristic.
    2. Optimize scores using Simulated Annealing:
       - Perturbation: Add Gaussian noise to a random subset of scores.
       - Acceptance: Accept worse solutions with probability exp(-delta/T).
       - Cooling: Geometric cooling schedule.
    """
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Precompute neighbor lists for fast simulation
    neighbors_list = [np.where(adj[i])[0].tolist() for i in range(n)]
    
    # 1. Initial Heuristic: Parallel Discrete-Time Exclusion Process
    active = np.ones(n, dtype=bool)
    selected_time = np.full(n, n + 1, dtype=float)
    round_num = 0
    
    while active.any():
        active_nbr_count = np.zeros(n, dtype=int)
        for v in np.where(active)[0]:
            nbrs = neighbors_list[v]
            if nbrs:
                active_nbr_count[v] = np.sum(active[nbrs])
        
        min_count = np.min(active_nbr_count[active])
        candidates = np.where(active & (active_nbr_count == min_count))[0]
        
        if len(candidates) == 0:
            break
            
        for v in candidates:
            selected_time[v] = round_num
            active[v] = False
            nbrs = neighbors_list[v]
            if nbrs:
                active[nbrs] = False
        
        round_num += 1
        if time.time() - start_time > 110:
            break
    
    scores = -selected_time
    deg = adj.sum(axis=1).astype(float)
    scores = scores - 0.01 * deg
    
    def simulate_fast(s):
        """Fast greedy simulation using numpy boolean array."""
        order = np.argsort(-s, kind='stable')
        active = np.ones(n, dtype=bool)
        count = 0
        for v in order:
            if active[v]:
                count += 1
                active[v] = False
                nbrs = neighbors_list[v]
                if nbrs:
                    active[nbrs] = False
        return count
    
    # 2. Baseline
    best_val = simulate_fast(scores)
    best_scores = scores.copy()
    current_scores = scores.copy()
    current_val = best_val
    
    # 3. Simulated Annealing
    max_time = 115.0  # Safety margin
    rng = np.random.default_rng(42)
    
    # Annealing parameters
    T_start = 1.0
    T_end = 0.001
    # Estimate number of steps we can afford. 
    # simulate_fast is O(n^2) roughly due to argsort and loop. n=200.
    # Let's run until time limit, with a fixed number of steps per "temperature" level?
    # Or just continuous cooling based on time?
    # Better: Fixed number of iterations, but adapt if time allows.
    # Let's try a fixed number of iterations first, say 5000, and see how long it takes.
    # If it's too slow, we reduce. If fast, we can increase.
    
    # Estimate time per iteration
    t0 = time.time()
    for _ in range(10):
        _ = simulate_fast(current_scores)
    time_per_iter = (time.time() - t0) / 10.0
    
    # Target: use 90% of available time for SA
    available_time = max_time - (time.time() - start_time)
    max_iters = int(available_time * 0.9 / time_per_iter)
    
    if max_iters < 100:
        max_iters = 100
        
    # Cooling schedule: T(t) = T_start * (T_end/T_start)^(t/max_iters)
    alpha = np.log(T_end / T_start) / max_iters
    
    T = T_start
    no_improve = 0
    max_no_improve = 200
    
    for i in range(max_iters):
        if time.time() - start_time > max_time:
            break
        
        # Perturb a random subset of scores (e.g., 5% of vertices)
        num_perturb = max(1, int(0.05 * n))
        indices = rng.integers(0, n, size=num_perturb)
        noise = rng.normal(0, 0.5, size=num_perturb)
        
        old_scores = current_scores[indices].copy()
        current_scores[indices] += noise
        
        new_val = simulate_fast(current_scores)
        delta = new_val - current_val
        
        # Acceptance criterion
        if delta > 0 or (T > 0 and rng.random() < np.exp(delta / T)):
            current_val = new_val
            no_improve = 0
            if current_val > best_val:
                best_val = current_val
                best_scores = current_scores.copy()
        else:
            # Reject: revert
            current_scores[indices] = old_scores
            no_improve += 1
            
        # Cool down
        T *= np.exp(alpha)
        
        # Optional: restart if stuck for too long? 
        # For now, just let it cool.
        
        # Early stop if T is very low and no improvement
        if no_improve > max_no_improve and T < 0.01:
            break
    
    return best_scores
