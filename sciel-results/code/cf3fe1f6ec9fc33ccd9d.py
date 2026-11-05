import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 128
    min_sum = 128
    max_sum = 256
    target_sums = np.arange(min_sum, max_sum + 1)
    num_targets = len(target_sums)
    
    # Precompute indices for target sums
    sum_to_idx = np.full(2 * n + 1, -1, dtype=np.int32)
    for i, s in enumerate(target_sums):
        sum_to_idx[s] = i
        
    def get_counts_binary(v):
        conv = np.convolve(v, v, mode='full')
        counts_ordered = conv[128:257]
        s_vals = np.arange(128, 257)
        is_even = (s_vals % 2 == 0)
        diag_terms = np.zeros_like(counts_ordered, dtype=np.int32)
        even_indices = np.where(is_even)[0]
        if len(even_indices) > 0:
            s_even = s_vals[even_indices]
            half_s = s_even // 2
            diag_terms[even_indices] = v[half_s]
        counts_unordered = (counts_ordered + diag_terms) // 2
        return counts_unordered

    def cost_function(v):
        counts = get_counts_binary(v)
        peak = np.max(counts)
        missing = np.sum(counts == 0)
        if missing > 0:
            return 10000 + missing * 10 + peak
        return peak

    def differential_evolution():
        NP = 40
        D = 129
        F = 0.7
        CR = 0.8
        
        pop = np.zeros((NP, D), dtype=np.int32)
        for i in range(NP):
            size = random.randint(40, 60)
            indices = random.sample(range(D), size)
            pop[i, indices] = 1
            
        costs = np.array([cost_function(pop[i]) for i in range(NP)])
        best_idx = np.argmin(costs)
        best_v = pop[best_idx].copy()
        best_cost = costs[best_idx]
        
        while time.time() < deadline - 5.0:
            trial_pop = np.zeros((NP, D), dtype=np.int32)
            for i in range(NP):
                r1, r2, r3 = np.random.choice(NP, 3, replace=False)
                v_cont = pop[r1] + F * (pop[r2] - pop[r3])
                mask = np.random.rand(D) < CR
                if not np.any(mask):
                    j = random.randint(0, D-1)
                    mask[j] = True
                v_trial = np.where(mask, np.round(v_cont).astype(np.int32), pop[i])
                v_trial = np.clip(v_trial, 0, 1)
                trial_pop[i] = v_trial
                
            trial_costs = np.array([cost_function(trial_pop[i]) for i in range(NP)])
            
            for i in range(NP):
                if trial_costs[i] < costs[i]:
                    pop[i] = trial_pop[i]
                    costs[i] = trial_costs[i]
                    if trial_costs[i] < best_cost:
                        best_cost = trial_costs[i]
                        best_v = pop[i].copy()
                        
        return best_v, best_cost

    def local_search(v, max_time=2.0):
        v = v.copy()
        current_cost = cost_function(v)
        start = time.time()
        
        improved = True
        while improved and time.time() < start + max_time:
            improved = False
            for _ in range(50):
                if random.random() < 0.5:
                    candidates = np.where(v == 0)[0]
                    if len(candidates) == 0: continue
                    x = random.choice(candidates)
                    v_new = v.copy()
                    v_new[x] = 1
                    new_cost = cost_function(v_new)
                    if new_cost < current_cost:
                        v = v_new
                        current_cost = new_cost
                        improved = True
                else:
                    candidates = np.where(v == 1)[0]
                    if len(candidates) == 0: continue
                    x = random.choice(candidates)
                    v_new = v.copy()
                    v_new[x] = 0
                    new_cost = cost_function(v_new)
                    if new_cost < current_cost:
                        v = v_new
                        current_cost = new_cost
                        improved = True
                        
            for _ in range(50):
                ones = np.where(v == 1)[0]
                zeros = np.where(v == 0)[0]
                if len(ones) == 0 or len(zeros) == 0: continue
                x = random.choice(ones)
                y = random.choice(zeros)
                v_new = v.copy()
                v_new[x] = 0
                v_new[y] = 1
                new_cost = cost_function(v_new)
                if new_cost < current_cost:
                    v = v_new
                    current_cost = new_cost
                    improved = True
                    
        return v, current_cost

    best_v, best_cost = differential_evolution()
    
    if time.time() < deadline - 2.0:
        best_v, best_cost = local_search(best_v, max_time=deadline - time.time() - 2.0)
        
    A = set(np.where(best_v == 1)[0].tolist())
    
    final_counts = get_counts_binary(best_v)
    if np.any(final_counts == 0):
        return list(range(64, 129))
        
    return list(A)
