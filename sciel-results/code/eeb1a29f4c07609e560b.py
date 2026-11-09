"""Rewritten program: maximum sunflower-free family via optimized greedy + local search."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 16
    k = 4
    
    # Generate all 4-element subsets as bitmasks
    all_sets = []
    for combo in combinations(range(1, n + 1), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
    
    N = len(all_sets)  # 1820
    
    # Precompute pairwise intersections
    inter = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            v = all_sets[i] & all_sets[j]
            inter[i][j] = v
            inter[j][i] = v
    
    def is_valid_triple(i, j, k):
        """Check if sets i,j,k form a sunflower"""
        return inter[i][j] == inter[i][k] == inter[j][k]
    
    def greedy_build(order, time_limit=None):
        chosen = []
        chosen_set = set()
        
        for idx in order:
            if time_limit and time.time() > time_limit:
                break
            s = all_sets[idx]
            ok = True
            # Check against all pairs in chosen
            for ai in range(len(chosen)):
                if not ok:
                    break
                a = all_sets[chosen[ai]]
                ia_s = a & s
                for bi in range(ai + 1, len(chosen)):
                    b = all_sets[chosen[bi]]
                    if inter[chosen[ai]][chosen[bi]] == ia_s == (b & s):
                        ok = False
                        break
            if ok:
                chosen.append(idx)
                chosen_set.add(idx)
        
        return chosen
    
    def try_add(sol_set, idx, time_limit=None):
        """Check if idx can be added to sol_set"""
        s = all_sets[idx]
        sol_list = list(sol_set)
        for i in range(len(sol_list)):
            if time_limit and time.time() > time_limit:
                return False
            a = all_sets[sol_list[i]]
            ia_s = a & s
            for j in range(i + 1, len(sol_list)):
                b = all_sets[sol_list[j]]
                if inter[sol_list[i]][sol_list[j]] == ia_s == (b & s):
                    return False
        return True
    
    best_solution = []
    best_size = 0
    
    random.seed(123)
    
    # Multiple greedy trials with different orderings
    num_greedy = 30
    for trial in range(num_greedy):
        if time.time() > deadline:
            break
        order = list(range(N))
        random.shuffle(order)
        sol = greedy_build(order, deadline)
        sol_set = set(sol)
        
        # Local improvement: add any set that fits
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(N):
                if idx in sol_set:
                    continue
                if try_add(sol_set, idx, deadline):
                    sol_set.add(idx)
                    improved = True
                    break
        
        if len(sol_set) > best_size:
            best_size = len(sol_set)
            best_solution = sol_set.copy()
    
    # Try swap improvements: remove one, add multiple
    if time.time() < deadline:
        sol_list = list(best_solution)
        for remove_idx in sol_list:
            if time.time() > deadline:
                break
            trial_set = best_solution - {remove_idx}
            added_count = 0
            for idx in range(N):
                if idx in trial_set:
                    continue
                if try_add(trial_set, idx, deadline):
                    trial_set.add(idx)
                    added_count += 1
            if len(trial_set) > len(best_solution):
                best_solution = trial_set
                best_size = len(best_solution)
    
    # Convert to sorted lists
    result = []
    for idx in best_solution:
        s = all_sets[idx]
        elems = [x for x in range(1, n + 1) if (s >> x) & 1]
        result.append(elems)
    
    return result
