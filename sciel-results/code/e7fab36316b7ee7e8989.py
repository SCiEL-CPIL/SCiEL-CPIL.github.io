import time
import math
import random
import numpy as np

N = 64

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 295.0  # Safety margin
    
    # Precompute coordinates
    xs = np.array([i // N for i in range(N * N)], dtype=np.int32)
    ys = np.array([i % N for i in range(N * N)], dtype=np.int32)
    
    all_points = list(range(N * N))
    
    # Precompute all pairwise squared distances
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist_matrix = (dx * dx + dy * dy).astype(np.int32)
    
    def solve_once(seed, time_limit):
        start = time.time()
        limit = start + time_limit
        
        rng = random.Random(seed)
        
        # Data structures
        selected = set()
        selected_list = []
        # dist_sets[i] = set of squared distances from i to other selected points
        dist_sets = {}
        
        # To speed up can_add, we can use a global distance counter?
        # No, the condition is specific to the pair.
        
        def can_add(p):
            if p in selected:
                return False
            
            # Get distances from p to all selected points
            # selected_indices = np.array(list(selected))
            # d_p_to_s = dist_matrix[p, selected_indices]
            
            # Condition 1: Distances from p to S must be unique (no duplicate distances)
            # Condition 2: For each q in S, d(p,q) must not be in dist_sets[q]
            
            # Optimization: Iterate through selected_list
            # If selected_list is small, this is fast.
            # If large, we might need to optimize.
            
            seen_dists = set()
            for q in selected_list:
                d = dist_matrix[p, q]
                
                # Check if d is already seen (x as apex)
                if d in seen_dists:
                    return False
                seen_dists.add(d)
                
                # Check if d is in dist_sets[q] (q as apex)
                if d in dist_sets[q]:
                    return False
                    
            return True

        def add(p):
            selected.add(p)
            selected_list.append(p)
            dist_sets[p] = set()
            # Update dist_sets for p and existing points
            for q in selected_list[:-1]:
                d = dist_matrix[p, q]
                dist_sets[p].add(d)
                dist_sets[q].add(d)

        def remove(p):
            if p not in selected:
                return
            selected.discard(p)
            selected_list.remove(p)
            # Update dist_sets
            for q in selected_list:
                d = dist_matrix[p, q]
                dist_sets[q].discard(d)
            del dist_sets[p]

        # Greedy Initialization
        orders = []
        
        # 1. Random
        ord1 = all_points.copy()
        rng.shuffle(ord1)
        orders.append(ord1)
        
        # 2. Center-out
        center = N // 2
        ord2 = sorted(all_points, key=lambda i: (xs[i]-center)**2 + (ys[i]-center)**2)
        orders.append(ord2)
        
        # 3. Corner-in
        ord3 = sorted(all_points, key=lambda i: min(xs[i], N-1-xs[i]) + min(ys[i], N-1-ys[i]))
        orders.append(ord3)
        
        # 4. Diagonal
        ord4 = sorted(all_points, key=lambda i: abs(xs[i] - ys[i]))
        orders.append(ord4)

        best_greedy_size = 0
        best_greedy_selected = set()
        best_greedy_dist_sets = {}
        best_greedy_list = []

        for order in orders:
            selected.clear()
            selected_list.clear()
            dist_sets.clear()
            
            for p in order:
                if can_add(p):
                    add(p)
            
            if len(selected) > best_greedy_size:
                best_greedy_size = len(selected)
                best_greedy_selected = set(selected)
                best_greedy_dist_sets = {k: set(v) for k, v in dist_sets.items()}
                best_greedy_list = list(selected_list)

        # Restore best greedy state
        selected = best_greedy_selected
        selected_list = best_greedy_list
        dist_sets = best_greedy_dist_sets
        
        unselected = [p for p in all_points if p not in selected]
        
        best_size = len(selected)
        best_solution = [(i // N, i % N) for i in selected]
        
        no_improve_count = 0
        max_no_improve = 30
        
        # Tabu list
        tabu_list = {}
        tabu_tenure = 50
        
        while time.time() < limit:
            improved = False
            
            # 1. Try to add any unselected point
            if unselected:
                # Sort unselected by potential "easiness"? 
                # Just shuffle for randomness
                rng.shuffle(unselected)
                # Try to add the first one that fits
                # We can try multiple candidates if the first fails
                candidates_to_try = min(50, len(unselected))
                for p in unselected[:candidates_to_try]:
                    if can_add(p):
                        add(p)
                        unselected.remove(p)
                        improved = True
                        break # Restart loop to re-shuffle? Or continue?
                        # Continuing might be faster if we just keep scanning
                        # But adding p changes validity of others.
                        # Let's break and let the outer loop handle it.
            
            if improved:
                no_improve_count = 0
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_solution = [(i // N, i % N) for i in selected]
                continue
            
            # 2. Try Swap (1-out, 1-in)
            if selected_list and unselected:
                # Pick a random point to remove
                # Prefer points that are "central" or have high degree?
                # For now, random.
                p_out = rng.choice(selected_list)
                
                # Check if p_out is tabu
                if p_out in tabu_list and tabu_list[p_out] > 0:
                    # Try another one
                    pass # Simplified: just proceed, tabu is soft constraint here
                
                found_swap = False
                # Try to find a point to add
                # We want to add a point that was blocked by p_out
                # So we should prioritize points that had conflicts with p_out
                # For now, random sampling from unselected
                for _ in range(20):
                    if not unselected:
                        break
                    p_in = rng.choice(unselected)
                    
                    # Check if swap is possible
                    remove(p_out)
                    if can_add(p_in):
                        add(p_in)
                        unselected.remove(p_in)
                        unselected.append(p_out)
                        found_swap = True
                        improved = True
                        # Update tabu
                        tabu_list[p_out] = tabu_tenure
                        break
                    else:
                        add(p_out)
                
                if found_swap:
                    no_improve_count = 0
                    if len(selected) > best_size:
                        best_size = len(selected)
                        best_solution = [(i // N, i % N) for i in selected]
                    continue
            
            # 3. Kick (Perturbation)
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                no_improve_count = 0
                # Kick: remove k random points
                k = rng.randint(3, 7)
                if len(selected_list) > k:
                    to_remove = rng.sample(selected_list, k)
                    for p in to_remove:
                        remove(p)
                        unselected.append(p)
                        tabu_list[p] = tabu_tenure
                    
                    # Refill greedily
                    # Shuffle unselected
                    rng.shuffle(unselected)
                    for p in unselected:
                        if can_add(p):
                            add(p)
                            unselected.remove(p)
                            # Don't break, try to add more
                            # But this is O(N) per add, so N^2 total.
                            # Might be slow. Let's limit refill attempts.
                            # Actually, can_add is O(K). So total O(K^2).
                            # K is ~10-20. So 400 ops. Fast.
                            
            # Decay tabu
            for p in list(tabu_list.keys()):
                if tabu_list[p] > 0:
                    tabu_list[p] -= 1
                    if tabu_list[p] == 0:
                        del tabu_list[p]

        return best_solution

    best_result = []
    best_size = 0
    
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808, 909, 111, 222, 333]
    
    for seed in seeds:
        if time.time() > deadline - 10:
            break
        
        remaining = deadline - time.time()
        runs_left = len(seeds) - seeds.index(seed)
        # Allocate more time to later seeds? Or equal?
        # Let's give each seed a fair share, but ensure we finish.
        time_per_run = min(15.0, max(5.0, (remaining - 5.0) / runs_left))
        
        if time.time() + time_per_run > deadline - 2:
            time_per_run = max(1.0, deadline - time.time() - 2)
            
        result = solve_once(seed, time_per_run)
        
        if len(result) > best_size:
            best_size = len(result)
            best_result = result
            
    return best_result

# EVOLVE-BLOCK-END
