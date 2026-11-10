import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Target n = 34
    n = 34
    
    # Precompute triples
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Precompute K4s and K5s
    k4_list = list(combinations(range(n), 4))
    num_k4 = len(k4_list)
    k5_list = list(combinations(range(n), 5))
    num_k5 = len(k5_list)
    
    # Map each triple to the K4s and K5s it belongs to
    triple_to_k4s = [[] for _ in range(num_triples)]
    triple_to_k5s = [[] for _ in range(num_triples)]
    
    # k4_data[i] = list of 4 triple indices in K4 i
    # k5_data[i] = list of 10 triple indices in K5 i
    k4_data = []
    k5_data = []
    
    for i, k4 in enumerate(k4_list):
        ts = []
        for j in range(4):
            t = tuple(sorted(k4[:j] + k4[j+1:]))
            idx = triple_to_idx[t]
            ts.append(idx)
            triple_to_k4s[idx].append(i)
        k4_data.append(ts)
        
    for i, k5 in enumerate(k5_list):
        ts = []
        for combo in combinations(range(5), 3):
            t = tuple(sorted([k5[j] for j in combo]))
            idx = triple_to_idx[t]
            ts.append(idx)
            triple_to_k5s[idx].append(i)
        k5_data.append(ts)

    # Convert to numpy arrays for fast vectorized operations
    k4_arr = np.array(k4_data, dtype=np.int32)
    k5_arr = np.array(k5_data, dtype=np.int32)
    
    # We will use a "score" function that is the number of violations.
    # Red K4 violation: count of reds == 4
    # Blue K5 violation: count of blues == 10
    
    def get_counts_and_violations(colors):
        # colors: 0=Red, 1=Blue
        # k4_counts: number of reds in each K4.
        # k5_counts: number of blues in each K5.
        
        # k4_counts[i] = sum(1 for t in k4_data[i] if colors[t]==0)
        # k5_counts[i] = sum(1 for t in k5_data[i] if colors[t]==1)
        
        # Vectorized:
        # red_mask = (colors == 0)
        # k4_counts = red_mask[k4_arr].sum(axis=1)
        
        # blue_mask = (colors == 1)
        # k5_counts = blue_mask[k5_arr].sum(axis=1)
        
        red_mask = (colors == 0)
        k4_counts = red_mask[k4_arr].sum(axis=1)
        
        blue_mask = (colors == 1)
        k5_counts = blue_mask[k5_arr].sum(axis=1)
        
        violations = np.sum(k4_counts == 4) + np.sum(k5_counts == 10)
        
        return k4_counts, k5_counts, violations

    # Initialize random coloring
    colors = np.random.randint(0, 2, size=num_triples).astype(np.int8)
    k4_counts, k5_counts, current_violations = get_counts_and_violations(colors)
    
    best_violations = current_violations
    best_colors = colors.copy()
    
    # WalkSAT-like local search with restarts
    # To make it faster, we maintain the counts incrementally.
    
    def flip_incremental(idx, colors, k4_counts, k5_counts, current_violations):
        old_val = colors[idx]
        new_val = 1 - old_val
        colors[idx] = new_val
        
        # Delta for Red count:
        # If 0->1 (R->B), Red count decreases by 1.
        # If 1->0 (B->R), Red count increases by 1.
        delta_red = 1 if new_val == 0 else -1
        
        # Delta for Blue count:
        # If 0->1 (R->B), Blue count increases by 1.
        # If 1->0 (B->R), Blue count decreases by 1.
        delta_blue = 1 if new_val == 1 else -1
        
        # Update K4s
        for k4_idx in triple_to_k4s[idx]:
            old_count = k4_counts[k4_idx]
            new_count = old_count + delta_red
            k4_counts[k4_idx] = new_count
            
            old_viol = 1 if old_count == 4 else 0
            new_viol = 1 if new_count == 4 else 0
            current_violations += (new_viol - old_viol)
            
        # Update K5s
        for k5_idx in triple_to_k5s[idx]:
            old_count = k5_counts[k5_idx]
            new_count = old_count + delta_blue
            k5_counts[k5_idx] = new_count
            
            old_viol = 1 if old_count == 10 else 0
            new_viol = 1 if new_count == 10 else 0
            current_violations += (new_viol - old_viol)
            
        return current_violations

    # Maintain sets of violated clauses for faster selection
    violated_k4s = set(np.where(k4_counts == 4)[0])
    violated_k5s = set(np.where(k5_counts == 10)[0])
    
    def flip_with_sets(idx, colors, k4_counts, k5_counts, violated_k4s, violated_k5s):
        old_val = colors[idx]
        new_val = 1 - old_val
        colors[idx] = new_val
        
        delta_red = 1 if new_val == 0 else -1
        delta_blue = 1 if new_val == 1 else -1
        
        for k4_idx in triple_to_k4s[idx]:
            old_count = k4_counts[k4_idx]
            new_count = old_count + delta_red
            k4_counts[k4_idx] = new_count
            
            was_viol = (old_count == 4)
            is_viol = (new_count == 4)
            
            if was_viol and not is_viol:
                violated_k4s.discard(k4_idx)
            elif not was_viol and is_viol:
                violated_k4s.add(k4_idx)
                
        for k5_idx in triple_to_k5s[idx]:
            old_count = k5_counts[k5_idx]
            new_count = old_count + delta_blue
            k5_counts[k5_idx] = new_count
            
            was_viol = (old_count == 10)
            is_viol = (new_count == 10)
            
            if was_viol and not is_viol:
                violated_k5s.discard(k5_idx)
            elif not was_viol and is_viol:
                violated_k5s.add(k5_idx)
                
        return len(violated_k4s) + len(violated_k5s)

    # Main search loop
    max_restarts = 2000
    # We use a hybrid approach:
    # 1. Min-Conflicts style: Pick a violated clause, pick a random variable in it, flip.
    # 2. If stuck, restart.
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize state for this restart
        if restart > 0 and random.random() < 0.3 and best_violations < 100:
            # Perturb best
            colors = best_colors.copy()
            k4_counts, k5_counts, current_violations = get_counts_and_violations(colors)
            violated_k4s = set(np.where(k4_counts == 4)[0])
            violated_k5s = set(np.where(k5_counts == 10)[0])
            
            # Do a few random flips to escape local min
            for _ in range(10):
                idx = random.randint(0, num_triples - 1)
                current_violations = flip_with_sets(idx, colors, k4_counts, k5_counts, violated_k4s, violated_k5s)
        else:
            # Random restart
            colors = np.random.randint(0, 2, size=num_triples).astype(np.int8)
            k4_counts, k5_counts, current_violations = get_counts_and_violations(colors)
            violated_k4s = set(np.where(k4_counts == 4)[0])
            violated_k5s = set(np.where(k5_counts == 10)[0])
            
        if current_violations == 0:
            return {'n': 34, 'colors': colors.tolist()}
            
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors.copy()
            
        # WalkSAT Loop
        run_start = time.time()
        max_steps = 20000 # Cap steps per restart to ensure progress
        steps = 0
        
        while steps < max_steps and time.time() < run_start + 1.5:
            if current_violations == 0:
                return {'n': 34, 'colors': colors.tolist()}
                
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors.copy()
                
            # Pick a violated clause
            # We have violated_k4s and violated_k5s sets.
            # Pick one randomly.
            
            # To avoid O(N) list conversion, we can pick a random triple from the union of triples in violated clauses?
            # Or just pick a random violated clause index.
            
            # Since sets don't support random.choice directly in a fast way without conversion,
            # and the number of violations is usually small in the later stages,
            # we can convert to list if the set is small, or just pick a random triple and check if it's in a violation.
            
            # Better heuristic:
            # If there are violations, pick a random triple that is part of a violation.
            # How to pick fast?
            # We can maintain a list of "active" violated triples?
            # Or just pick a random triple, and if it's not in a violation, pick another?
            # This is inefficient if violations are rare.
            
            # Let's convert the violated sets to lists if they are not too big.
            # If they are big, we are in the early stages, so random flips are fine.
            
            if len(violated_k4s) + len(violated_k5s) < 1000:
                # Convert to list for fast random choice
                all_violated = list(violated_k4s) + list(violated_k5s)
                # This is a mix of K4 and K5 indices, which are different types?
                # No, they are just indices. But they refer to different lists.
                # We need to know which type it is to find the triples.
                
                # Let's pick a random violated K4 or K5.
                if random.random() < 0.5 and violated_k4s:
                    k4_idx = random.choice(list(violated_k4s))
                    # Pick a random triple in this K4
                    t_idx = random.choice(k4_data[k4_idx])
                elif violated_k5s:
                    k5_idx = random.choice(list(violated_k5s))
                    # Pick a random triple in this K5
                    t_idx = random.choice(k5_data[k5_idx])
                else:
                    # Should not happen if current_violations > 0
                    t_idx = random.randint(0, num_triples - 1)
            else:
                # Early stage, just pick random triple
                t_idx = random.randint(0, num_triples - 1)
                
            current_violations = flip_with_sets(t_idx, colors, k4_counts, k5_counts, violated_k4s, violated_k5s)
            steps += 1
            
        if best_violations == 0:
            break

    if best_violations == 0:
        return {'n': 34, 'colors': best_colors.tolist()}
    else:
        # Fallback to smaller n if we failed
        # n=9 is a safe bet
        n_fallback = 9
        colors_fallback = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n_fallback), 3)]
        return {'n': n_fallback, 'colors': colors_fallback}

# EVOLVE-BLOCK-END
