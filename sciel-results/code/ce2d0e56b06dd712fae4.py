import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin to stay under 60s
    
    n = 128
    targets = range(128, 257)
    target_min = 128
    target_max = 256
    
    # Initialize A
    # A must cover [128, 256].
    # Minimum size is around 65. Let's start with a random set of size 70-80 
    # that is likely to cover, or use a greedy initialization.
    
    def calculate_counts(A):
        counts = np.zeros(257, dtype=int)
        A_list = list(A)
        L = len(A_list)
        for i in range(L):
            ai = A_list[i]
            for j in range(i, L):
                aj = A_list[j]
                s = ai + aj
                if target_min <= s <= target_max:
                    counts[s] += 1
        return counts

    def get_metrics(counts):
        target_counts = counts[target_min:target_max+1]
        peak = np.max(target_counts)
        uncovered = np.sum(target_counts == 0)
        # Sum of squares for flattening
        ss = np.sum(target_counts**2)
        return peak, uncovered, ss

    # Greedy Initialization to ensure coverage
    A = set()
    uncovered = set(targets)
    
    # Start with a few elements to get coverage going
    # 128 covers 256. 0 covers nothing new initially unless paired with others.
    # Let's pick elements that cover high sums first, then fill gaps.
    # Actually, a random set of size 80 usually covers [128, 256] easily.
    # Let's try a structured start: A = {0, 1, ..., 10} U {118, ..., 128} ?
    # Sums: 0+128=128, 10+118=128.
    # Let's just use a random set of size 80 and check coverage.
    
    while True:
        A = set(random.sample(range(n + 1), 80))
        counts = calculate_counts(A)
        peak, unc, ss = get_metrics(counts)
        if unc == 0:
            break
        # If not covered, add elements greedily
        while unc > 0:
            best_x = -1
            best_unc_reduction = 0
            for x in range(n + 1):
                if x in A:
                    continue
                # How many uncovered targets does x cover?
                # x covers k if k-x in A.
                # We can just simulate adding x to a copy? Too slow.
                # Approximate: x helps cover k if k-x is in A.
                # Let's just pick a random x not in A and add it if it helps.
                # Or simpler: if we fail random init, use the previous program's greedy logic.
                
            # Fallback to simple greedy if random fails (unlikely for size 80)
            # But to be safe, let's just run the previous greedy if this loop takes too long.
            # Actually, size 80 from 129 has very high probability of covering [128, 256].
            # Probability that a specific k is NOT covered:
            # P(k not in A+A) is very low for |A|=80.
            # So this loop should break immediately.
            if len(A) > 100:
                 # Force add if stuck
                 for x in range(n+1):
                     if x not in A:
                         A.add(x)
                         break
            else:
                 # Add random
                 x = random.choice([i for i in range(n+1) if i not in A])
                 A.add(x)
                 counts = calculate_counts(A)
                 peak, unc, ss = get_metrics(counts)

    curr_counts = calculate_counts(A)
    curr_peak, curr_unc, curr_ss = get_metrics(curr_counts)
    
    best_A = set(A)
    best_peak = curr_peak
    best_ss = curr_ss
    best_counts = curr_counts.copy()
    
    # Simulated Annealing
    T = 5.0
    cooling = 0.995
    min_T = 0.01
    
    # Precompute neighbors for speed? No, dynamic A.
    # We need to update counts efficiently.
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            T = min_T
            # Restart with higher T if stuck? Or just continue.
            
        # Pick a move type
        # 1. Swap (most likely)
        # 2. Remove
        # 3. Add
        
        move_type = random.random()
        A_list = list(A)
        L = len(A_list)
        
        if L == 0:
            break
            
        if move_type < 0.7: # Swap
            if L < 2: continue
            x = random.choice(A_list)
            # Find y not in A
            # Optimization: maintain a list of non-A elements?
            # For now, random sampling from range is O(1) expected.
            while True:
                y = random.randint(0, n)
                if y not in A:
                    break
            
            # Apply swap tentatively
            # x is removed, y is added.
            # Pairs involving x: (x, a) for a in A, a != x. And (x, x).
            # Pairs involving y: (y, a) for a in A, a != x (since x is gone, but y is added to A\{x}).
            # Wait, A_new = (A - {x}) U {y}.
            # Pairs in A_new:
            # 1. Pairs in A - {x, y} (y not in A so just A-{x}). These remain.
            # 2. Pairs involving x in A are removed.
            # 3. Pairs involving y in A_new are added.
            
            # Delta calculation:
            # Remove x:
            #   For each a in A such that a != x: count[x+a] -= 1
            #   count[2*x] -= 1
            # Add y:
            #   For each a in A such that a != x: count[y+a] += 1
            #   count[2*y] += 1
            
            # Note: If y is in A, this logic fails. But we ensured y not in A.
            # So the set of "other" elements is A - {x}.
            
            trial_counts = curr_counts.copy()
            others = [a for a in A if a != x]
            
            for a in others:
                s_remove = x + a
                if target_min <= s_remove <= target_max:
                    trial_counts[s_remove] -= 1
                
                s_add = y + a
                if target_min <= s_add <= target_max:
                    trial_counts[s_add] += 1
            
            s_remove_2x = 2 * x
            if target_min <= s_remove_2x <= target_max:
                trial_counts[s_remove_2x] -= 1
            
            s_add_2y = 2 * y
            if target_min <= s_add_2y <= target_max:
                trial_counts[s_add_2y] += 1
            
            new_peak, new_unc, new_ss = get_metrics(trial_counts)
            
            # Check validity
            if new_unc == 0:
                # Evaluate acceptance
                delta_peak = new_peak - curr_peak
                if delta_peak < 0:
                    # Accept
                    curr_counts = trial_counts
                    curr_peak = new_peak
                    curr_ss = new_ss
                    A.remove(x)
                    A.add(y)
                    
                    if curr_peak < best_peak or (curr_peak == best_peak and curr_ss < best_ss):
                        best_peak = curr_peak
                        best_ss = curr_ss
                        best_A = set(A)
                        best_counts = curr_counts.copy()
                elif delta_peak == 0:
                    if new_ss < curr_ss:
                        # Accept if flatter
                        curr_counts = trial_counts
                        curr_ss = new_ss
                        A.remove(x)
                        A.add(y)
                        
                        if curr_peak < best_peak or (curr_peak == best_peak and curr_ss < best_ss):
                            best_peak = curr_peak
                            best_ss = curr_ss
                            best_A = set(A)
                            best_counts = curr_counts.copy()
                    # If ss is worse, reject (or accept with prob? No, keep strict for ss)
                else:
                    # delta_peak > 0
                    # Accept with prob exp(-delta_peak / T)
                    if random.random() < np.exp(-delta_peak / T):
                        curr_counts = trial_counts
                        curr_peak = new_peak
                        curr_ss = new_ss
                        A.remove(x)
                        A.add(y)
        elif move_type < 0.85: # Remove
            if L <= 1: continue
            x = random.choice(A_list)
            trial_counts = curr_counts.copy()
            others = [a for a in A if a != x]
            
            for a in others:
                s_remove = x + a
                if target_min <= s_remove <= target_max:
                    trial_counts[s_remove] -= 1
            
            s_remove_2x = 2 * x
            if target_min <= s_remove_2x <= target_max:
                trial_counts[s_remove_2x] -= 1
            
            new_peak, new_unc, new_ss = get_metrics(trial_counts)
            
            if new_unc == 0:
                delta_peak = new_peak - curr_peak
                if delta_peak <= 0:
                    curr_counts = trial_counts
                    curr_peak = new_peak
                    curr_ss = new_ss
                    A.remove(x)
                    
                    if curr_peak < best_peak or (curr_peak == best_peak and curr_ss < best_ss):
                        best_peak = curr_peak
                        best_ss = curr_ss
                        best_A = set(A)
                        best_counts = curr_counts.copy()
        else: # Add
            # Add y not in A
            while True:
                y = random.randint(0, n)
                if y not in A:
                    break
            
            trial_counts = curr_counts.copy()
            
            for a in A:
                s_add = y + a
                if target_min <= s_add <= target_max:
                    trial_counts[s_add] += 1
            
            s_add_2y = 2 * y
            if target_min <= s_add_2y <= target_max:
                trial_counts[s_add_2y] += 1
            
            new_peak, new_unc, new_ss = get_metrics(trial_counts)
            
            # Adding usually increases peak, so only accept if it allows future improvement?
            # In SA, we might accept worse peak to explore.
            # But coverage is already satisfied, so adding doesn't help coverage.
            # It only increases complexity.
            # We generally don't want to add unless it's part of a swap.
            # However, sometimes a "dead end" in A requires adding an element to bridge a gap
            # for a subsequent swap.
            # Let's accept with very low probability or only if peak doesn't increase.
            delta_peak = new_peak - curr_peak
            if delta_peak <= 0:
                curr_counts = trial_counts
                curr_peak = new_peak
                curr_ss = new_ss
                A.add(y)
                
                if curr_peak < best_peak or (curr_peak == best_peak and curr_ss < best_ss):
                    best_peak = curr_peak
                    best_ss = curr_ss
                    best_A = set(A)
                    best_counts = curr_counts.copy()
            elif random.random() < np.exp(-delta_peak / T):
                curr_counts = trial_counts
                curr_peak = new_peak
                curr_ss = new_ss
                A.add(y)
                
                if curr_peak < best_peak or (curr_peak == best_peak and curr_ss < best_ss):
                    best_peak = curr_peak
                    best_ss = curr_ss
                    best_A = set(A)
                    best_counts = curr_counts.copy()

        T *= cooling

    return sorted(list(best_A))
