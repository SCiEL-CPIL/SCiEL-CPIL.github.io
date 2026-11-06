import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    best_set = []
    best_size = 0
    
    def is_valid_addition(current_set, x, counts):
        """Check if adding x to current_set keeps max reps <= 2."""
        # We need to update counts for sums involving x
        # Sums: x + y for y in current_set, and 2*x
        # We must check if any of these sums would exceed 2.
        # Note: counts[s] is the number of representations of s in current_set.
        
        # Optimization: Precompute potential conflicts?
        # For now, direct check.
        
        # If x is already in set, invalid (but we assume x not in set)
        if x in current_set:
            return False
            
        new_sums = [x + y for y in current_set]
        new_sums.append(2 * x)
        
        # Check if any new sum would push count > 2
        # We need to know the current count for each sum.
        # If count[s] + delta[s] > 2, invalid.
        
        # Since we are adding only one element, delta[s] is 1 for each s in new_sums.
        # So we just check if counts.get(s, 0) >= 2 for any s in new_sums.
        # Wait, if counts.get(s, 0) is 2, adding one more makes it 3. Invalid.
        # If counts.get(s, 0) is 1, adding one more makes it 2. Valid.
        
        for s in new_sums:
            if counts.get(s, 0) >= 2:
                return False
        return True

    def get_counts(current_set):
        counts = {}
        for i, a in enumerate(current_set):
            for b in current_set[i:]:
                s = a + b
                counts[s] = counts.get(s, 0) + 1
        return counts

    def greedy_build(skip_prob=0.0):
        current_set = []
        counts = {}
        
        for x in range(1, 1001):
            if time.time() > deadline:
                return current_set, counts
            
            # Skip logic
            if skip_prob > 0 and random.random() < skip_prob:
                continue
                
            if is_valid_addition(current_set, x, counts):
                current_set.append(x)
                # Update counts
                for y in current_set[:-1]:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                s2 = 2 * x
                counts[s2] = counts.get(s2, 0) + 1
                
        return current_set, counts

    def local_search(current_set):
        current_set = sorted(current_set)
        counts = get_counts(current_set)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing each element and refilling
            for i in range(len(current_set)):
                if time.time() > deadline:
                    break
                x = current_set[i]
                # Remove x
                new_set = current_set[:i] + current_set[i+1:]
                # Rebuild counts for new_set? Expensive.
                # Better: Calculate impact of removing x.
                
                # Let's just rebuild counts for simplicity if set is small (~50)
                # 50^2 = 2500 ops. Very fast.
                new_counts = get_counts(new_set)
                
                # Try to add new elements greedily
                old_size = len(new_set)
                added_any = False
                for y in range(1, 1001):
                    if time.time() > deadline:
                        break
                    if y in new_set:
                        continue
                    if is_valid_addition(new_set, y, new_counts):
                        new_set.append(y)
                        for z in new_set[:-1]:
                            s = y + z
                            new_counts[s] = new_counts.get(s, 0) + 1
                        s2 = 2 * y
                        new_counts[s2] = new_counts.get(s2, 0) + 1
                        added_any = True
                        # We can stop adding if we reach a certain size or just continue greedily
                        # Greedy addition is deterministic given the set.
                        
                if len(new_set) > len(current_set):
                    current_set = new_set
                    counts = new_counts
                    improved = True
                    break # Restart loop with new set
                elif len(new_set) == len(current_set) and added_any:
                    # Same size, but maybe different composition allows further growth?
                    # We can accept if it's "better" in some heuristic, but for now just size.
                    pass

        return current_set

    # Run multiple strategies
    # 1. Pure Greedy
    s, c = greedy_build(0.0)
    s = local_search(s)
    if len(s) > best_size:
        best_set = s
        best_size = len(s)

    # 2. Greedy with various skip probabilities
    for p in [0.1, 0.2, 0.3, 0.5]:
        for _ in range(5): # 5 restarts per p
            if time.time() > deadline:
                break
            s, c = greedy_build(p)
            s = local_search(s)
            if len(s) > best_size:
                best_set = s
                best_size = len(s)

    # 3. Randomized Greedy: Pick random valid x from a window
    # This is harder to implement efficiently with the "skip" logic, 
    # but we can simulate by shuffling the candidate order? No, order matters for greedy.
    # Alternative: Start with a random subset and expand?
    
    # Let's try a "Randomized Local Search" directly
    # Start with greedy, then random swaps
    if time.time() < deadline:
        s = best_set.copy()
        for _ in range(100): # 100 swap attempts
            if time.time() > deadline:
                break
            if not s: continue
            # Pick a random element to remove
            idx = random.randint(0, len(s)-1)
            x = s[idx]
            new_set = s[:idx] + s[idx+1:]
            new_counts = get_counts(new_set)
            
            # Try to add a random candidate from 1..1000 not in new_set
            candidates = [y for y in range(1, 1001) if y not in new_set]
            if not candidates: continue
            
            # Try a few random candidates
            random.shuffle(candidates)
            for y in candidates[:20]: # Try 20 random candidates
                if is_valid_addition(new_set, y, new_counts):
                    # Check if this swap improves potential
                    # Simple metric: size stays same (1 removed, 1 added).
                    # But maybe this y allows adding more later?
                    # Let's just accept if valid, and run a quick greedy fill to see if size grows.
                    trial_set = new_set + [y]
                    trial_counts = dict(new_counts)
                    # Update counts for y
                    for z in new_set:
                        s_sum = y + z
                        trial_counts[s_sum] = trial_counts.get(s_sum, 0) + 1
                    s2 = 2*y
                    trial_counts[s2] = trial_counts.get(s2, 0) + 1
                    
                    # Greedy fill from trial_set
                    improved = False
                    for w in range(1, 1001):
                        if w in trial_set: continue
                        if is_valid_addition(trial_set, w, trial_counts):
                            trial_set.append(w)
                            for z in trial_set[:-1]:
                                s_sum = w + z
                                trial_counts[s_sum] = trial_counts.get(s_sum, 0) + 1
                            s2 = 2*w
                            trial_counts[s2] = trial_counts.get(s2, 0) + 1
                            improved = True
                            
                    if len(trial_set) > len(s):
                        s = trial_set
                        break
                    elif len(trial_set) == len(s) and random.random() < 0.1:
                        # Accept same size with small prob to escape local minima
                        s = trial_set
                        break
                        
            if len(s) > best_size:
                best_set = s
                best_size = len(s)

    return best_set
