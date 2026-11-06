def construct():
    import time
    import random

    deadline = time.time() + 140.0

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        if not S:
            return True
        S_list = list(S)
        S_set = set(S_list)
        for i, a in enumerate(S_list):
            for j in range(i, len(S_list)):
                b = S_list[j]
                if a + b in S_set:
                    return False
        return True

    def can_add(S, x):
        # Check if adding x to S violates sum-free property
        # 1. x + x = 2x in S?
        if 2 * x in S:
            return False
        # 2. x + y in S for any y in S?
        for y in S:
            if x + y in S:
                return False
        # 3. y + z = x for any y, z in S? Equivalent to: exists y in S such that x - y in S
        for y in S:
            if x - y in S:
                return False
        return True

    def solve_instance(vals, time_limit):
        n = len(vals)
        vals_set = set(vals)
        start_time = time.time()
        
        # 1. Generate diverse initial candidates
        candidates = []
        
        # Structural heuristics
        odds = [x for x in vals if x % 2 != 0]
        candidates.append(set(odds))
        
        pos = [x for x in vals if x > 0]
        candidates.append(set(pos))
        
        neg = [x for x in vals if x < 0]
        candidates.append(set(neg))
        
        sorted_vals = sorted(vals, reverse=True)
        top_half = sorted_vals[:n//2 + 1]
        candidates.append(set(top_half))
        
        bottom_half = sorted_vals[-(n//2 + 1):]
        candidates.append(set(bottom_half))
        
        sorted_abs = sorted(vals, key=lambda x: -abs(x))
        top_abs = sorted_abs[:n//2 + 1]
        candidates.append(set(top_abs))
        
        bottom_abs = sorted_abs[-(n//2 + 1):]
        candidates.append(set(bottom_abs))
        
        # Random samples
        rng = random.Random(42)
        for _ in range(10):
            sample = rng.sample(vals, n//2 + 1)
            candidates.append(set(sample))
        
        # Greedy constructions with different orderings
        orderings = [
            sorted(vals, key=lambda x: -abs(x)),
            sorted(vals, key=lambda x: x),
            sorted(vals, key=lambda x: -x),
            sorted(vals, key=lambda x: abs(x)),
        ]
        
        for ordering in orderings:
            S = set()
            for x in ordering:
                if can_add(S, x):
                    S.add(x)
            candidates.append(S)
            
        # Verify and filter candidates
        valid_candidates = []
        for S in candidates:
            if is_sum_free(S):
                valid_candidates.append(S)
            else:
                # Try to repair by removing conflicting elements? 
                # Or just skip. Skipping is safer and faster.
                # A simple repair: remove elements that cause conflicts.
                # But let's just skip invalid ones for now to maintain speed.
                pass
        
        if not valid_candidates:
             # Fallback: just odds
             valid_candidates = [set(x for x in vals if x % 2 != 0)]

        # Sort by size descending to prioritize good starting points
        valid_candidates.sort(key=len, reverse=True)
        
        best_S = valid_candidates[0]
        best_size = len(best_S)
        
        # 2. Local Search / Hill Climbing with Random Restarts
        # We will iterate through the best candidates and try to improve them.
        # Improvement move: 
        #   a) Try to add a random element not in S. If it fits, add it.
        #   b) If no element can be added, try a "swap": remove one element, try to add another.
        #      Or simply: remove a random element, then greedily fill.
        
        def improve(S, max_time):
            S = set(S)
            local_rng = random.Random(hash(frozenset(S)) % (2**32))
            last_improvement_time = time.time()
            
            # Phase 1: Greedy Add
            changed = True
            while changed and time.time() - start_time < max_time:
                changed = False
                # Shuffle to avoid bias? Or just iterate.
                # Iterating over all not-in-S is O(n^2) per pass.
                # For n=240, 240*240 = 57600 ops. Fast enough.
                for x in vals_set:
                    if x in S:
                        continue
                    if can_add(S, x):
                        S.add(x)
                        changed = True
                        last_improvement_time = time.time()
                        # Don't break, keep adding if possible? 
                        # Actually, adding one might block others. 
                        # Standard greedy adds one and restarts or continues.
                        # Continuing is fine.
            
            # Phase 2: Random Walk / Swaps
            # If we are stuck, try removing an element and re-greedy-ing.
            # This can escape local optima where a specific element blocks many others.
            while time.time() - start_time < max_time:
                # Try a random swap or removal
                if len(S) > 1:
                    # Pick a random element to remove
                    to_remove = local_rng.choice(list(S))
                    S.remove(to_remove)
                    
                    # Greedy fill
                    added = False
                    for x in vals_set:
                        if x in S:
                            continue
                        if can_add(S, x):
                            S.add(x)
                            added = True
                    
                    if added:
                        last_improvement_time = time.time()
                    else:
                        # If nothing was added, we lost an element. 
                        # Put it back? Or keep the smaller set?
                        # Keeping the smaller set is risky. 
                        # Let's put it back if it doesn't help.
                        S.add(to_remove)
                
                # Check time
                if time.time() - last_improvement_time > 2.0:
                    # Stagnant for 2 seconds, break
                    break

            return S

        # Allocate time per candidate
        # We have `time_limit` total.
        # Let's spend time on the top few candidates.
        num_to_try = min(len(valid_candidates), 5)
        time_per_candidate = time_limit / num_to_try
        
        for i in range(num_to_try):
            if time.time() - start_time > time_limit - 2.0:
                break
                
            S = valid_candidates[i]
            # Pass remaining time for this candidate
            remaining_time = time_limit - (time.time() - start_time)
            if remaining_time <= 0:
                break
            
            improved_S = improve(S, remaining_time)
            
            if len(improved_S) > best_size:
                best_size = len(improved_S)
                best_S = improved_S
        
        return best_S

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = instance(size, seed)
        
        # Allocate time based on instance size
        # n=60: easy, n=120: medium, n=240: hard
        if size == 60:
            t_lim = 20.0
        elif size == 120:
            t_lim = 40.0
        else:
            t_lim = 60.0
            
        S = solve_instance(vals, t_lim)
        
        # Final verification
        if not is_sum_free(S):
            # Fallback to odds
            S = set(x for x in vals if x % 2 != 0)
        
        results.append(sorted(S))
        
        if time.time() > deadline:
            # If we ran out of time, return what we have
            break

    # Ensure we return 3 results
    while len(results) < 3:
        results.append([])

    return results[:3]
