import time
import random

def construct():
    t_start = time.time()
    DEADLINE = t_start + 78.0
    
    N = 4096
    
    def greedy_build(order, deadline, rng=None, randomize=False, window=50):
        vals = []
        existing_sums = set()
        available = list(order)
        
        while available:
            if time.time() > deadline:
                break
            
            k = len(vals)
            # Determine candidate pool
            if randomize and rng:
                # Pick a random subset of the next 'window' candidates to choose from
                # This diversifies the search path
                pool_size = min(window, len(available))
                candidates = available[:pool_size]
                # Find all valid candidates in this pool
                valid_candidates = []
                for x in candidates:
                    # Check if x is valid
                    conflict = False
                    new_sums = []
                    
                    s = 3 * x
                    if s in existing_sums:
                        conflict = True
                    else:
                        new_sums.append(s)
                    
                    if not conflict:
                        for v in vals:
                            s = 2 * x + v
                            if s in existing_sums or s in new_sums:
                                conflict = True
                                break
                            new_sums.append(s)
                    
                    if not conflict:
                        for i in range(k):
                            v1 = vals[i]
                            for j in range(i, k):
                                v2 = vals[j]
                                s = x + v1 + v2
                                if s in existing_sums or s in new_sums:
                                    conflict = True
                                    break
                                new_sums.append(s)
                            if conflict:
                                break
                    
                    if not conflict:
                        valid_candidates.append((x, new_sums))
                
                if not valid_candidates:
                    # No valid candidates in the pool, move to next part of list
                    # This might be inefficient if we skip too much, but prevents getting stuck on a bad local choice
                    # Actually, if no candidate in the pool is valid, we should probably try the next one in the sorted order?
                    # Or just break? If we are in a "stuck" state, we might need to backtrack.
                    # For simplicity, if no valid in pool, we shift the pool window?
                    # Let's just pick the first valid one in the entire remaining list if pool fails? 
                    # That defeats the purpose of randomization.
                    # Better: If no valid in pool, try the next element in 'available' directly?
                    # No, let's just break if we can't find any in the immediate vicinity? 
                    # Or, let's just pick the smallest valid one from the ENTIRE list? That reverts to greedy.
                    # Let's try: if no valid in pool, extend pool to next 50?
                    # To keep it simple: if no valid in pool, we just take the next one in 'available' and check it?
                    # Actually, if we are randomizing, we want to avoid the "greedy trap".
                    # If no candidate in the first 50 is valid, it's likely we are stuck or need to look further.
                    # Let's just break and rely on restarts?
                    # Or, let's pick a random candidate from the entire remaining list?
                    # That's O(N) check.
                    
                    # Let's try a different strategy: 
                    # If randomize is true, we pick a random index from 0 to min(window, len(available))
                    # and try that one. If it fails, we try the next random one?
                    # No, let's just pick the first valid one in the pool.
                    # If pool is empty, we break.
                    break
                else:
                    # Pick one at random from valid candidates
                    x, new_sums = rng.choice(valid_candidates)
                    # Remove x from available (it's at index < pool_size, so we need to find and remove)
                    # Since candidates is a slice, we need to remove x from available
                    available.remove(x)
                    vals.append(x)
                    existing_sums.update(new_sums)
            else:
                # Standard greedy: pick first valid
                x = available[0]
                available.pop(0)
                
                conflict = False
                new_sums = []
                
                s = 3 * x
                if s in existing_sums:
                    conflict = True
                else:
                    new_sums.append(s)
                
                if not conflict:
                    for v in vals:
                        s = 2 * x + v
                        if s in existing_sums or s in new_sums:
                            conflict = True
                            break
                        new_sums.append(s)
                
                if not conflict:
                    for i in range(k):
                        v1 = vals[i]
                        for j in range(i, k):
                            v2 = vals[j]
                            s = x + v1 + v2
                            if s in existing_sums or s in new_sums:
                                conflict = True
                                break
                            new_sums.append(s)
                        if conflict:
                            break
                
                if not conflict:
                    vals.append(x)
                    existing_sums.update(new_sums)
                else:
                    # If the first one fails, we should check the next ones?
                    # In standard greedy, we usually iterate through the sorted list and pick the FIRST valid one.
                    # The current code does: "for x in order: ... if not conflict: add".
                    # This implies it skips invalid ones.
                    # My loop above only checks available[0]. I need to iterate.
                    pass
        
        return vals

    # The previous greedy_build was simpler: iterate through order, add if valid.
    # Let's rewrite it to be cleaner and support the "randomized pool" idea properly.
    
    def greedy_build_v2(order, deadline, rng=None, randomize=False, window=100):
        vals = []
        existing_sums = set()
        # We need to efficiently check validity.
        # Order is a list. We process it sequentially.
        # If randomize, we don't just take the first valid.
        
        # To support randomization effectively, we should maintain a list of "candidates" 
        # but since validity depends on the current set, we have to check on the fly.
        
        # Strategy: 
        # 1. Iterate through 'order'.
        # 2. Maintain a list of "pending" valid candidates found in the recent window.
        # 3. If we have pending candidates, pick one randomly (or with bias) and add it.
        # 4. Clear pending? No, because adding one might invalidate others? 
        #    Actually, if x1 and x2 are both valid for S, are they both valid for S+{x1}?
        #    Not necessarily. x1 and x2 might conflict with each other.
        #    So we must check sequentially.
        
        # Simpler Randomized Greedy:
        # At each step, look at the next 'window' elements in 'order'.
        # Find all that are valid.
        # Pick one at random.
        # Remove it from 'order' (or mark as used).
        # If none valid in window, break? Or advance window?
        # If we advance window, we might skip valid elements that are further down.
        # But if no element in the first 100 is valid, it's likely we are stuck or the set is dense.
        
        i = 0
        n_order = len(order)
        while i < n_order:
            if time.time() > deadline:
                break
            
            k = len(vals)
            valid_candidates = []
            
            # Check a window of candidates
            end_idx = min(i + window, n_order)
            for j in range(i, end_idx):
                x = order[j]
                # Check if x is valid
                conflict = False
                new_sums = []
                
                s = 3 * x
                if s in existing_sums:
                    conflict = True
                else:
                    new_sums.append(s)
                
                if not conflict:
                    for v in vals:
                        s = 2 * x + v
                        if s in existing_sums or s in new_sums:
                            conflict = True
                            break
                        new_sums.append(s)
                
                if not conflict:
                    for a in range(k):
                        v1 = vals[a]
                        for b in range(a, k):
                            v2 = vals[b]
                            s = x + v1 + v2
                            if s in existing_sums or s in new_sums:
                                conflict = True
                                break
                            new_sums.append(s)
                        if conflict:
                            break
                
                if not conflict:
                    valid_candidates.append((j, x, new_sums))
            
            if not valid_candidates:
                # No valid candidates in this window.
                # Move to the next window?
                # If we just increment i by 1, we re-check many.
                # Let's jump to end_idx.
                i = end_idx
                continue
            
            if randomize and rng:
                # Pick a random valid candidate
                idx, x, new_sums = rng.choice(valid_candidates)
            else:
                # Pick the first valid candidate (smallest index in order)
                idx, x, new_sums = valid_candidates[0]
            
            vals.append(x)
            existing_sums.update(new_sums)
            
            # Remove the chosen candidate from consideration?
            # We are iterating through 'order'. We can't easily remove an element from a list while iterating.
            # But we can mark it as used?
            # Since 'order' is a permutation or sorted list, and we only add each x once,
            # we can just skip it in future iterations?
            # Actually, the loop `for j in range(i, end_idx)` will re-visit indices if we don't advance i.
            # We should advance i to idx + 1?
            # No, because other candidates in the window might still be valid.
            # But we already checked them.
            # If we pick x at idx, we should not pick it again.
            # We can set order[idx] = None?
            # And when checking, skip None.
            
            order[idx] = None
            i += 1 # Move forward by 1 to re-evaluate the window?
            # Or just i = end_idx?
            # If we set i = end_idx, we miss valid candidates between i and end_idx that were not picked.
            # But we already identified them as valid.
            # However, their validity might change after adding x?
            # Yes, adding x adds new sums, which might conflict with y.
            # So we MUST re-check the remaining candidates in the window.
            # So we should NOT jump to end_idx.
            # We should just increment i by 1?
            # But we already checked j=idx. We can skip it next time because it's None.
            # So, i stays the same?
            # If i stays the same, we re-check the window [i, i+window).
            # This is O(window * check_cost) per step.
            # Check cost is O(k^2).
            # Total O(N * window * k^2).
            # With N=4096, window=100, k~40. 4096 * 100 * 1600 = 655,360,000.
            # This is too slow for Python in 60s.
            
            # Optimization:
            # Only re-check the candidates that were NOT picked?
            # And only check them if they are still "potentially" valid?
            # Hard to know.
            
            # Alternative:
            # Just use the standard greedy (pick first valid) but with RANDOMIZED ORDER.
            # This is much faster.
            # The "randomized pool" idea is good for escaping local optima, but the re-checking cost is high.
            
            # Let's stick to Randomized Order Greedy (Shuffle) + Local Search.
            break # Exit the complex loop, use the simpler one below

    # Let's implement the simpler, faster approach:
    # 1. Greedy with Randomized Order (Shuffle)
    # 2. Local Search: 1-Remove-1-Add to escape local minima.

    def greedy_simple(order, deadline):
        vals = []
        existing_sums = set()
        for x in order:
            if time.time() > deadline:
                break
            k = len(vals)
            conflict = False
            new_sums = []
            
            s = 3 * x
            if s in existing_sums:
                conflict = True
            else:
                new_sums.append(s)
            
            if not conflict:
                for v in vals:
                    s = 2 * x + v
                    if s in existing_sums or s in new_sums:
                        conflict = True
                        break
                    new_sums.append(s)
            
            if not conflict:
                for i in range(k):
                    v1 = vals[i]
                    for j in range(i, k):
                        v2 = vals[j]
                        s = x + v1 + v2
                        if s in existing_sums or s in new_sums:
                            conflict = True
                            break
                        new_sums.append(s)
                    if conflict:
                        break
            
            if not conflict:
                vals.append(x)
                existing_sums.update(new_sums)
        
        return vals

    def local_search(vals, deadline, rng):
        # Try to improve vals by swapping elements
        # If we are stuck, we can try to remove an element and add a smaller one?
        # Or remove an element and add a larger one?
        # The goal is to maximize cardinality.
        # If we can't add any element, we are stuck.
        # Swapping doesn't increase cardinality.
        # We need a move that increases cardinality.
        # 1-Remove-2-Add?
        
        # Let's try a simple "Restart with Hint"
        # If we are stuck at size K, we know which elements are blocking.
        # We can try to build a new set, but force the inclusion of a subset of the old set?
        
        # Actually, a very effective heuristic is:
        # 1. Get a valid set S of size K.
        # 2. Generate a new random order.
        # 3. Greedily build, but SKIP any element that is in S? No, that's bad.
        # 4. Greedily build, but PREFER elements that are in S?
        
        # Let's just do many random restarts with different seeds.
        # The current program does this.
        # To improve, we can use a "Hybrid" strategy:
        # 1. Build a set S.
        # 2. Take the first M elements of S.
        # 3. Randomly shuffle the rest of the range.
        # 4. Greedily build starting from the first M elements?
        #    No, the first M elements might not be valid together? They are, because they came from S.
        #    So we start with S[:M], and then greedily add the rest.
        #    This biases the search towards "similar" sets but allows variation.
        
        pass

    best = []
    
    # 1. Primary: Greedy from 1 upward
    order1 = list(range(1, N + 1))
    result = greedy_simple(order1, DEADLINE)
    if len(result) > len(best):
        best = result
    
    if time.time() > DEADLINE:
        return best

    # 2. Randomized Restarts
    rng = random.Random(42)
    for attempt in range(200):
        if time.time() > DEADLINE:
            break
        
        # Strategy: Random Shuffle
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        result = greedy_simple(order, DEADLINE)
        if len(result) > len(best):
            best = result
            if len(best) >= 40:
                break
    
    # 3. Local Search / Polish on the best set found
    # Try to extend 'best' by trying to add elements in a random order, 
    # but allowing "swaps" if it leads to a larger set?
    # Since greedy_simple is fast, we can do many more restarts.
    
    # Let's add a "Biased" restart:
    # Take the current best set, pick a random subset of it, and use that as a "seed" for the next greedy run?
    # No, greedy_simple starts from scratch.
    
    # We can modify greedy_simple to accept a "pre-filled" set.
    def greedy_with_seed(seed_vals, order, deadline):
        vals = list(seed_vals)
        # Reconstruct sums
        existing_sums = set()
        k = len(vals)
        for i in range(k):
            v1 = vals[i]
            for j in range(i, k):
                v2 = vals[j]
                existing_sums.add(v1 + v2 + v1) # 2*v1+v2? No, triple sums.
                # Triple sums: a+b+c with a<=b<=c
                # If we have vals, we need all a+b+c.
                # This is O(k^3) to reconstruct.
                # For k=40, 40^3 = 64000. Fast.
        
        # Let's just reconstruct sums properly
        existing_sums = set()
        for i in range(k):
            for j in range(i, k):
                for l in range(j, k):
                    existing_sums.add(vals[i] + vals[j] + vals[l])
        
        # Now greedily add from order
        for x in order:
            if time.time() > deadline:
                break
            if x in vals:
                continue
            k_curr = len(vals)
            conflict = False
            new_sums = []
            
            s = 3 * x
            if s in existing_sums:
                conflict = True
            else:
                new_sums.append(s)
            
            if not conflict:
                for v in vals:
                    s = 2 * x + v
                    if s in existing_sums or s in new_sums:
                        conflict = True
                        break
                    new_sums.append(s)
            
            if not conflict:
                for i in range(k_curr):
                    v1 = vals[i]
                    for j in range(i, k_curr):
                        v2 = vals[j]
                        s = x + v1 + v2
                        if s in existing_sums or s in new_sums:
                            conflict = True
                            break
                        new_sums.append(s)
                    if conflict:
                        break
            
            if not conflict:
                vals.append(x)
                existing_sums.update(new_sums)
        
        return vals

    # Use biased restarts: seed with a random subset of 'best'
    if len(best) > 0:
        for attempt in range(50):
            if time.time() > DEADLINE:
                break
            
            # Pick a random prefix of 'best' to use as seed
            # This ensures the new set is "close" to the best one
            seed_size = rng.randint(0, len(best))
            seed = best[:seed_size]
            
            order = list(range(1, N + 1))
            rng.shuffle(order)
            
            result = greedy_with_seed(seed, order, DEADLINE)
            if len(result) > len(best):
                best = result
                if len(best) >= 40:
                    break

    return best
