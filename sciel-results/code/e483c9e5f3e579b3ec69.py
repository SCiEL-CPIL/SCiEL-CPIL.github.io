"""Improved program using Local Search (Hill Climbing) for sum-free subsets."""

import random
import time

def construct():
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def max_sum_free_local_search(vals, time_limit=10.0):
        n = len(vals)
        if n == 0:
            return []
        
        # Precompute indices for faster lookup? 
        # Since n is small (max 240), O(n^2) checks are fine.
        # We represent the set as a boolean array for speed.
        
        # Initial solution: Random subset of size ~ n/2
        # Or a simple greedy to get a decent start.
        # Let's start with a random permutation and greedy add.
        rng = random.Random(42) # Fixed seed for reproducibility of the search path itself? 
                                # Actually, the input is fixed, so the search should be deterministic or use a fixed seed for the search logic.
        
        # Let's use a deterministic local search with random restarts if stuck, 
        # but we have a time limit.
        
        best_set = set()
        best_size = 0
        
        start_time = time.time()
        
        # Try multiple restarts
        for restart in range(50):
            # 1. Generate an initial candidate
            # Strategy: Randomly shuffle vals, then greedily add if possible.
            cand = vals[:]
            rng.shuffle(cand)
            current_set = set()
            current_sums = set()
            current_diffs = set()
            
            for x in cand:
                if x in current_sums or x in current_diffs or (2*x in current_set):
                    continue
                # Check if 2*x is in current_set is handled by diffs?
                # If 2*x in current_set, then 2*x = x + x. 
                # When we added x, we added 2*x to sums. 
                # Wait, the check `x in current_sums` prevents adding x if x = a+b.
                # The check `2*x in current_set` prevents adding x if 2x is already there.
                # If 2x is in current_set, then 2x = y+z for some y,z in set? Not necessarily.
                # But if 2x is in set, and we add x, then x+x=2x is a violation.
                # So we must check if 2x is in current_set.
                
                # However, the previous greedy code had:
                # if x in sums: continue
                # if x in diffs: continue
                # if 2*x in S: continue
                
                # Let's stick to that logic for initialization.
                if x in current_sums:
                    continue
                if x in current_diffs:
                    continue
                if 2 * x in current_set:
                    continue
                
                current_set.add(x)
                current_sums.add(2 * x)
                for y in current_set:
                    if y == x:
                        continue
                    current_sums.add(x + y)
                    current_diffs.add(x - y)
                    current_diffs.add(y - x)
            
            # Now perform local search: Remove conflicting elements, then add back.
            # We iterate until no improvement or time limit.
            
            # To make it faster, we can just do a few passes of "remove worst, add best".
            # A simpler effective heuristic:
            # 1. Find all triples (a,b,c) in current_set such that a+b=c.
            # 2. If none, we have a sum-free set.
            # 3. If yes, remove one element from a random violating triple.
            # 4. Try to add back any element from the original `vals` not in current_set 
            #    that doesn't create a new violation.
            
            improved = True
            while improved and (time.time() - start_time) < time_limit:
                improved = False
                
                # Find violations
                # For each pair (a, b) in current_set, check if a+b is in current_set.
                # To speed up, we can just check all pairs.
                # N is at most 240. Pairs ~ 28k. Check set membership is O(1).
                # This is fast enough.
                
                violations = []
                items = list(current_set)
                for i in range(len(items)):
                    a = items[i]
                    for j in range(i, len(items)): # a can be b
                        b = items[j]
                        c = a + b
                        if c in current_set:
                            violations.append((a, b, c))
                
                if not violations:
                    # No violations. Try to add more elements.
                    # Sort remaining elements by some heuristic? 
                    # Just iterate through vals not in current_set.
                    added_any = False
                    # To avoid O(n^2) per check, we can just try adding them one by one.
                    # But adding one might block others. 
                    # Greedy addition:
                    # Check each candidate x in vals \ current_set.
                    # If x can be added (no x in sums, no x in diffs, 2x not in set), add it.
                    
                    # We need to maintain sums and diffs for the current_set to do this fast.
                    # Rebuilding sums/diffs for current_set (size ~100) is O(k^2) ~ 10k ops.
                    # Doing this for each of ~100 candidates is 1M ops. Acceptable.
                    
                    # Let's rebuild sums and diffs for current_set
                    curr_sums = set()
                    curr_diffs = set()
                    for a in current_set:
                        curr_sums.add(2*a)
                        for b in current_set:
                            if a != b:
                                curr_sums.add(a+b)
                                curr_diffs.add(a-b)
                                curr_diffs.add(b-a)
                    
                    candidates = [x for x in vals if x not in current_set]
                    # Sort candidates by some metric? 
                    # Elements with fewer conflicts are better? 
                    # For now, just iterate.
                    for x in candidates:
                        if x in curr_sums or x in curr_diffs or (2*x in current_set):
                            continue
                        
                        # Check if adding x creates a violation with existing elements?
                        # The checks above ensure:
                        # 1. x is not a sum of two existing elements.
                        # 2. x is not a difference (which implies a-b=x or b-a=x, i.e., a=x+b or b=x+a).
                        #    If a = x+b, then x+a = 2x+b? No.
                        #    If x = a-b, then a = x+b. So x+b = a (in set). Violation x+b=a.
                        #    So checking x in diffs prevents x+b=a and b+x=a.
                        # 3. 2x not in set prevents x+x=2x.
                        
                        # So if these pass, x is safe to add.
                        current_set.add(x)
                        # Update sums and diffs
                        curr_sums.add(2*x)
                        for y in current_set:
                            if y == x:
                                continue
                            curr_sums.add(x+y)
                            curr_diffs.add(x-y)
                            curr_diffs.add(y-x)
                        added_any = True
                        improved = True # We made progress, restart the loop to find new violations or more additions?
                        # Actually, if we add, we might create new violations? No, we checked safety.
                        # But adding x might make *other* candidates unsafe? Yes.
                        # But we are iterating through candidates. We should probably break and restart the outer loop
                        # or just continue checking other candidates against the *updated* sums/diffs.
                        # The code above updates curr_sums/diffs, so subsequent checks are valid.
                    
                    if not added_any:
                        # Local optimum reached
                        break

                else:
                    # We have violations.
                    # Remove one element from a random violation.
                    # Heuristic: Remove the element that is involved in the most violations?
                    # For simplicity, remove a random one from the first violation found.
                    # Better: Count frequency of elements in violations.
                    freq = {}
                    for a, b, c in violations:
                        freq[a] = freq.get(a, 0) + 1
                        freq[b] = freq.get(b, 0) + 1
                        freq[c] = freq.get(c, 0) + 1
                    
                    # Pick element with max frequency
                    worst_elem = max(freq.keys(), key=lambda k: freq[k])
                    current_set.remove(worst_elem)
                    improved = True
            
            # End of restart loop iteration
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            
            # Check time
            if time.time() - start_time > time_limit:
                break

        return list(best_set)

    results = []
    # Total time budget is 120s. 
    # Instances: 60, 120, 240.
    # Allocate time based on size? Or equal?
    # 240 is hardest. 
    # Let's give 15s to 60, 25s to 120, 70s to 240. Total 110s. Safe.
    time_alloc = {60: 15.0, 120: 25.0, 240: 70.0}
    
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = instance(size, seed)
        t_limit = time_alloc.get(size, 10.0)
        subset = max_sum_free_local_search(vals, t_limit)
        results.append(subset)
        
    return results
