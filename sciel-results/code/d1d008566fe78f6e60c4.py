"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 256

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s kill
    
    # Precompute range
    range_1_N = range(1, N + 1)
    
    # Precompute all possible sums for Sidon check
    # A set S is Sidon if all a+b for a<=b in S are distinct.
    # Max sum is 256+256 = 512.
    MAX_SUM = 2 * N
    
    def is_valid_sidon(selected):
        """Check if selected is a Sidon set."""
        sums = set()
        for i, a in enumerate(selected):
            for b in selected[i:]:
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        return True

    def get_sums(selected):
        sums = set()
        for i, a in enumerate(selected):
            for b in selected[i:]:
                sums.add(a + b)
        return sums

    def can_add(x, selected, sums):
        """Check if x can be added to selected without violating Sidon property."""
        if 2 * x in sums:
            return False
        for a in selected:
            if x + a in sums:
                return False
        return True

    def is_maximal(selected, sums):
        """Check if no element from [1, N] can be added."""
        selected_set = set(selected)
        for x in range_1_N:
            if x in selected_set:
                continue
            if can_add(x, selected, sums):
                return False
        return True

    def random_greedy(seed):
        """Construct a maximal Sidon set using randomized greedy."""
        random.seed(seed)
        candidates = list(range_1_N)
        random.shuffle(candidates)
        
        selected = []
        sums = set()
        
        for x in candidates:
            if can_add(x, selected, sums):
                selected.append(x)
                sums.add(2 * x)
                for a in selected[:-1]:
                    sums.add(x + a)
        return selected

    def local_search_minimize(selected, sums):
        """Try to remove elements while maintaining maximality."""
        selected = list(selected)
        sums = get_sums(selected)
        
        changed = True
        while changed:
            changed = False
            for i in range(len(selected)):
                x = selected[i]
                new_selected = selected[:i] + selected[i+1:]
                if not new_selected:
                    continue
                
                new_sums = get_sums(new_selected)
                
                if is_maximal(new_selected, new_sums):
                    selected = new_selected
                    sums = new_sums
                    changed = True
                    break
        return selected

    # --- NEW STRATEGY: Direct Search for Small Maximal Sets ---
    
    def dfs_direct_search(max_size, current, sums, used_elements, seed):
        """
        Depth-first search to find a Sidon set of size <= max_size that is maximal.
        Returns (found, set) or (False, None).
        """
        if time.time() > deadline:
            return False, None
            
        # Pruning: if current size + remaining possible < max_size, we can't fill up?
        # Actually, we want to find ANY maximal set of size <= max_size.
        # A maximal set must block all other elements.
        
        if len(current) > max_size:
            return False, None
            
        # Check if current set is already maximal
        if is_maximal(current, sums):
            return True, list(current)
            
        # If we have hit the max_size limit and are not maximal, fail
        if len(current) == max_size:
            return False, None
            
        # Heuristic: Order candidates by "blocking power"
        # An element x blocks y if y cannot be added due to x.
        # We want to pick elements that block many others.
        
        # To speed up, we can precompute which elements are currently "free" (can be added).
        # If there are free elements, we MUST pick one of them or an element that blocks them.
        # Actually, for a set to be maximal, every free element must be blocked by the set.
        # If we haven't picked enough elements, we might leave some free.
        
        # Strategy:
        # 1. Identify all x in [1, N] not in current that CAN be added (free candidates).
        # 2. If there are no free candidates, the set is maximal (handled above).
        # 3. We must pick the next element from the set of elements that will help block free candidates?
        #    No, we just need to pick any valid element such that eventually all free ones are blocked.
        #    However, if we pick an element that is NOT a free candidate, it might not help block the current free candidates?
        #    Actually, adding an element z to S adds sums {2z, z+a}. This might block some free candidates.
        
        # Let's identify the "critical" free candidates.
        free_candidates = []
        current_set = set(current)
        for x in range_1_N:
            if x in current_set:
                continue
            if can_add(x, current, sums):
                free_candidates.append(x)
                
        if not free_candidates:
             # This case is handled by is_maximal check above, but just in case
             return True, list(current)

        # We need to select the next element.
        # If we don't select an element that blocks a specific free candidate, that candidate might remain free.
        # But we have max_size - len(current) moves left.
        # Each move can block multiple candidates.
        
        # Heuristic: Sort potential next elements by how many free candidates they block.
        # Potential next elements are those that can be added to current.
        potential_next = []
        for x in range_1_N:
            if x in current_set:
                continue
            if can_add(x, current, sums):
                # How many free candidates does x block?
                # x blocks y if (y+x in sums_new) or (2y in sums_new) ... 
                # Actually, if we add x, y is blocked if y+x is in sums or 2y is in sums.
                # sums_new = sums U {2x} U {x+a for a in current}
                # y is blocked if:
                # 1. 2y is in sums_new
                # 2. y+x is in sums_new
                # 3. y+a is in sums_new for some a in current (this is already true if y was free? No, y was free so y+a not in sums)
                # So y is blocked by x if:
                # - 2y == 2x (y=x, impossible)
                # - 2y == x+a (for some a in current)
                # - y+x == 2x (y=x, impossible)
                # - y+x == x+a (y=a, impossible)
                # - y+x == 2x' (no, sums contains sums of pairs)
                # Wait, the condition for y to be blocked is:
                # can_add(y, current + [x], sums_new) == False
                
                # Let's just count how many free candidates are blocked by adding x.
                blocked_count = 0
                # Optimization: just check the intersection of free_candidates with elements blocked by x
                # For simplicity in this DFS, we might just iterate.
                for y in free_candidates:
                    # Check if y is blocked by adding x
                    # y is blocked if 2y in sums_new or y+z in sums_new for z in current+[x]
                    # sums_new contains:
                    # - old sums
                    # - 2x
                    # - x + a for a in current
                    
                    # y was free, so 2y not in sums and y+a not in sums for a in current.
                    # So y is blocked by x if:
                    # 1. 2y == 2x -> y=x (no)
                    # 2. 2y == x+a -> y = (x+a)/2
                    # 3. y+x == 2x -> y=x (no)
                    # 4. y+x == x+a -> y=a (no)
                    # 5. y+x is in sums? (old sums) -> if y+x in sums, then y would have been blocked already?
                    #    can_add(y, current, sums) checks if y+a in sums for a in current.
                    #    It does NOT check if y+x is in sums because x is not in current.
                    #    So if y+x is in sums (from pairs in current), then y is blocked by x?
                    #    No, the Sidon condition for set S U {x} requires all sums a+b distinct.
                    #    If we add x, we add sums {2x, x+a}.
                    #    y is blocked if adding y creates a duplicate sum.
                    #    Duplicate if:
                    #    - 2y in sums_new
                    #    - y+z in sums_new for z in S U {x}
                    
                    # Since y was free w.r.t S:
                    # - 2y not in sums
                    # - y+z not in sums for z in S
                    
                    # So y is blocked by x if:
                    # - 2y in {2x} U {x+a for a in S}
                    # - y+z in {2x} U {x+a for a in S} U {y+x} (y+x is new sum, but we check against existing sums in S U {x})
                    #   Wait, the check is: does y+x conflict with existing sums in S U {x}?
                    #   Existing sums in S U {x} are sums(S) U {2x} U {x+a}.
                    #   So we check if y+x is in sums(S) U {2x} U {x+a}.
                    
                    # Case 1: 2y == 2x => y=x (No)
                    # Case 2: 2y == x+a => y = (x+a)/2
                    # Case 3: y+x == 2x => y=x (No)
                    # Case 4: y+x == x+a => y=a (No)
                    # Case 5: y+x in sums(S)
                    
                    # So y is blocked if:
                    # (2y == x+a for some a in S) OR (y+x in sums)
                    
                    blocked = False
                    # Check y+x in sums
                    if y + x in sums:
                        blocked = True
                    else:
                        # Check 2y == x+a
                        # x+a = 2y => a = 2y - x
                        a_needed = 2 * y - x
                        if a_needed in current_set:
                            blocked = True
                            
                    if blocked:
                        blocked_count += 1
                        
                if blocked_count > 0:
                    potential_next.append((blocked_count, x))
                    
        if not potential_next:
            # No element can be added that blocks any free candidate?
            # This means we can't make it maximal with max_size?
            # Or we are stuck.
            return False, None
            
        # Sort by blocking power (desc), then by value (for determinism/variance)
        potential_next.sort(key=lambda t: (-t[0], t[1]))
        
        # To avoid symmetry and explore different branches, we can shuffle top candidates
        # But for now, let's try the best one first, then others if it fails.
        
        # Randomization: pick from top K candidates
        K = 5
        candidates_to_try = [x for _, x in potential_next[:K]]
        random.shuffle(candidates_to_try)
        
        for x in candidates_to_try:
            new_current = current + [x]
            new_sums = set(sums)
            new_sums.add(2 * x)
            for a in current:
                new_sums.add(x + a)
                
            found, result = dfs_direct_search(max_size, new_current, new_sums, used_elements | {x}, seed)
            if found:
                return True, result
                
        return False, None

    best_result = None
    best_size = float('inf')
    
    # 1. Get a baseline from random greedy + local search (fast)
    for trial in range(50):
        if time.time() > deadline:
            break
        selected = random_greedy(trial)
        sums = get_sums(selected)
        selected_min = local_search_minimize(selected, sums)
        sums_final = get_sums(selected_min)
        if is_maximal(selected_min, sums_final) and len(selected_min) < best_size:
            best_result = selected_min
            best_size = len(selected_min)
            
    # 2. Direct DFS for smaller sizes
    # Theoretical lower bound for maximal Sidon set size is ~ sqrt(N) ?
    # Actually, a random Sidon set of size k has k^2/2 sums.
    # To block N elements, we need sums to cover "blocking conditions".
    # Let's try sizes starting from max(5, best_size - 5) up to best_size - 1
    
    # If we found a decent baseline, try to beat it.
    start_size = max(5, best_size - 10)
    
    for target_size in range(start_size, best_size):
        if time.time() > deadline:
            break
            
        # Try multiple seeds for the randomization in DFS
        for seed in range(20):
            if time.time() > deadline:
                break
            
            found, result = dfs_direct_search(target_size, [], set(), set(), seed)
            if found:
                if len(result) < best_size:
                    best_result = result
                    best_size = len(result)
                break # Found one for this size, try smaller? No, we are iterating up.
                     # Actually, we want the MINIMUM. So if we find one of size T, we should check T-1?
                     # The loop goes start_size to best_size.
                     # If we find size 10, we update best_size to 10.
                     # Next iteration is 11? No, we want smaller.
                     # So we should iterate DOWN from best_size?
                     # Or iterate UP from 1?
                     # Iterating UP is safer because small sets are harder to find (more constraints).
                     # If we find size 10, we know 10 is possible. We should then check if 9 is possible.
                     # So if we iterate 5, 6, 7... and find 7, we update best. Then check 8? No, 8 is worse.
                     # So if we iterate UP, and find 7, we can stop? No, maybe 6 exists but we missed it?
                     # DFS is exact for the path explored. But we limited branching.
                     # So we should iterate DOWN from current best to 1?
                     # But small sizes are very hard.
                     # Let's iterate UP from a reasonable lower bound.
                     # If we find size K, we record it. We continue to K+1? No, that's larger.
                     # We want the MINIMAL.
                     # So if we iterate 5, 6, 7...
                     # If we find 7, best=7.
                     # Next is 8. We don't care about 8 if we have 7.
                     # So we should break if we find a solution?
                     # No, because we might find 6 in a later seed?
                     # The loop structure:
                     # for target_size in range(start_size, best_size):
                     #   if found: update best, break inner?
                     #   continue outer?

    # Refined DFS loop
    # We have best_size from greedy.
    # Try to find smaller.
    for target_size in range(best_size - 1, 0, -1):
        if time.time() > deadline:
            break
        
        # Try a few seeds
        found_any = False
        for seed in range(10):
            if time.time() > deadline:
                break
            found, result = dfs_direct_search(target_size, [], set(), set(), seed)
            if found:
                if len(result) < best_size:
                    best_result = result
                    best_size = len(result)
                found_any = True
                break
        
        if found_any:
            # If we found a set of size T, we should check T-1 next.
            # The loop naturally goes to T-1.
            continue
        else:
            # If we failed to find size T, it might be too hard or doesn't exist.
            # Continue to T-1?
            # Yes, maybe T-1 is easier? Unlikely, but possible.
            # However, if T is hard, T-1 is harder.
            # So we can stop if we fail on a small size?
            # Let's just keep going until deadline.
            pass

    if best_result is not None:
        return best_result
        
    # Fallback
    return random_greedy(0)
# EVOLVE-BLOCK-END
