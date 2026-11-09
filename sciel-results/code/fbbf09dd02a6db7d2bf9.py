import time
import random
import numpy as np

def construct():
    t_start = time.time()
    DEADLINE = t_start + 85.0
    N = 4096
    MAX_SUM = 3 * N
    
    # sums[i] = 1 if sum i is used
    # We will use a list of bools or a bytearray for speed
    # bytearray is faster to index than a list of bools in some contexts, 
    # but numpy is likely fastest for bit operations if we used bits. 
    # Given MAX_SUM ~ 12k, a simple list of bools or set is fine. 
    # Set is O(1) average. List is O(1) but memory heavy if not careful. 
    # Let's use a set for simplicity and speed of 'in' check.
    
    def try_add(x, selected, sums):
        """
        Checks if x can be added to selected.
        Returns (ok, new_sums_to_add)
        """
        n = len(selected)
        new_sums = []
        
        # Triples with one x: x + a_i + a_j
        # We need to check all i <= j
        # If x is added, it becomes part of the set.
        # The new sums involving x are:
        # 1. x + a_i + a_j for 0 <= i <= j < n
        # 2. 2*x + a_i for 0 <= i < n
        # 3. 3*x
        
        # Optimization: 
        # If x + min(a) + min(a) > max_sum_in_sums? No, sums can be anything.
        # Just check.
        
        # Generate candidate sums
        # To avoid duplicates in new_sums list (which would cause false positives if we check 'in sums' 
        # and then add to sums, but we check against 'sums' and 'seen_new'), we can use a local set.
        
        seen_new = set()
        
        for i in range(n):
            ai = selected[i]
            # x + ai + ai
            s = x + 2*ai
            if s in sums or s in seen_new:
                return False, None
            seen_new.add(s)
            
            for j in range(i+1, n):
                aj = selected[j]
                s = x + ai + aj
                if s in sums or s in seen_new:
                    return False, None
                seen_new.add(s)
        
        for i in range(n):
            ai = selected[i]
            s = 2*x + ai
            if s in sums or s in seen_new:
                return False, None
            seen_new.add(s)
        
        s = 3*x
        if s in sums or s in seen_new:
            return False, None
        seen_new.add(s)
        
        return True, list(seen_new)

    def greedy_fill(selected, sums, start=1, end=N, skip_set=None):
        """
        Greedily adds numbers from start to end to selected.
        """
        if skip_set is None:
            skip_set = set()
            
        for x in range(start, end + 1):
            if time.time() > DEADLINE - 5:
                break
            if x in skip_set:
                continue
            ok, new_sums = try_add(x, selected, sums)
            if ok:
                selected.append(x)
                sums.update(new_sums)
                
    def local_search_improve(selected, sums):
        """
        Tries to increase size by removing 1 and adding 2.
        Returns (improved, new_selected, new_sums)
        """
        improved = False
        # Sort selected to ensure consistent ordering for removal? 
        # The set doesn't need to be sorted for validity, but for deterministic search.
        
        # We iterate through indices to remove
        n = len(selected)
        # We should try removing elements that are "expensive" or "blocking".
        # Random order might be better to avoid local minima.
        indices = list(range(n))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > DEADLINE - 10:
                break
                
            # Tentatively remove selected[idx]
            removed = selected[idx]
            temp_selected = [x for i, x in enumerate(selected) if i != idx]
            
            # Rebuild sums for temp_selected
            # This is O(n^3) which is ~ 8000 ops. Acceptable.
            temp_sums = set()
            m = len(temp_selected)
            for i in range(m):
                ai = temp_selected[i]
                for j in range(i, m):
                    aj = temp_selected[j]
                    for k in range(j, m):
                        ak = temp_selected[k]
                        temp_sums.add(ai + aj + ak)
            
            # Try to add 2 new numbers
            added = 0
            new_selected = list(temp_selected)
            new_sums = set(temp_sums)
            
            # Scan candidates. 
            # To maximize chance, scan from 1 to N? Or random?
            # Scanning from 1 is deterministic. 
            # Let's scan from 1.
            
            # Optimization: We only need to add 2.
            # We can break early if we find 2.
            
            # To avoid adding the same number twice, we track added numbers.
            # Actually, try_add checks against new_sums. 
            # If we add x1, it updates new_sums. Then we try x2.
            
            for x in range(1, N + 1):
                if time.time() > DEADLINE - 10:
                    break
                if x in new_selected:
                    continue
                # Also skip the removed element? 
                # If we re-add the removed element, we are back to start (size n). 
                # We want size n+1. So we must add 2 DIFFERENT elements not in temp_selected.
                # If x == removed, we are just swapping? No, we removed 1, adding 1 back is size n.
                # We need to add 2 NEW ones.
                if x == removed:
                    continue
                    
                ok, new_sums_to_add = try_add(x, new_selected, new_sums)
                if ok:
                    new_selected.append(x)
                    new_sums.update(new_sums_to_add)
                    added += 1
                    if added == 2:
                        break
            
            if added == 2:
                # Success! Size increased by 1.
                return True, new_selected, new_sums
                
        return False, selected, sums

    # Main Logic
    best_size = 0
    best_selected = []
    best_sums = set()
    
    # 1. Standard Greedy
    selected = []
    sums = set()
    greedy_fill(selected, sums)
    
    if len(selected) > best_size:
        best_size = len(selected)
        best_selected = list(selected)
        best_sums = set(sums)
        
    # 2. Local Search on the result
    improved, sel, sums = local_search_improve(selected, sums)
    if improved and len(sel) > best_size:
        best_size = len(sel)
        best_selected = list(sel)
        best_sums = set(sums)
        
    # 3. Try more local search rounds on the improved set
    for _ in range(5):
        if time.time() > DEADLINE - 20:
            break
        improved, sel, sums = local_search_improve(best_selected, best_sums)
        if improved and len(sel) > best_size:
            best_size = len(sel)
            best_selected = list(sel)
            best_sums = set(sums)
        else:
            break
            
    # 4. Randomized Restarts
    # Try starting with a few random numbers and filling greedily?
    # Or skipping random numbers in the greedy pass?
    
    rng = random.Random(42)
    
    for restart in range(50):
        if time.time() > DEADLINE - 10:
            break
            
        # Strategy: Randomly skip some small numbers to see if it allows better packing later
        # Or: Start with a random valid seed of size 3-5 and greedy fill.
        
        # Let's try: Random seed of size 3.
        # Pick 3 random numbers from 1..N.
        # Check if they form a valid B3 set (trivially yes if distinct? No, 3 distinct numbers always form 1 triple sum. 
        # Wait, if a,b,c are distinct, sums are a+b+c. Just 1 sum. Always valid.
        # If we pick a, a, b? Sums: 3a, 2a+b, a+2b, a+b+c? No, set elements are distinct.
        # So any 3 distinct numbers are a valid B3 set of size 3.
        
        seed_size = rng.randint(3, 5)
        seed = rng.sample(range(1, N+1), seed_size)
        seed.sort()
        
        # Verify seed is valid (it should be, but just in case of logic errors)
        # For size < 3, always valid. For size 3, 1 sum. For size 4, 4 sums.
        # Let's just build sums.
        sel = list(seed)
        sums = set()
        m = len(sel)
        for i in range(m):
            for j in range(i, m):
                for k in range(j, m):
                    sums.add(sel[i] + sel[j] + sel[k])
                    
        # Greedy fill
        greedy_fill(sel, sums, start=1, end=N)
        
        if len(sel) > best_size:
            best_size = len(sel)
            best_selected = list(sel)
            best_sums = set(sums)
            
            # Try local search on this new best
            if time.time() < DEADLINE - 15:
                improved, sel, sums = local_search_improve(sel, sums)
                if improved and len(sel) > best_size:
                    best_size = len(sel)
                    best_selected = list(sel)
                    best_sums = set(sums)

    return best_selected
