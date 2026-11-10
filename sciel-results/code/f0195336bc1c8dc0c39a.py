import time
import random
from collections import defaultdict

def _greedy_build(candidates, counts_init=None, chosen_init=None):
    """Greedy: add candidates in given order if B2[2] constraint holds."""
    if chosen_init is None:
        chosen = []
        chosen_set = set()
    else:
        chosen = list(chosen_init)
        chosen_set = set(chosen)
        
    if counts_init is None:
        counts = defaultdict(int)
    else:
        counts = dict(counts_init)
        
    for x in candidates:
        if x in chosen_set:
            continue
            
        # Check if adding x is valid
        # New sums: x + y for y in chosen, and 2*x
        ok = True
        new_sums = []
        
        # Check sums with existing elements
        for y in chosen:
            s = x + y
            if counts[s] + 1 > 2:
                ok = False
                break
            new_sums.append(s)
            
        if not ok:
            continue
            
        # Check double sum
        s2 = 2 * x
        if counts[s2] + 1 > 2:
            continue
        new_sums.append(s2)
        
        # If valid, add x
        for s in new_sums:
            counts[s] += 1
        chosen.append(x)
        chosen_set.add(x)
        
    return chosen, counts, chosen_set

def _local_search(chosen, time_limit=50.0):
    """
    Local search to improve set size.
    1. Try to insert any valid element.
    2. Try to swap: remove one, add another.
    """
    chosen = sorted(chosen)
    n = len(chosen)
    if n == 0:
        return chosen
        
    # Initialize counts
    counts = defaultdict(int)
    for i in range(n):
        for j in range(i, n):
            counts[chosen[i] + chosen[j]] += 1
            
    chosen_set = set(chosen)
    start_time = time.time()
    improved = True
    max_iter = 10000 # Limit iterations to prevent infinite loops if stuck
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        iter_count = 0
        
        # Phase 1: Try Insertions
        # We try to add any number from 1 to 1000 not in chosen_set
        # To make this faster, we can shuffle the candidates
        candidates = [x for x in range(1, 1001) if x not in chosen_set]
        random.shuffle(candidates)
        
        for y in candidates:
            if time.time() - start_time > time_limit:
                break
            if iter_count > 5000: # Safety break for inner loop
                break
                
            # Check if y can be added
            ok = True
            # Check sums with all existing
            for x in chosen:
                s = y + x
                if counts[s] + 1 > 2:
                    ok = False
                    break
            
            if not ok:
                continue
                
            s2 = 2 * y
            if counts[s2] + 1 > 2:
                continue
                
            # Add y
            for x in chosen:
                counts[y + x] += 1
            counts[2 * y] += 1
            chosen.append(y)
            chosen_set.add(y)
            n += 1
            improved = True
            break # Restart search with new set
            
        if improved:
            continue
            
        # Phase 2: Try Swaps
        # If no insertion, try to swap one element out and one in
        # We iterate through elements to remove
        # To save time, we don't try all 1000 candidates for addition for every removal.
        # We try a subset.
        
        # Create a list of indices to try removing
        indices_to_try = list(range(n))
        random.shuffle(indices_to_try)
        
        for idx in indices_to_try:
            if time.time() - start_time > time_limit:
                break
                
            x = chosen[idx]
            
            # Temporarily remove x
            # Update counts: decrement counts for sums involving x
            # Sums: x + chosen[j] for j != idx, and 2*x
            # Note: chosen is a list, so we need to be careful with indices
            
            # Optimization: Instead of updating counts immediately and reverting,
            # we can just check validity of adding y to (chosen \ {x})
            # But updating counts is O(N). Checking validity is O(N).
            # So it's similar cost.
            
            # Let's do the swap in-place for speed
            
            # Decrement counts for x
            counts[2 * x] -= 1
            for j in range(n):
                if j != idx:
                    counts[x + chosen[j]] -= 1
            
            # Remove x from list and set
            # Swap with last element to pop
            chosen[idx] = chosen[-1]
            chosen.pop()
            chosen_set.discard(x)
            n -= 1
            
            # Try to add a new element y
            # We prefer y > x or y that fills gaps?
            # Let's try random candidates, biased towards values not yet tried
            # or values that are "close" to existing ones?
            
            # For efficiency, we try a limited number of random candidates
            # If we don't find one, we revert.
            
            swap_found = False
            # Try 20 random candidates
            for _ in range(20):
                y = random.randint(1, 1000)
                if y in chosen_set:
                    continue
                    
                ok = True
                for z in chosen:
                    s = y + z
                    if counts[s] + 1 > 2:
                        ok = False
                        break
                if not ok:
                    continue
                s2 = 2 * y
                if counts[s2] + 1 > 2:
                    continue
                
                # Add y
                for z in chosen:
                    counts[y + z] += 1
                counts[2 * y] += 1
                chosen.append(y)
                chosen_set.add(y)
                n += 1
                swap_found = True
                improved = True
                break
            
            if not swap_found:
                # Revert: Add x back
                chosen.append(x)
                chosen_set.add(x)
                n += 1
                counts[2 * x] += 1
                for z in chosen:
                    if z != x: # z is the newly added x? No, x is in chosen now.
                        # Wait, chosen contains x at the end now?
                        # We did chosen.append(x)
                        # So we need to add counts for x + other elements
                        pass
                
                # Correct Revert Logic:
                # We removed x. We are adding it back.
                # The counts currently reflect (chosen_without_x).
                # We need to add counts for x + (chosen_without_x) and 2*x.
                # Note: x is NOT in chosen yet in the counts update?
                # Yes, we did chosen.append(x) above.
                # So we iterate over chosen, but skip x itself?
                # Actually, when we removed x, we decremented counts for x + y (y in chosen_without_x).
                # Now we add x back. We need to increment counts for x + y (y in chosen_without_x).
                # And 2*x.
                
                # But wait, in the revert block:
                # chosen currently has x at the end.
                # The other elements are the same as when we removed x.
                # So we iterate over chosen[:-1] (the elements other than x)
                for z in chosen[:-1]:
                    counts[x + z] += 1
                # counts[2*x] was decremented, so increment it back
                # We already did counts[2*x] -= 1 before removal.
                # So we need counts[2*x] += 1 here.
                # BUT, we already incremented it in the "Add y" logic if we failed? No.
                # If swap_found is False, we didn't add y.
                # So counts are in state "x removed".
                # We add x back.
                counts[2 * x] += 1
                for z in chosen[:-1]:
                    counts[x + z] += 1
                
            # End of swap attempt for this idx

    return chosen

def _polish(chosen):
    """Try to add any remaining valid elements."""
    chosen, counts, chosen_set = _greedy_build([], counts_init=None, chosen_init=None) # Dummy
    # Re-init counts properly
    counts = defaultdict(int)
    for i in range(len(chosen)):
        for j in range(i, len(chosen)):
            counts[chosen[i] + chosen[j]] += 1
    chosen_set = set(chosen)
    
    for y in range(1, 1001):
        if y in chosen_set:
            continue
        ok = True
        for x in chosen:
            if counts[y + x] + 1 > 2:
                ok = False
                break
        if not ok:
            continue
        if counts[2 * y] + 1 > 2:
            continue
        
        # Add
        for x in chosen:
            counts[y + x] += 1
        counts[2 * y] += 1
        chosen.append(y)
        chosen_set.add(y)
        
    return chosen

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    best_set = []
    best_size = 0
    
    # Strategy 1: Greedy with different orderings
    orderings = [
        list(range(1, 1001)),
        list(range(1000, 0, -1)),
    ]
    
    # Add some random orderings
    rng = random.Random(42)
    for _ in range(5):
        order = list(range(1, 1001))
        rng.shuffle(order)
        orderings.append(order)
    
    # Add weighted ordering (bias towards smaller numbers? or larger?)
    # Smaller numbers create smaller sums, which are more constrained (fewer available sums).
    # Larger numbers create larger sums, which have more "room"?
    # Actually, sums are in 2..2000.
    # Let's try a mix.
    
    for order in orderings:
        if time.time() > deadline - 10.0:
            break
        s, _, _ = _greedy_build(order)
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
            
    # Strategy 2: Local Search on the best found so far
    if time.time() < deadline - 5.0:
        time_left = deadline - time.time() - 5.0
        improved = _local_search(best_set, time_limit=time_left)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(improved)
            
    # Strategy 3: More restarts with local search
    # If we have time, try a few more random starts + local search
    if time.time() < deadline - 10.0:
        for seed in range(10):
            if time.time() > deadline - 10.0:
                break
            rng = random.Random(seed + 100)
            order = list(range(1, 1001))
            rng.shuffle(order)
            s, _, _ = _greedy_build(order)
            
            # Quick local search with short time limit
            time_left = deadline - time.time() - 5.0
            if time_left > 2.0:
                s = _local_search(s, time_limit=min(5.0, time_left * 0.5))
                
            if len(s) > best_size:
                best_set = s
                best_size = len(s)
                
    # Final Polish
    if time.time() < deadline - 1.0:
        polished = _polish(best_set)
        if len(polished) > best_size:
            best_set = polished
            best_size = len(polished)
            
    return best_set

# Test/Run
if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
