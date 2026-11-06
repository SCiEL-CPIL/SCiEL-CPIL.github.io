def construct():
    import math
    import random
    import time

    n = 128
    elements = list(range(1, n + 1))
    
    # Precompute LCMs
    def get_lcm(a, b):
        return a * b // math.gcd(a, b)

    # Precompute bad pairs for each element x
    # bad_pairs_for[x] is a list of tuples (a, b) such that {x, a, b} is a bad triple
    # We store a < b to be consistent
    bad_pairs_for = [[] for _ in range(n + 1)]
    
    # To speed up checking, we can also precompute a bitmask of 'partners' for each pair?
    # Actually, checking if (a,b) is in S is O(1) with bitmasks.
    # Let's precompute the list of bad pairs for each x.
    
    # First, find all bad triples
    # A triple (a,b,c) with a<b<c is bad if lcm(a,b)==lcm(a,c)==lcm(b,c)
    
    # Optimization: Iterate a, b, c.
    # For each x, we want pairs (a,b) such that {x,a,b} is bad.
    # We can just loop all triples and populate bad_pairs_for.
    
    # Number of triples is C(128,3) ~ 350k. This is fast.
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            L_ab = get_lcm(a, b)
            for c in range(b + 1, n + 1):
                L_ac = get_lcm(a, c)
                if L_ac != L_ab:
                    continue
                L_bc = get_lcm(b, c)
                if L_bc == L_ab:
                    # Triple (a, b, c) is bad
                    # Add (b, c) to bad_pairs_for[a]
                    bad_pairs_for[a].append((b, c))
                    # Add (a, c) to bad_pairs_for[b]
                    bad_pairs_for[b].append((a, c))
                    # Add (a, b) to bad_pairs_for[c]
                    bad_pairs_for[c].append((a, b))

    def popcount(x):
        return bin(x).count('1')

    def is_valid_addition(S_mask, x):
        """
        Check if adding x to S_mask results in a valid set.
        x is 1-indexed.
        """
        # For each pair (a, b) in bad_pairs_for[x], check if both a and b are in S_mask
        for a, b in bad_pairs_for[x]:
            if (S_mask & (1 << (a - 1))) and (S_mask & (1 << (b - 1))):
                return False
        return True

    def greedy_construct():
        """
        Greedily construct a valid set.
        """
        S_mask = 0
        # Randomize order for diversity, or use a heuristic?
        # Let's use random shuffle for ILS diversity
        elems = elements[:]
        random.shuffle(elems)
        for x in elems:
            if is_valid_addition(S_mask, x):
                S_mask |= (1 << (x - 1))
        return S_mask

    def local_search(S_mask):
        """
        Improve S_mask using local search (Add and Swap).
        """
        improved = True
        while improved:
            improved = False
            
            # Phase 1: Try to add any element
            added = False
            for x in elements:
                if not (S_mask & (1 << (x - 1))):
                    if is_valid_addition(S_mask, x):
                        S_mask |= (1 << (x - 1))
                        improved = True
                        added = True
                        break # Restart loop to check for more adds
            
            # Phase 2: If no add possible, try swaps
            if not improved:
                # Try swapping: remove y, add x
                # To be efficient, we only try swaps that might lead to improvement?
                # Since we only increase size by 1 with a swap (remove 1, add 1), 
                # and we already failed to add, a swap won't increase size immediately.
                # Wait, a swap keeps size same. It helps escape local optima for ADD.
                # But if we are maximizing size, we only accept swaps if they allow a subsequent ADD?
                # That's complex.
                
                # Alternative: Try to remove a "bad" element to allow an add?
                # Let's try a simple 1-swap: remove y, add x.
                # This doesn't change size. It's only useful if it enables a future add.
                # For simplicity in the inner loop, let's just try to add.
                # If we can't add, we are stuck in a local maximum for this specific greedy order.
                
                # To escape, we need a perturbation. Local search alone with just "add" is a local max.
                # We need to allow "remove and add" that *temporarily* keeps size or reduces it?
                # No, for max size, we want to strictly increase size.
                # If we can't increase size by adding 1, we can't increase size by swapping 1 for 1.
                # We would need to remove 1 and add 2? Or remove 2 and add 3?
                
                # Let's implement a "Remove and Re-add" strategy in the ILS loop,
                # not in the tight local search loop, to keep local search fast and strictly improving.
                
                pass

        return S_mask

    def improve(S_mask):
        """
        Run local search until no more single-adds are possible.
        """
        changed = True
        while changed:
            changed = False
            for x in elements:
                if not (S_mask & (1 << (x - 1))):
                    if is_valid_addition(S_mask, x):
                        S_mask |= (1 << (x - 1))
                        changed = True
                        # Don't break, keep looking for more adds in same pass
                        # Actually, adding one might block others, but might also unblock? No, adding only adds constraints.
                        # So if x is valid, we add it. Other elements might become invalid.
                        # But we check validity dynamically.
                        # We should restart the scan or just continue?
                        # Continuing is fine, we just check current S_mask.
        return S_mask

    start_time = time.time()
    max_time = 55.0
    
    rng = random.Random(42)
    
    best_mask = 0
    best_size = 0
    
    # Initial Greedy
    curr_mask = greedy_construct()
    curr_mask = improve(curr_mask)
    curr_size = popcount(curr_mask)
    
    if curr_size > best_size:
        best_size = curr_size
        best_mask = curr_mask

    # Iterated Local Search
    while time.time() - start_time < max_time:
        # Perturb best solution
        # Remove a random number of elements (e.g., 5-10)
        S_mask = best_mask
        num_to_remove = rng.randint(3, 8)
        
        # Get list of elements in S_mask
        in_set = [x for x in elements if (S_mask & (1 << (x - 1)))]
        if len(in_set) < num_to_remove:
            continue
            
        rng.shuffle(in_set)
        for x in in_set[:num_to_remove]:
            S_mask &= ~(1 << (x - 1))
            
        # Local Search (Improve)
        S_mask = improve(S_mask)
        
        curr_size = popcount(S_mask)
        
        if curr_size > best_size:
            best_size = curr_size
            best_mask = S_mask
        else:
            # If we didn't improve, we might want to try a different perturbation strength
            # or just continue.
            pass

    result = []
    for i in range(1, n + 1):
        if best_mask & (1 << (i - 1)):
            result.append(i)
            
    if not result:
        result = [1]
        
    return result
