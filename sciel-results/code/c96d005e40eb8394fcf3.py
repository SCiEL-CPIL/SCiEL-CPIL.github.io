import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    def get_counts(S):
        counts = {}
        S_sorted = sorted(S)
        for i, a in enumerate(S_sorted):
            for b in S_sorted[i:]:
                s = a + b
                counts[s] = counts.get(s, 0) + 1
        return counts

    def is_valid(S):
        # Optimized check
        if len(S) == 0:
            return True
        S_sorted = sorted(S)
        counts = {}
        for i, a in enumerate(S_sorted):
            for b in S_sorted[i:]:
                s = a + b
                counts[s] = counts.get(s, 0) + 1
                if counts[s] > 2:
                    return False
        return True

    def greedy_construct(order):
        S = []
        counts = {}
        for x in order:
            # Check if adding x is valid
            # New sums: x + y for y in S, and 2x
            # We need to check if any new sum exceeds 2
            valid = True
            for y in S:
                s = x + y
                if counts.get(s, 0) + 1 > 2:
                    valid = False
                    break
            if valid:
                s = 2 * x
                if counts.get(s, 0) + 1 > 2:
                    valid = False
            
            if valid:
                S.append(x)
                for y in S[:-1]:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                s = 2 * x
                counts[s] = counts.get(s, 0) + 1
        
        return S, counts

    best_set = []
    best_size = 0
    
    # Phase 1: Randomized Greedy
    # Try multiple random permutations
    base_order = list(range(1, 1001))
    
    # Always try the natural order first (seed behavior)
    S, counts = greedy_construct(base_order)
    best_set = S
    best_size = len(S)
    
    while time.time() < deadline:
        random.shuffle(base_order)
        S, counts = greedy_construct(base_order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
    # Phase 2: Local Search on the best set found
    # We try to add elements or swap to allow adding elements
    # We need to maintain counts for efficiency
    
    if best_set:
        S = best_set
        S_set = set(S)
        # Reconstruct counts for the current S
        counts = {}
        S_sorted = sorted(S)
        for i, a in enumerate(S_sorted):
            for b in S_sorted[i:]:
                s = a + b
                counts[s] = counts.get(s, 0) + 1
        
        # Function to try adding x
        def try_add(x, S, S_set, counts):
            if x in S_set:
                return False
            # Check validity
            valid = True
            for y in S:
                s = x + y
                if counts.get(s, 0) + 1 > 2:
                    valid = False
                    break
            if valid:
                s = 2 * x
                if counts.get(s, 0) + 1 > 2:
                    valid = False
            
            if valid:
                # Commit
                S.append(x)
                S_set.add(x)
                for y in S[:-1]:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                s = 2 * x
                counts[s] = counts.get(s, 0) + 1
                return True
            return False

        # Function to try swap y -> x
        def try_swap(y, x, S, S_set, counts):
            if x in S_set or x == y:
                return False
            
            # Tentatively remove y
            # We need to decrement counts for sums involving y
            S_without_y = [z for z in S if z != y]
            # This is O(N). For N~200, it's fine.
            
            # But we need to check if adding x to S_without_y is valid.
            # Let's simulate.
            # New sums from x: x+z for z in S_without_y, and 2x.
            # Old counts minus contributions from y.
            
            # To be efficient, let's just check validity of (S \ {y}) U {x}
            # We can do this by checking the "delta".
            
            # Simpler: Just check if (S \ {y}) U {x} is valid using is_valid.
            # is_valid is O(N^2). With N~200, N^2=40000.
            # If we do this many times, it might be slow.
            # But we have 115 seconds.
            
            candidate = S_without_y + [x]
            if is_valid(candidate):
                # Commit
                S.remove(y)
                S_set.remove(y)
                S.append(x)
                S_set.add(x)
                
                # Update counts
                # Recalculate counts for simplicity? Or update incrementally?
                # Recalculating is O(N^2) = 40000 ops.
                # If we do 1000 swaps, that's 40M ops. Python might struggle in 115s?
                # 40M simple ops in Python is ~2-4 seconds. So we can do ~30-50 recalcs per second.
                # So ~1000-2000 swaps in 115s. That's reasonable.
                
                counts = {}
                S_sorted = sorted(S)
                for i, a in enumerate(S_sorted):
                    for b in S_sorted[i:]:
                        s = a + b
                        counts[s] = counts.get(s, 0) + 1
                return True
            return False

        # Local Search Loop
        # Try to add any missing element
        missing = [x for x in range(1, 1001) if x not in S_set]
        
        while time.time() < deadline:
            # 1. Try to add a random missing element
            if missing:
                random.shuffle(missing)
                added = False
                for x in missing:
                    if try_add(x, S, S_set, counts):
                        missing.remove(x)
                        added = True
                        break
                if added:
                    continue
            
            # 2. If no element could be added, try swaps to escape local optimum
            # Pick a random y from S, and a random x from missing
            # To be effective, we should pick x that is "close" to being valid
            # For now, random swap.
            
            if not missing:
                break # All elements included, done
            
            # Try a few random swaps
            swap_success = False
            for _ in range(10): # Try 10 random swaps
                if not missing:
                    break
                y = random.choice(S)
                x = random.choice(missing)
                if try_swap(y, x, S, S_set, counts):
                    # Update missing list
                    if x not in missing:
                        pass
                    else:
                        missing.remove(x)
                    # y is now missing
                    if y not in missing:
                        missing.append(y)
                    swap_success = True
                    break
            
            if not swap_success:
                # Try a different strategy: randomize S slightly?
                # Or just break if stuck for too long?
                # Let's try a "kick": remove 2, add 1? No, we want to increase size.
                # Since we can't increase size with 1-for-1 swap, we are stuck at this size.
                # To increase size, we need to find a configuration where an element CAN be added.
                # This requires changing the set structure.
                # A 2-for-1 swap (remove 2, add 1) decreases size. We don't want that.
                # A 1-for-2 swap (remove 1, add 2) increases size by 1.
                # Let's try 1-for-2 swaps.
                
                # Pick y in S, x1, x2 in missing.
                # Check if (S \ {y}) U {x1, x2} is valid.
                # This is more expensive.
                
                # For now, let's just break if we can't find a simple swap.
                # The randomized greedy phase should have found a good starting point.
                break

    return best_set
