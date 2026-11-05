def construct():
    import random
    import time
    from math import gcd
    from itertools import combinations

    def can_add(x, current_set):
        # Check if adding x to current_set creates a forbidden triple
        for a, b in combinations(current_set, 2):
            if gcd(a, b) == gcd(a, x) == gcd(b, x):
                return False
        return True

    def greedy_fill(start_set, remaining):
        """
        Greedily add elements from remaining to start_set.
        Returns the new set.
        """
        current_set = start_set.copy()
        # Sort remaining by some heuristic? Random is fine for restarts.
        # For local search, we might want deterministic or random.
        for x in remaining:
            if can_add(x, current_set):
                current_set.add(x)
        return current_set

    def local_search_improve(S):
        """
        Try to improve set S by removing one element and refilling.
        Returns the best set found.
        """
        best_local = S
        best_size = len(S)
        remaining = set(range(1, 129)) - S
        
        # Try removing each element
        for y in list(S):
            S_minus_y = S - {y}
            # Refill with greedy
            # We need to be careful: can_add checks against the current set.
            # We should try to add elements from (remaining U {y}) to S_minus_y
            candidates = list(remaining) + [y]
            random.shuffle(candidates) # Randomize refill order to escape local minima
            
            new_set = S_minus_y
            for x in candidates:
                if can_add(x, new_set):
                    new_set.add(x)
            
            if len(new_set) > best_size:
                best_size = len(new_set)
                best_local = new_set.copy()
                # Early exit if we found a larger set? Or continue to find even larger?
                # Continuing might be expensive. Let's just take the first improvement or best of few.
                # Given time limits, maybe just check a subset of y's?
                # With 128 elements, checking all y's is 128 * (Greedy Cost).
                # Greedy Cost is ~ 128 * (128^2) ~ 2M ops.
                # 128 * 2M = 256M ops. Too slow for many iterations.
                # We need to optimize can_add or limit checks.
                
        return best_local

    best_set = set()
    best_size = 0
    start_time = time.time()
    time_limit = 55  # Leave 5s buffer
    
    # Phase 1: Randomized Greedy Restarts
    restarts = 0
    while time.time() - start_time < time_limit / 2:
        current_set = set()
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(x, current_set):
                current_set.add(x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            restarts += 1
            
    # Phase 2: Local Search on Best Set
    # Try to improve best_set
    # We will try a few rounds of local search
    for _ in range(10):
        if time.time() - start_time > time_limit:
            break
            
        # Try removing each element in best_set and refilling
        # To speed up, we can randomize the order of y's and stop if we find an improvement
        ys = list(best_set)
        random.shuffle(ys)
        improved = False
        for y in ys:
            if time.time() - start_time > time_limit:
                break
            S_minus_y = best_set - {y}
            remaining = set(range(1, 129)) - best_set
            candidates = list(remaining) + [y]
            random.shuffle(candidates)
            
            new_set = S_minus_y
            for x in candidates:
                if can_add(x, new_set):
                    new_set.add(x)
            
            if len(new_set) > len(best_set):
                best_set = new_set
                best_size = len(best_set)
                improved = True
                break # Restart local search with new best
        
    # Final pass: try to add any remaining
    improved = True
    while improved:
        improved = False
        remaining = set(range(1, 129)) - best_set
        for x in list(remaining):
            if can_add(x, best_set):
                best_set.add(x)
                remaining.discard(x)
                improved = True
                
    return list(best_set)
