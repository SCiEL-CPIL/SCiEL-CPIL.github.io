import time
import random
import itertools

# Global cache
_best_cap_set_cache = {}
_start_time = None

def _is_cap_set_addition_ok(vector, current_set, n):
    """
    Check if adding 'vector' to 'current_set' keeps it a cap set.
    """
    for u in current_set:
        w = tuple((-vector[i] - u[i]) % 3 for i in range(n))
        if w in current_set:
            return False
    return True

def _improve_set(current_set, n, time_limit):
    """
    Local search to improve the cap set size.
    """
    start = time.time()
    current_set = set(current_set)
    best_size = len(current_set)
    
    # Pre-compute all vectors for small n to allow full iteration
    if 3**n <= 50000:
        all_vectors = list(itertools.product([0,1,2], repeat=n))
    else:
        all_vectors = None

    while time.time() - start < time_limit:
        improved = False
        
        # Try to add random vectors
        if all_vectors:
            candidates = [v for v in all_vectors if v not in current_set]
            random.shuffle(candidates)
        else:
            candidates = [tuple(random.randint(0,2) for _ in range(n)) for _ in range(1000)]
            
        for v in candidates:
            if v in current_set:
                continue
            if _is_cap_set_addition_ok(v, current_set, n):
                current_set.add(v)
                improved = True
                if len(current_set) > best_size:
                    best_size = len(current_set)
        
        # If we couldn't add any, try some removals and re-additions?
        # Or just restart the random shuffle.
        if not improved:
             # Try a random restart of the improvement phase or break
             # To avoid infinite loop on stuck sets, we can just break or try random swaps.
             # For now, breaking is safer if we've exhausted easy additions.
             # But we can try to remove a random element and see if we can add 2 new ones?
             # Let's keep it simple: break if no improvement in a full pass.
             break

    return current_set

def priority(vector, n):
    global _best_cap_set_cache, _start_time
    
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 280.0

    if n in _best_cap_set_cache:
        best_set = _best_cap_set_cache[n]
        return 1.0 if tuple(vector) in best_set else 0.0

    # Precompute the best cap set for this n
    # We have ~280s.
    search_time = 250.0
    
    best_set = set()
    best_size = 0
    
    start_search = time.time()
    
    # Phase 1: Multiple random greedy passes
    # Each pass is O(3^n * |Set|). For n=8, 3^8=6561. |Set|~24. 6561*24 ~ 150k ops. Very fast.
    # We can do thousands of passes.
    
    num_passes = 0
    while time.time() - start_search < search_time * 0.5:
        current_set = set()
        candidates = list(itertools.product([0,1,2], repeat=n))
        random.shuffle(candidates)
        
        for v in candidates:
            if _is_cap_set_addition_ok(v, current_set, n):
                current_set.add(v)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
        num_passes += 1
        
    # Phase 2: Local improvement on the best set found
    remaining_time = search_time - (time.time() - start_search)
    if remaining_time > 5:
        improved_set = _improve_set(best_set, n, remaining_time)
        if len(improved_set) > best_size:
            best_set = improved_set
            best_size = len(best_set)

    _best_cap_set_cache[n] = best_set
    
    return 1.0 if tuple(vector) in best_set else 0.0
