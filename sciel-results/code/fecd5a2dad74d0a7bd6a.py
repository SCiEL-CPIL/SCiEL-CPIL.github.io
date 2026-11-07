"""Combined program: product-based check + prime-first systematic search + local improvement."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 40, 'maximum_size': 18}

def _is_prime(a):
    if a < 2:
        return False
    if a < 4:
        return True
    if a % 2 == 0 or a % 3 == 0:
        return False
    d = 5
    while d * d <= a:
        if a % d == 0 or a % (d + 2) == 0:
            return False
        d += 6
    return True

def _check_valid(S):
    """
    Validate that all subset products of S are distinct.
    Uses the incremental set expansion method.
    """
    current = {1}
    for x in S:
        new_vals = {p * x for p in current}
        if current & new_vals:
            return False
        current |= new_vals
    return True

def _try_add(S, x):
    """Check if x can be added to S while maintaining distinct subset products."""
    current = {1}
    for v in S:
        new_vals = {p * v for p in current}
        if current & new_vals:
            return False
        current |= new_vals
    # Now check x
    new_vals = {p * x for p in current}
    if current & new_vals:
        return False
    return True

def _try_greedy(order, existing=None):
    """
    Greedily add elements from order to build a valid set.
    """
    current_set = list(existing) if existing else []
    prods = {1}
    for v in order:
        if v in current_set:
            continue
        new_prods = []
        valid = True
        for p in prods:
            np = v * p
            if np in prods:
                valid = False
                break
            new_prods.append(np)
        if valid:
            current_set.append(v)
            prods.update(new_prods)
    return current_set

def _local_search(S, rng, time_limit):
    """
    Local search: try swapping one element for another to improve the set.
    """
    best = list(S)
    best_size = len(best)
    candidates = list(range(2, P['n'] + 1))
    
    start_time = time.time()
    while time.time() < time_limit:
        improved = False
        for i in range(len(best)):
            for j in candidates:
                if j in best:
                    continue
                # Try replacing best[i] with j
                new_set = best[:i] + best[i+1:] + [j]
                if _check_valid(new_set):
                    # Try to extend
                    remaining = [x for x in candidates if x not in new_set]
                    rng.shuffle(remaining)
                    extended = _try_greedy(remaining, existing=new_set)
                    if len(extended) > best_size:
                        best = extended
                        best_size = len(best)
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    candidates = list(range(2, P['n'] + 1))
    primes = [a for a in candidates if _is_prime(a)]
    composites = [a for a in candidates if not _is_prime(a)]
    
    # Start with all primes (always valid)
    best_set = primes[:]
    best_size = len(best_set)
    
    # Systematically try adding composites
    # For each composite, check if it can be added
    for c in composites:
        if _try_add(best_set, c):
            best_set.append(c)
            best_size += 1
    
    # Now try local search to improve
    rng = random.Random(42)
    remaining_time = deadline - time.time()
    if remaining_time > 2.0:
        local_time = min(remaining_time * 0.5, 20.0)
        improved = _local_search(best_set, rng, local_time)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved[:]
    
    # Randomized restarts if time remains
    remaining_time = deadline - time.time()
    while remaining_time > 1.0:
        # Shuffle composites and try greedy
        c_shuffled = composites[:]
        rng.shuffle(c_shuffled)
        order = primes + c_shuffled
        sol = _try_greedy(order)
        if len(sol) > best_size:
            best_size = len(sol)
            best_set = sol[:]
        
        # Local search on new best
        remaining_time = deadline - time.time()
        if remaining_time > 3.0:
            local_time = min(remaining_time * 0.3, 8.0)
            improved = _local_search(best_set, rng, local_time)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved[:]
        
        if best_size >= 18:
            break
        remaining_time = deadline - time.time()
    
    # Final validation
    if not _check_valid(best_set):
        best_set = primes[:]
        best_size = len(best_set)
    
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
