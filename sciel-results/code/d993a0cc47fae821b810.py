import time

_P0 = time.time()
_DEADLINE = _P0 + 85.0

def _can_add(prods, x):
    """Check if adding x to a set with existing products prods keeps all distinct."""
    for p in prods:
        if p * x in prods:
            return False
    return True

def _update_prods(prods, x):
    """Update products set by multiplying by x."""
    new_prods = set(prods)
    for p in prods:
        new_prods.add(p * x)
    return new_prods

def construct():
    import time
    deadline = _DEADLINE
    
    # Primes in 2..40
    primes = [a for a in range(2, 41) if all(a % d for d in range(2, int(a**0.5)+1))]
    composites = [a for a in range(2, 41) if a not in primes]
    
    best_set = primes[:]
    best_prods = _update_prods({1}, None)
    # Recompute properly
    best_prods = {1}
    for x in best_set:
        best_prods = _update_prods(best_prods, x)
    
    # Try greedy with different orderings
    orders = [
        primes + composites,
        composites + primes,
        list(range(2, 41)),
        list(range(40, 1, -1)),
        # Sort by some heuristic
        sorted(range(2, 41), key=lambda x: -len([d for d in range(2, x) if x % d == 0])),
        sorted(range(2, 41), key=lambda x: x),
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        cur_set = []
        cur_prods = {1}
        for x in order:
            if time.time() > deadline:
                break
            if _can_add(cur_prods, x):
                cur_set.append(x)
                cur_prods = _update_prods(cur_prods, x)
        
        if len(cur_set) > len(best_set):
            best_set = cur_set[:]
            best_prods = cur_prods
    
    # Local search: try swaps
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(len(best_set)):
                if time.time() > deadline - 2:
                    break
                for x in range(2, 41):
                    if x in best_set:
                        continue
                    # Try swap: remove best_set[i], add x
                    test_set = [a for j, a in enumerate(best_set) if j != i] + [x]
                    test_prods = {1}
                    valid = True
                    for y in test_set:
                        if not _can_add(test_prods, y):
                            valid = False
                            break
                        test_prods = _update_prods(test_prods, y)
                    if valid and len(test_set) == len(best_set):
                        # Same size, but maybe allows further additions?
                        # For now, just track if we can improve
                        pass
                    if valid:
                        # Try to add more after swap
                        new_set = test_set[:]
                        new_prods = test_prods
                        for z in range(2, 41):
                            if z in new_set:
                                continue
                            if _can_add(new_prods, z):
                                new_set.append(z)
                                new_prods = _update_prods(new_prods, z)
                        if len(new_set) > len(best_set):
                            best_set = new_set
                            best_prods = new_prods
                            improved = True
                            break
                if improved:
                    break
            # Also try just adding
            for x in range(2, 41):
                if x in best_set:
                    continue
                if _can_add(best_prods, x):
                    best_set.append(x)
                    best_prods = _update_prods(best_prods, x)
                    improved = True
                    break
    
    return list(best_set)
