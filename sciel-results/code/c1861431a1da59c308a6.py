import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    # Common denominator: lcm(1, 2, ..., 22) = 232792560
    L = 232792560
    Ldiv = {x: L // x for x in range(1, 23)}
    
    def can_add(sums_set, x):
        val = Ldiv[x]
        for s in sums_set:
            if s + val in sums_set:
                return False
        return True
    
    def add_element(sums_set, x):
        val = Ldiv[x]
        return sums_set | {s + val for s in sums_set}
    
    def greedy(order):
        result = []
        sums_set = {0}
        for x in order:
            if can_add(sums_set, x):
                sums_set = add_element(sums_set, x)
                result.append(x)
        return result
    
    best = None
    
    # Try multiple orderings
    orderings = [
        list(range(1, 23)),           # increasing
        list(range(22, 0, -1)),       # decreasing
    ]
    
    random.seed(42)
    for _ in range(100):
        order = list(range(1, 23))
        random.shuffle(order)
        orderings.append(order)
    
    for order in orderings:
        if time.time() > deadline:
            break
        result = greedy(order)
        if best is None or len(result) > len(best):
            best = result
    
    # Try to extend the best solution
    if best:
        best_set = set(best)
        sums_set = {0}
        for x in best:
            sums_set = add_element(sums_set, x)
        
        remaining = [x for x in range(1, 23) if x not in best_set]
        for x in remaining:
            if time.time() > deadline:
                break
            if can_add(sums_set, x):
                sums_set = add_element(sums_set, x)
                best.append(x)
    
    if not best:
        best = [1]
    
    # Final verification
    sums_set = {0}
    for x in best:
        sums_set = add_element(sums_set, x)
    assert len(sums_set) == 2 ** len(best), "Validation failed"
    
    return sorted(best)
