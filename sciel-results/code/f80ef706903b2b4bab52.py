import random
import time

def construct():
    """
    Returns a nonempty subset of [1, 100] with no subset summing to 200.
    Maximizes cardinality.
    """
    TARGET = 200
    N = 100
    MASK = (1 << (TARGET + 1)) - 1
    
    def try_greedy(order, initial_bits=None, initial_selected=None):
        """Greedy: add elements from order if they don't create sum TARGET."""
        if initial_bits is not None:
            bits = initial_bits
            selected = list(initial_selected)
        else:
            bits = 1  # Bit 0 is set, representing sum 0
            selected = []
        for x in order:
            # Check if adding x creates a subset sum of 200
            if (bits | (bits << x)) & (1 << TARGET):
                continue
            # Safe to add x
            selected.append(x)
            bits = (bits | (bits << x)) & MASK
        return selected, bits
    
    # Start with proven base: {66, 67, ..., 100} - 35 elements
    # No two elements can sum to 200 (min pair sum = 133, but we need to check all subsets)
    # Actually, since all elements >= 66, and 66+67=133, 66+68=134, ..., 
    # The only way to get 200 is with 2 or 3 elements.
    # 2 elements: a+b=200, a,b in [66,100]. Pairs: (66,134) no, (67,133) no, ...
    # Actually max element is 100, so a+b=200 requires both >= 100, i.e., (100,100) but no duplicates.
    # So no 2-element subset sums to 200.
    # 3 elements: a+b+c=200, min sum = 66+67+68=201 > 200. So no 3+ element subset sums to 200.
    # Therefore {66,...,100} is valid!
    
    base_set = list(range(66, 101))
    base_bits = 1
    for x in base_set:
        base_bits = (base_bits | (base_bits << x)) & MASK
    
    best_set = base_set[:]
    best_size = len(base_set)
    
    # Now try to add elements from [1, 65] using various greedy orders
    candidates = list(range(1, 66))
    
    # Deterministic orders
    for order in [candidates, candidates[::-1]]:
        result, _ = try_greedy(order, base_bits, best_set)
        if len(result) > best_size:
            best_set = result
            best_size = len(result)
    
    # Random restarts with a time limit
    start_time = time.time()
    time_limit = 50.0  # Safety margin before 90s kill
    
    while time.time() - start_time < time_limit:
        random_order = candidates[:]
        random.shuffle(random_order)
        result, _ = try_greedy(random_order, base_bits, best_set)
        if len(result) > best_size:
            best_set = result
            best_size = len(result)
    
    return best_set
