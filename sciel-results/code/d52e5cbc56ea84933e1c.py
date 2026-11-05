import time
import random

def construct():
    N = 512
    
    # Use a list for O(1) membership test, faster than set in tight loops in Python
    in_set = [False] * (N + 1)
    S = set() # Keep set for easy iteration/sampling, or just use the list?
    # Actually, maintaining both is overhead. Let's use the list for checks and a list of active elements for iteration.
    # Or just use the set, but the lookup `x in S` is O(1) average.
    # The bottleneck is the loop in is_valid_addition.
    
    def is_valid_addition(x):
        """
        Check if adding x to the current set (represented by in_set) creates a 4-AP.
        """
        # x is term 1: x+d, x+2d, x+3d in S
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            if in_set[x + d] and in_set[x + 2 * d] and in_set[x + 3 * d]:
                return False
        
        # x is term 2: x-d, x+d, x+2d in S
        max_d = min(x - 1, (N - x) // 2)
        for d in range(1, max_d + 1):
            if in_set[x - d] and in_set[x + d] and in_set[x + 2 * d]:
                return False
        
        # x is term 3: x-2d, x-d, x+d in S
        max_d = min((x - 1) // 2, N - x)
        for d in range(1, max_d + 1):
            if in_set[x - 2 * d] and in_set[x - d] and in_set[x + d]:
                return False
        
        # x is term 4: x-3d, x-2d, x-d in S
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if in_set[x - 3 * d] and in_set[x - 2 * d] and in_set[x - d]:
                return False
        
        return True

    def add_to_set(x):
        in_set[x] = True
        S.add(x)

    def remove_from_set(x):
        in_set[x] = False
        S.discard(x)

    def greedy_fill(candidates=None, order=None):
        """
        Fill the set with valid elements.
        """
        if order is None:
            if candidates is None:
                candidates = [x for x in range(1, N + 1) if not in_set[x]]
            order = candidates
        else:
            # order is a list of elements to try
            pass
            
        for x in order:
            if not in_set[x]:
                if is_valid_addition(x):
                    add_to_set(x)

    def local_optimize():
        """
        Repeatedly try to add any valid element until no more can be added.
        Since adding elements only restricts future additions, a single pass 
        over a *fixed* list is not enough if the list order is bad?
        Actually, if we iterate 1..N, and add 1, it might block 2. 
        If we iterated 2..1, we might add 2, which might block 1.
        So the result depends on order.
        To find a local optimum, we should try to add *any* valid element.
        We can do this by repeatedly scanning the entire range [1, N] and adding 
        any valid element found, until a full scan adds nothing.
        This is O(N^2 * cost_check) worst case, but N=512 is small.
        """
        improved = True
        while improved:
            improved = False
            for x in range(1, N + 1):
                if not in_set[x]:
                    if is_valid_addition(x):
                        add_to_set(x)
                        improved = True
        return len(S)

    # 1. Initial Greedy (Deterministic Order 1..N)
    for x in range(1, N + 1):
        if is_valid_addition(x):
            add_to_set(x)
    
    # Ensure we are at a local optimum
    local_optimize()
    
    best_size = len(S)
    best_set_copy = set(S)
    
    start_time = time.time()
    timeout = 55  # Safety margin
    
    # 2. Iterated Local Search
    # We will perturb the best set and re-optimize.
    while time.time() - start_time < timeout:
        # Create a perturbed version
        # We work on the current global in_set/S state? 
        # It's safer to snapshot the best state, perturb it, and if it improves, commit.
        # But snapshotting a set of ~200 ints is cheap.
        
        # Restore best state
        S.clear()
        in_set = [False] * (N + 1)
        for x in best_set_copy:
            in_set[x] = True
            S.add(x)
            
        # Perturb: Remove k random elements
        # k should be small enough to keep structure, large enough to change basin.
        k = random.randint(5, 15)
        if len(S) <= k:
            continue
            
        elements_to_remove = random.sample(list(S), k)
        for e in elements_to_remove:
            remove_from_set(e)
            
        # Re-optimize
        # Instead of just local_optimize (which adds in 1..N order), 
        # we can try a few random orders for the initial fill phase of the local search?
        # local_optimize already does a "fixpoint" search which is order-independent 
        # in the sense that it finds *a* maximal independent set relative to the 
        # current state? No, the *set* of elements added can depend on the order 
        # if there are conflicts. But since we loop until no change, and we check 
        # ALL elements in each pass, the final set is the *unique* maximal set 
        # containing the current S? 
        # Wait. If A and B conflict, and neither is in S. 
        # Pass 1: Check A. Valid? Add A. Check B. Invalid (conflicts with A).
        # If order was B then A: Check B. Valid? Add B. Check A. Invalid.
        # So the result is NOT unique. It depends on the order of checks in the `for x in range(1, N+1)` loop.
        # The current `local_optimize` always uses 1..N order.
        # To explore more, we can randomize the order in `local_optimize`?
        # That would be expensive (multiple passes).
        
        # Let's try: Run local_optimize with a random permutation of candidates?
        # No, local_optimize as defined above scans 1..N repeatedly.
        # Let's modify local_optimize to accept an order or use random shuffling of the candidate list in the first pass?
        
        # Simpler: Just run the standard local_optimize. The perturbation changes the starting point.
        local_optimize()
        
        current_size = len(S)
        if current_size > best_size:
            best_size = current_size
            best_set_copy = set(S)
            
    return sorted(best_set_copy)

# EVOLVE-BLOCK-END
