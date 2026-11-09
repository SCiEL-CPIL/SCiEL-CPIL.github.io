import time
import random

START_TIME = time.time()
DEADLINE = 55.0  # Safety margin

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

def compute_sums(S):
    """
    Compute the bitset representation of subset sums for each cardinality.
    sums[k] is an integer where bit s is 1 if sum s is achievable with k elements.
    """
    n = len(S)
    sums = [0] * (n + 1)
    sums[0] = 1  # Empty set sum is 0
    
    for x in S:
        # Update from high cardinality to low to avoid using x multiple times in one step
        # But since we are iterating over elements, we update k from current_max down to 0
        # Actually, standard 0/1 knapsack style update:
        # for k in range(current_n, -1, -1):
        #    sums[k+1] |= sums[k] << x
        
        # We need to know the current max cardinality processed.
        # Let's just iterate k from n-1 down to 0, but sums[k] might be 0 if k > current processed count.
        # It's safer to track the current size or just iterate all and rely on 0s.
        for k in range(n - 1, -1, -1):
            if sums[k]:
                sums[k + 1] |= (sums[k] << x)
                
    return sums

def is_valid(sums):
    """
    Check if the bitsets in 'sums' represent a valid cardinality-separated set.
    sums[i] & sums[j] must be 0 for all i != j.
    """
    n = len(sums) - 1
    for i in range(n + 1):
        for j in range(i + 1, n + 2):
            if sums[i] & sums[j]:
                return False
    return True

def greedy_initial():
    """
    Generate an initial valid set using greedy ascending approach.
    """
    A = []
    sums = [0] * 61
    sums[0] = 1
    n = 0
    for x in range(1, 257):
        if is_done():
            break
        # Try adding x
        new_sums = sums[:]
        for k in range(n, -1, -1):
            new_sums[k + 1] |= (new_sums[k] << x)
        
        if is_valid(new_sums):
            A.append(x)
            sums = new_sums
            n += 1
            if n >= 60:
                break
    return A, sums, n

def construct():
    # 1. Initial Greedy
    best_set, best_sums, best_n = greedy_initial()
    
    # 2. Local Search / Simulated Annealing style exploration
    # We maintain the current best set and try to improve it or find a larger one via swaps.
    # To allow exploration, we keep a "current" set which might be different from best_set,
    # but we only update best_set if current is larger.
    
    current_set, current_sums, current_n = best_set[:], best_sums[:], best_n
    
    # Precompute all possible candidates
    all_candidates = set(range(1, 257))
    
    while not is_done():
        # Try to add an element to current_set
        added = False
        # Shuffle candidates to avoid bias
        candidates_to_try = [x for x in all_candidates if x not in current_set]
        random.shuffle(candidates_to_try)
        
        for x in candidates_to_try:
            if is_done():
                break
            # Try adding x
            new_sums = current_sums[:]
            # We need to extend new_sums if current_n + 1 > len(new_sums)-1? 
            # We allocated 61 slots, so up to size 60 is fine.
            for k in range(current_n, -1, -1):
                new_sums[k + 1] |= (new_sums[k] << x)
            
            if is_valid(new_sums):
                current_set.append(x)
                current_sums = new_sums
                current_n += 1
                added = True
                if current_n > best_n:
                    best_set = current_set[:]
                    best_sums = current_sums[:]
                    best_n = current_n
                break
        
        if added:
            continue
        
        # If no element could be added, try swaps to change the structure
        # This allows us to escape local optima where no single addition is valid,
        # but a swap might open up the space for an addition.
        
        # Pick a random element to remove
        if current_n > 0:
            u_idx = random.randint(0, current_n - 1)
            u = current_set[u_idx]
            
            # Pick a random candidate to add
            candidates_to_swap = [x for x in all_candidates if x not in current_set]
            if candidates_to_swap:
                v = random.choice(candidates_to_swap)
                
                # Form new set
                # Remove u, add v
                new_set = current_set[:]
                new_set[u_idx] = v
                
                # Recompute sums for new_set
                # This is O(N * WordOps), which is fast for N < 60
                new_sums = compute_sums(new_set)
                
                if is_valid(new_sums):
                    # Accept the swap
                    current_set = new_set
                    current_sums = new_sums
                    # current_n remains the same
                    # Note: We don't update best_set here unless we can add later
                    # We just move the "current" state to a new valid configuration
                    pass

    # Final validation and return
    # Ensure types are correct
    return [int(x) for x in best_set]

