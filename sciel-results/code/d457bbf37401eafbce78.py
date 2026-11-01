import itertools
import time
import random

def construct(q, n, radius):
    # This solution is specialized for q=3, n=6, radius=1
    if q != 3 or n != 6 or radius != 1:
        # Fallback to tetracode direct sum for other cases (though task is fixed)
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(itertools.product(range(3), repeat=n))[0]] # Dummy

    start_time = time.time()
    deadline = start_time + 115.0 # 5s safety margin
    
    # Precompute neighbors for all 729 words
    neighbors = []
    for c in range(729):
        digs = [(c // (3**i)) % 3 for i in range(6)]
        ns = [c]
        for i in range(6):
            for delta in [-1, 1]:
                new_dig = (digs[i] + delta) % 3
                # Calculate new value
                # Instead of full conversion, we can update based on place value
                # 3^i is the weight of digit i
                weight = 3**i
                old_val = digs[i] * weight
                new_val = new_dig * weight
                ns.append(c - old_val + new_val)
        neighbors.append(ns)

    # Initial set: Tetracode direct sum
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codes = []
    for h in tetracode:
        for t in itertools.product(range(3), repeat=2):
            code_digits = h + list(t)
            val = 0
            for i, d in enumerate(code_digits):
                val += d * (3**i)
            initial_codes.append(val)
    
    current_codes = list(set(initial_codes)) # Ensure unique, though they should be
    code_set = set(current_codes)
    
    # Coverage array
    cov = [0] * 729
    # Coverers: list of sets, coverers[w] = set of codeword values covering w
    coverers = [set() for _ in range(729)]
    
    for c in current_codes:
        for w in neighbors[c]:
            cov[w] += 1
            coverers[w].add(c)
            
    best_size = len(current_codes)
    best_codes = current_codes[:]
    
    def remove_code(c):
        for w in neighbors[c]:
            cov[w] -= 1
            coverers[w].remove(c)
        code_set.remove(c)
        current_codes.remove(c)
        
    def add_code(c):
        for w in neighbors[c]:
            cov[w] += 1
            coverers[w].add(c)
        code_set.add(c)
        current_codes.append(c)
        
    def is_removable(c):
        for w in neighbors[c]:
            if cov[w] == 1:
                return False
        return True
        
    def try_removals():
        removed = False
        # Repeat until no more removals possible
        while True:
            any_removed = False
            # Iterate over a copy because we modify the list
            for c in current_codes[:]:
                if is_removable(c):
                    remove_code(c)
                    any_removed = True
                    removed = True
            if not any_removed:
                break
        return removed

    def try_swaps():
        swapped = False
        # Try to find a codeword to swap
        # Heuristic: Find codeword with smallest exclusive set
        candidates_for_swap = []
        for c in current_codes:
            exclusive = []
            for w in neighbors[c]:
                if cov[w] == 1:
                    exclusive.append(w)
            if exclusive:
                candidates_for_swap.append((len(exclusive), c, exclusive))
                
        if not candidates_for_swap:
            return False # Should have been removed in try_removals
            
        # Sort by size of exclusive set
        candidates_for_swap.sort(key=lambda x: x[0])
        
        for size, c, exclusive in candidates_for_swap:
            # Look for a new codeword c' that covers all words in 'exclusive'
            # and is not in current_codes
            potential_new = None
            # Intersect coverers of all exclusive words
            common = set(coverers[exclusive[0]])
            for w in exclusive[1:]:
                common &= coverers[w]
                if not common:
                    break
            
            if common:
                # Remove current codewords from common
                common -= code_set
                if common:
                    # Pick one (randomly to escape local minima?)
                    potential_new = next(iter(common))
                    # Perform swap
                    remove_code(c)
                    add_code(potential_new)
                    swapped = True
                    break
        
        return swapped

    # Main Loop
    while time.time() < deadline:
        # 1. Try removing
        if try_removals():
            if len(current_codes) < best_size:
                best_size = len(current_codes)
                best_codes = current_codes[:]
            # After removals, maybe more removals are possible? 
            # try_removals loops until stable, so we are stable now.
            # But the structure changed, so maybe swaps are possible now?
            # Or just continue to next iteration.
            continue
            
        # 2. Try swapping
        if try_swaps():
            # After a swap, the coverage changed.
            # It might allow new removals.
            # So we go back to top of loop which calls try_removals.
            continue
            
        # 3. Stuck. Perturb.
        # Add a random codeword not in set
        # To speed up, pick a random word from 0..728
        # If it's in set, try another
        attempts = 0
        while attempts < 10:
            r = random.randint(0, 728)
            if r not in code_set:
                add_code(r)
                break
            attempts += 1
        else:
            # If we couldn't add a random code (set is full? unlikely), break
            break
            
        # After perturbation, we have a larger set.
        # Run removals to get back down.
        try_removals()
        
        if len(current_codes) < best_size:
            best_size = len(current_codes)
            best_codes = current_codes[:]
            
        # If we didn't improve after perturbation and subsequent removals,
        # and we are stuck in a loop, we might want to revert to best?
        # For now, just keep exploring. The random add helps escape local minima.
        
    # Convert best_codes back to list of lists
    result = []
    for c in best_codes:
        digs = [(c // (3**i)) % 3 for i in range(6)]
        result.append(digs)
        
    return result
