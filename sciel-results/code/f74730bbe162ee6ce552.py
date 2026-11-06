<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    # 1. Cantor set seed
    cantor = _cantor_bits()
    best = cantor
    best_size = bin(cantor).count('1')
    
    # 2. Try forward greedy from scratch
    fwd = _greedy_forward()
    fwd_size = bin(fwd).count('1')
    if fwd_size > best_size:
        best = fwd
        best_size = fwd_size
    
    # 3. Local search on the best so far
    if time.time() < deadline - 30:
        improved = _local_search(best, deadline - 10)
        imp_size = bin(improved).count('1')
        if imp_size > best_size:
            best = improved
            best_size = imp_size
    
    # 4. Randomized restarts
    rng = random.Random(12345)
    restart_count = 0
    while time.time() < deadline - 20 and restart_count < 30:
        restart_count += 1
        
        # Different strategies
        r = rng.random()
        if r < 0.3:
            # Start from Cantor, add randomly
            bits = _greedy_random(cantor, rng)
        elif r < 0.6:
            # Pure random greedy
            bits = _greedy_random(0, rng)
        else:
            # Start from best, perturb: remove a random element, then re-greedy
            bits = best
            if best_size > 1:
                # Remove a random element
                elems = _bits_to_set(bits)
                if elems:
                    rem = rng.choice(elems)
                    bits = bits & ~(1 << rem)
                    bits = _greedy_forward(bits)
        
        size = bin(bits).count('1')
        if size > best_size:
            best = bits
            best_size = size
        
        # Local search on this result
        if time.time() < deadline - 15:
            improved = _local_search(bits, deadline - 10)
            imp_size = bin(improved).count('1')
            if imp_size > best_size:
                best = improved
                best_size = imp_size
    
    # 5. Final local search on best
    if time.time() < deadline - 5:
        improved = _local_search(best, deadline)
        imp_size = bin(improved).count('1')
        if imp_size > best_size:
            best = improved
            best_size = imp_size
    
    # 6. Verify
    if not _is_valid(best):
        best = cantor
    
    return sorted(_bits_to_set(best))
=======
def _kick_search(bits, deadline, rng, max_kicks=10):
    """
    Kick search: remove k random elements, re-greedy, and keep if better.
    Helps escape local optima.
    """
    current = bits
    current_size = bin(current).count('1')
    
    for _kick in range(max_kicks):
        if time.time() > deadline - 3:
            break
        
        # Choose how many elements to remove
        k = rng.choice([2, 3, 4, 5])
        if current_size <= k:
            break
        
        # Remove k random elements
        elems = _bits_to_set(current)
        to_remove = rng.sample(elems, min(k, len(elems)))
        trial = current
        for rem in to_remove:
            trial &= ~(1 << rem)
        
        # Re-greedy
        trial = _greedy_forward(trial, deadline - 3)
        trial_size = bin(trial).count('1')
        
        if trial_size > current_size:
            current = trial
            current_size = trial_size
    
    return current


def construct():
    start_time = time.time()
    deadline = start_time + 88  # Safety margin for 90s limit
    
    # 1. Cantor set seed
    cantor = _cantor_bits()
    best = cantor
    best_size = bin(cantor).count('1')
    
    # 2. Try forward greedy from scratch
    fwd = _greedy_forward()
    fwd_size = bin(fwd).count('1')
    if fwd_size > best_size:
        best = fwd
        best_size = fwd_size
    
    # 3. Local search on the best so far
    if time.time() < deadline - 40:
        improved = _local_search(best, deadline - 20)
        imp_size = bin(improved).count('1')
        if imp_size > best_size:
            best = improved
            best_size = imp_size
    
    # 4. Randomized restarts with kicks
    rng = random.Random(12345)
    restart_count = 0
    while time.time() < deadline - 25 and restart_count < 40:
        restart_count += 1
        
        # Different strategies
        r = rng.random()
        if r < 0.25:
            # Start from Cantor, add randomly
            bits = _greedy_random(cantor, rng)
        elif r < 0.5:
            # Pure random greedy
            bits = _greedy_random(0, rng)
        elif r < 0.75:
            # Start from best, perturb: remove 1-2 random elements, then re-greedy
            bits = best
            if best_size > 2:
                elems = _bits_to_set(bits)
                if elems:
                    k = rng.choice([1, 2])
                    to_remove = rng.sample(elems, min(k, len(elems)))
                    for rem in to_remove:
                        bits &= ~(1 << rem)
                    bits = _greedy_forward(bits, deadline - 10)
        else:
            # Kick search on current best
            if time.time() < deadline - 15:
                bits = _kick_search(best, deadline - 10, rng, max_kicks=5)
            else:
                bits = best
        
        size = bin(bits).count('1')
        if size > best_size:
            best = bits
            best_size = size
        
        # Local search on this result
        if time.time() < deadline - 15:
            improved = _local_search(bits, deadline - 10)
            imp_size = bin(improved).count('1')
            if imp_size > best_size:
                best = improved
                best_size = imp_size
        
        # Also try randomized local search
        if time.time() < deadline - 10:
            improved = _randomized_local_search(best, deadline - 5, rng)
            imp_size = bin(improved).count('1')
            if imp_size > best_size:
                best = improved
                best_size = imp_size
    
    # 5. Final local search on best
    if time.time() < deadline - 5:
        improved = _local_search(best, deadline)
        imp_size = bin(improved).count('1')
        if imp_size > best_size:
            best = improved
            best_size = imp_size
    
    # 6. Final randomized local search
    if time.time() < deadline - 2:
        improved = _randomized_local_search(best, deadline, rng)
        imp_size = bin(improved).count('1')
        if imp_size > best_size:
            best = improved
            best_size = imp_size
    
    # 7. Verify
    if not _is_valid(best):
        best = cantor
    
    return sorted(_bits_to_set(best))
>>>>>>> REPLACE

