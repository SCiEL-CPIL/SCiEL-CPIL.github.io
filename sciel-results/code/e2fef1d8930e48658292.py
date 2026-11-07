import random
import time

def construct():
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    # Precompute exponent vectors
    vecs = {}
    for n in range(2, 41):
        v = [0] * num_primes
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
        vecs[n] = tuple(v)
        
    zero = (0,) * num_primes
    
    def check_collision(sums, v):
        """Check if adding v to any sum in sums causes a collision."""
        for s in sums:
            ns = tuple(s[i] + v[i] for i in range(num_primes))
            if ns in sums:
                return True
        return False
    
    def greedy_build(candidates, order):
        """Greedy build using a specific ordering of candidates."""
        sums = {zero}
        current_set = []
        for idx in order:
            n = candidates[idx]
            v = vecs[n]
            if not check_collision(sums, v):
                new_sums = [tuple(s[i] + v[i] for i in range(num_primes)) for s in sums]
                sums.update(new_sums)
                current_set.append(n)
        return current_set
    
    # Initial good set: primes
    initial_set = primes[:]
    best_set = initial_set[:]
    best_len = len(initial_set)
    
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # Strategy 1: Greedy with different orderings
    # Sort by number of prime factors (harder numbers first)
    candidates = list(range(2, 41))
    prime_factor_count = {}
    for n in range(2, 41):
        count = 0
        temp = n
        for p in primes:
            while temp % p == 0:
                count += 1
                temp //= p
        prime_factor_count[n] = count
    
    # Ordering 1: Most composite first
    order_composite = sorted(range(39), key=lambda i: -prime_factor_count[candidates[i]])
    result = greedy_build(candidates, order_composite)
    if len(result) > best_len:
        best_len = len(result)
        best_set = result[:]
    
    # Ordering 2: Primes first, then by value
    order_primes_first = sorted(range(39), key=lambda i: (prime_factor_count[candidates[i]] == 0, candidates[i]))
    result = greedy_build(candidates, order_primes_first)
    if len(result) > best_len:
        best_len = len(result)
        best_set = result[:]
    
    # Strategy 2: Random restarts with local search
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Random permutation
        order = list(range(39))
        rng.shuffle(order)
        
        # Greedy build
        sums = {zero}
        current_set = []
        in_set = [False] * 39
        for idx in order:
            n = candidates[idx]
            v = vecs[n]
            if not check_collision(sums, v):
                new_sums = [tuple(s[i] + v[i] for i in range(num_primes)) for s in sums]
                sums.update(new_sums)
                current_set.append(n)
                in_set[idx] = True
        
        current_len = len(current_set)
        
        # Local search: try swaps
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing one and adding another
            for remove_idx in range(len(current_set)):
                n_remove = current_set[remove_idx]
                remove_candidate_idx = candidates.index(n_remove)
                
                # Try adding each candidate not in set
                for add_idx in range(39):
                    if in_set[add_idx]:
                        continue
                    n_add = candidates[add_idx]
                    
                    # Check if we can swap
                    # Remove n_remove, then try to add n_add
                    # This is expensive, so do a quick check first
                    
                    # Build sums without n_remove
                    sums_without = {zero}
                    valid = True
                    for n in current_set:
                        if n == n_remove:
                            continue
                        v = vecs[n]
                        new_sums = []
                        for s in sums_without:
                            ns = tuple(s[i] + v[i] for i in range(num_primes))
                            if ns in sums_without:
                                valid = False
                                break
                            new_sums.append(ns)
                        if not valid:
                            break
                        sums_without.update(new_sums)
                    
                    if not valid:
                        continue
                    
                    # Now try to add n_add
                    v_add = vecs[n_add]
                    can_add = True
                    for s in sums_without:
                        ns = tuple(s[i] + v_add[i] for i in range(num_primes))
                        if ns in sums_without:
                            can_add = False
                            break
                    
                    if can_add:
                        # Perform the swap
                        current_set[remove_idx] = n_add
                        in_set[remove_candidate_idx] = False
                        in_set[add_idx] = True
                        improved = True
                        break
                if improved:
                    break
        
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set[:]
        
        # If we found a perfect 18, stop early
        if best_len >= 18:
            break
    
    return best_set
