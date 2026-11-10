"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 40, 'maximum_size': 18}

def _get_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]

def _prime_factorization(n, primes):
    """Return dict of prime -> exponent for n."""
    factors = {}
    for p in primes:
        if p * p > n:
            break
        while n % p == 0:
            factors[p] = factors.get(p, 0) + 1
            n //= p
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _build_vectors(candidates, primes):
    """Precompute prime exponent vectors for all candidates."""
    n_primes = len(primes)
    vecs = {}
    for num in candidates:
        fac = _prime_factorization(num, primes)
        vec = [0] * n_primes
        for i, p in enumerate(primes):
            if p in fac:
                vec[i] = fac[p]
        vecs[num] = vec
    return vecs

def _add_element_check(existing_products, new_vec, n_primes):
    """Check if adding new_vec to existing set of product exponent tuples creates a collision.
    existing_products: set of tuples representing all current subset product exponent vectors.
    new_vec: list of prime exponents for the new element.
    Returns (True, new_products) if no collision, else (False, None).
    """
    # New products are: old_product * new_element for each old_product in existing_products
    # We need to check that none of these new products already exist in existing_products
    # (which would mean a collision between a subset not containing new element and one containing it)
    # AND that all new products are distinct from each other (they will be since multiplication by
    # a fixed non-zero vector is injective on exponent vectors).
    
    # Actually, the new products are {p + new_vec : p in existing_products}
    # Collision occurs if (p1 + new_vec) == p2 for some p1, p2 in existing_products
    # i.e., p1 + new_vec == p2, i.e., p2 - p1 == new_vec
    # Since all exponents are non-negative, this means new_vec must equal the difference of two existing product vectors.
    
    # More efficiently: for each p in existing_products, compute p + new_vec and check if it's in existing_products.
    # Also need to ensure all p + new_vec are distinct, which they are automatically.
    
    new_products = set()
    for p in existing_products:
        # Compute p + new_vec
        np = tuple(p[i] + new_vec[i] for i in range(n_primes))
        if np in existing_products:
            return (False, None)
        new_products.add(np)
    
    return (True, new_products)

def _full_subset_products(vecs_list, n_primes):
    """Compute all 2^k subset product exponent tuples from a list of vectors."""
    k = len(vecs_list)
    if k == 0:
        return {(0,) * n_primes}
    
    products = {(0,) * n_primes}
    for v in vecs_list:
        new_products = set()
        for p in products:
            np = tuple(p[i] + v[i] for i in range(n_primes))
            new_products.add(np)
        products |= new_products
    return products

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin within 60s limit
    
    n = P['n']
    max_size = P['maximum_size']
    
    primes = _get_primes(n)
    n_primes = len(primes)
    
    # Candidate numbers: 2..n
    candidates = list(range(2, n + 1))
    
    # Precompute vectors
    all_vecs = _build_vectors(candidates, primes)
    
    # Sort candidates: prefer primes (single prime factor), then prime powers, then others
    # Key: (number of distinct prime factors, total prime factors, value)
    def sort_key(num):
        fac = _prime_factorization(num, primes)
        distinct_pf = len(fac)
        total_pf = sum(fac.values())
        return (distinct_pf, total_pf, num)
    
    candidates_sorted = sorted(candidates, key=sort_key)
    
    def _time_left():
        return time.time() < deadline
    
    # Phase 1: Greedy construction
    selected = []
    existing_products = {(0,) * n_primes}
    
    for cand in candidates_sorted:
        if not _time_left():
            break
        if len(selected) >= max_size:
            break
        new_vec = all_vecs[cand]
        ok, new_prods = _add_element_check(existing_products, new_vec, n_primes)
        if ok:
            selected.append(cand)
            existing_products = existing_products | new_prods
    
    best_selected = list(selected)
    best_size = len(selected)
    
    # Phase 2: Try to improve via swaps
    # For each element in selected, try removing it and greedily adding others
    if best_size < max_size and _time_left():
        # Build a set of selected for quick lookup
        selected_set = set(selected)
        
        for i in range(len(best_selected)):
            if not _time_left():
                break
            if best_size >= max_size:
                break
            
            # Remove element i
            removed = best_selected[i]
            trial = best_selected[:i] + best_selected[i+1:]
            trial_set = set(trial)
            
            # Recompute existing products for trial
            trial_vecs = [all_vecs[x] for x in trial]
            trial_products = _full_subset_products(trial_vecs, n_primes)
            
            # Greedily add candidates not in trial
            for cand in candidates_sorted:
                if not _time_left():
                    break
                if len(trial) >= max_size:
                    break
                if cand in trial_set:
                    continue
                new_vec = all_vecs[cand]
                ok, new_prods = _add_element_check(trial_products, new_vec, n_primes)
                if ok:
                    trial.append(cand)
                    trial_set.add(cand)
                    trial_products = trial_products | new_prods
            
            if len(trial) > best_size:
                best_selected = trial
                best_size = len(trial)
    
    # Phase 3: More aggressive - try removing pairs of elements
    if best_size < max_size and _time_left():
        selected_set = set(best_selected)
        
        for i in range(len(best_selected)):
            if not _time_left():
                break
            if best_size >= max_size:
                break
            for j in range(i + 1, len(best_selected)):
                if not _time_left():
                    break
                if best_size >= max_size:
                    break
                
                trial = best_selected[:i] + best_selected[i+1:j] + best_selected[j+1:]
                trial_set = set(trial)
                
                trial_vecs = [all_vecs[x] for x in trial]
                trial_products = _full_subset_products(trial_vecs, n_primes)
                
                for cand in candidates_sorted:
                    if not _time_left():
                        break
                    if len(trial) >= max_size:
                        break
                    if cand in trial_set:
                        continue
                    new_vec = all_vecs[cand]
                    ok, new_prods = _add_element_check(trial_products, new_vec, n_primes)
                    if ok:
                        trial.append(cand)
                        trial_set.add(cand)
                        trial_products = trial_products | new_prods
                
                if len(trial) > best_size:
                    best_selected = trial
                    best_size = len(trial)
    
    # Phase 4: Final greedy fill attempt with current best
    if best_size < max_size and _time_left():
        selected_set = set(best_selected)
        best_vecs = [all_vecs[x] for x in best_selected]
        existing_products = _full_subset_products(best_vecs, n_primes)
        
        for cand in candidates_sorted:
            if not _time_left():
                break
            if len(best_selected) >= max_size:
                break
            if cand in selected_set:
                continue
            new_vec = all_vecs[cand]
            ok, new_prods = _add_element_check(existing_products, new_vec, n_primes)
            if ok:
                best_selected.append(cand)
                selected_set.add(cand)
                existing_products = existing_products | new_prods
    
    return list(best_selected)
# EVOLVE-BLOCK-END
