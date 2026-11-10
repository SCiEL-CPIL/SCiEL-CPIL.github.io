"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

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
    new_products = set()
    for p in existing_products:
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

def _greedy_select(candidates_sorted, all_vecs, n_primes, max_size, deadline, existing_products=None, existing_selected=None):
    """Greedy selection with incremental collision checking."""
    if existing_products is None:
        existing_products = {(0,) * n_primes}
    if existing_selected is None:
        existing_selected = []
    
    selected = list(existing_selected)
    products = set(existing_products)
    
    for cand in candidates_sorted:
        if time.time() > deadline:
            break
        if len(selected) >= max_size:
            break
        if cand in selected:
            continue
        new_vec = all_vecs[cand]
        ok, new_prods = _add_element_check(products, new_vec, n_primes)
        if ok:
            selected.append(cand)
            products = products | new_prods
    return selected, products

def construct():
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
    def sort_key(num):
        fac = _prime_factorization(num, primes)
        distinct_pf = len(fac)
        total_pf = sum(fac.values())
        return (distinct_pf, total_pf, num)
    
    candidates_sorted = sorted(candidates, key=sort_key)
    
    def _time_left():
        return time.time() < deadline
    
    # Phase 1: Initial greedy construction
    best_selected, _ = _greedy_select(candidates_sorted, all_vecs, n_primes, max_size, deadline)
    best_size = len(best_selected)
    
    # Phase 2: Multiple restarts with different orderings to escape local optima
    # Try several different sorting strategies
    if best_size < max_size and _time_left():
        # Strategy 1: Sort by value (ascending)
        candidates_by_value = sorted(candidates, key=lambda x: x)
        sel1, _ = _greedy_select(candidates_by_value, all_vecs, n_primes, max_size, deadline)
        if len(sel1) > best_size:
            best_selected = sel1
            best_size = len(sel1)
        
        # Strategy 2: Sort by value (descending) - large numbers first
        candidates_by_value_desc = sorted(candidates, key=lambda x: -x)
        sel2, _ = _greedy_select(candidates_by_value_desc, all_vecs, n_primes, max_size, deadline)
        if len(sel2) > best_size:
            best_selected = sel2
            best_size = len(sel2)
        
        # Strategy 3: Randomized restarts
        for _restart in range(50):
            if not _time_left():
                break
            if best_size >= max_size:
                break
            
            # Shuffle candidates with bias toward "easy" numbers
            shuffled = list(candidates)
            random.shuffle(shuffled)
            # But keep primes at the front with some probability
            prime_candidates = [c for c in shuffled if _prime_factorization(c, primes) and len(_prime_factorization(c, primes)) == 1]
            composite_candidates = [c for c in shuffled if c not in prime_candidates]
            
            # Mix: some primes first, then shuffled rest
            mixed = prime_candidates[:min(3, len(prime_candidates))] + composite_candidates
            random.shuffle(mixed)
            
            sel_r, _ = _greedy_select(mixed, all_vecs, n_primes, max_size, deadline)
            if len(sel_r) > best_size:
                best_selected = sel_r
                best_size = len(sel_r)
    
    # Phase 3: Local search - try swapping elements to improve
    if best_size < max_size and _time_left():
        selected_set = set(best_selected)
        
        # Try removing each element and greedily adding others
        for i in range(len(best_selected)):
            if not _time_left():
                break
            if best_size >= max_size:
                break
            
            removed = best_selected[i]
            trial = best_selected[:i] + best_selected[i+1:]
            trial_set = set(trial)
            
            trial_vecs = [all_vecs[x] for x in trial]
            trial_products = _full_subset_products(trial_vecs, n_primes)
            
            # Greedily add candidates not in trial
            trial, _ = _greedy_select(candidates_sorted, all_vecs, n_primes, max_size, deadline, trial_products, trial)
            
            if len(trial) > best_size:
                best_selected = trial
                best_size = len(trial)
                selected_set = set(best_selected)
    
    # Phase 4: More aggressive - try removing pairs of elements
    if best_size < max_size and _time_left():
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
                
                trial_vecs = [all_vecs[x] for x in trial]
                trial_products = _full_subset_products(trial_vecs, n_primes)
                
                trial, _ = _greedy_select(candidates_sorted, all_vecs, n_primes, max_size, deadline, trial_products, trial)
                
                if len(trial) > best_size:
                    best_selected = trial
                    best_size = len(trial)
    
    # Final validation: ensure all subset products are truly distinct
    # This is a safety check
    if len(best_selected) > 0:
        vecs_list = [all_vecs[x] for x in best_selected]
        all_products = _full_subset_products(vecs_list, n_primes)
        if len(all_products) != (1 << len(best_selected)):
            # Collision detected - this shouldn't happen with our incremental check
            # but if it does, fall back to a simpler valid set
            best_selected = [c for c in candidates_sorted if c in set(best_selected)][:best_size]
    
    return list(best_selected)
# EVOLVE-BLOCK-END
