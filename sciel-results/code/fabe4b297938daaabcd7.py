import time
import random
from itertools import combinations
from math import gcd

def construct():
    start_time = time.time()
    deadline = 80.0  # Safety margin
    
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute prime factors and exponents for each number
    # primes up to 128
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    
    # exponents[i] is a dict or list of (prime, exponent)
    # Actually, for speed, let's just store the exponents for all primes in a list
    # But most numbers have few prime factors.
    # Let's store a dictionary {prime: exponent} for each number.
    factorization = [{} for _ in range(N + 1)]
    for i in range(2, N + 1):
        x = i
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                exp = 0
                while x % p == 0:
                    x //= p
                    exp += 1
                factorization[i][p] = exp
        if x > 1:
            factorization[i][x] = 1
    # factorization[1] is empty dict
    
    # Precompute bad triples
    # A triple (a,b,c) is bad if there exists a prime p such that
    # the maximum exponent of p in a,b,c is unique to one of them.
    # i.e., max(v_p(a), v_p(b), v_p(c)) is achieved by exactly one of a,b,c.
    
    bad_triples = []
    # To speed up, we can group by primes?
    # For a fixed prime p, a triple is "bad due to p" if the max exponent is unique.
    # A triple is bad if it is bad due to ANY prime.
    
    # Iterating all triples is ~350k.
    # For each triple, checking all primes is ~25 checks.
    # Total ~9M ops. This is fine.
    
    for a, b, c in combinations(elements, 3):
        is_bad = False
        # Union of primes involved in a,b,c
        # We can just check all primes in the union of their factorizations
        # Or just iterate all primes up to 128.
        # Iterating all primes is simpler and fast enough.
        for p in primes:
            va = factorization[a].get(p, 0)
            vb = factorization[b].get(p, 0)
            vc = factorization[c].get(p, 0)
            
            # Check if max is unique
            max_exp = max(va, vb, vc)
            count_max = 0
            if va == max_exp: count_max += 1
            if vb == max_exp: count_max += 1
            if vc == max_exp: count_max += 1
            
            if count_max == 1:
                is_bad = True
                break
        
        if is_bad:
            bad_triples.append((a, b, c))
    
    # Heuristic: Remove Conflict
    # We want to find a large subset with no bad triples.
    # Start with all elements. Remove elements to break all bad triples.
    
    # Data structures for the search
    # active_elements: set of currently included elements
    # triple_active: set of indices of bad_triples that are still "active" (all 3 elements in active_elements)
    # element_triple_count: for each element, number of active triples it's part of
    
    def run_greedy_remove(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        active = set(elements)
        # Map element to set of triple indices it belongs to
        # And triple index to set of elements
        # Since we start with all elements, all triples are active initially.
        
        # Precompute adjacency for triples
        # elem_to_triples: dict {elem: set of triple_indices}
        elem_to_triples = {e: set() for e in elements}
        for idx, (a, b, c) in enumerate(bad_triples):
            elem_to_triples[a].add(idx)
            elem_to_triples[b].add(idx)
            elem_to_triples[c].add(idx)
        
        # active_triples: set of indices
        active_triples = set(range(len(bad_triples)))
        
        # counts
        counts = {e: len(elem_to_triples[e]) for e in elements}
        
        # Remove elements
        while active_triples:
            # Find element with max count among active elements
            # We need to ensure the element is in 'active'
            # Since we only remove elements, 'active' shrinks.
            # We can just iterate over active elements and find max count.
            # Optimization: Keep a list of (count, elem) and sort? Or just max().
            # max() over 128 elements is fast.
            
            best_elem = None
            best_count = -1
            # To break ties randomly, we can collect all maxes
            candidates = []
            for e in active:
                if counts[e] > best_count:
                    best_count = counts[e]
                    best_elem = e
                    candidates = [e]
                elif counts[e] == best_count and best_count > 0:
                    candidates.append(e)
            
            if not candidates:
                break
                
            # Pick randomly from candidates if we want variety, or just first
            # Let's pick randomly to explore different solutions
            elem_to_remove = random.choice(candidates)
            
            # Remove elem_to_remove
            active.remove(elem_to_remove)
            
            # Update active_triples and counts
            # Find all active triples containing elem_to_remove
            triples_to_remove = []
            for t_idx in elem_to_triples[elem_to_remove]:
                if t_idx in active_triples:
                    triples_to_remove.append(t_idx)
            
            for t_idx in triples_to_remove:
                active_triples.remove(t_idx)
                a, b, c = bad_triples[t_idx]
                # Decrement counts for a, b, c
                for e in (a, b, c):
                    if e in active: # Only update if still active?
                        # Actually, counts should reflect active triples.
                        # If e is removed, its count doesn't matter.
                        # But if e is active, its count decreases by 1.
                        counts[e] -= 1
                # Note: elem_to_remove is no longer in active, so we don't care about its count anymore.
                # But we should probably set it to 0 or ignore it.
            
        return active
    
    best_solution = set()
    best_size = 0
    
    # Run multiple restarts
    # Each run is fast.
    num_restarts = 200
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        sol = run_greedy_remove(random_seed=i)
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = sol
            
        # Early termination if we found a very large set?
        # Theoretical max is 128.
        # If best_size is high, we might stop.
        # But let's run all or until time.

    # Final check and return
    # Ensure non-empty
    if not best_solution:
        return [1]
        
    return list(best_solution)
