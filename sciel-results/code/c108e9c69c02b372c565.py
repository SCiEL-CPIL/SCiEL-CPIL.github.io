import time
import random
from math import gcd
from collections import Counter

def construct():
    deadline = time.time() + 55
    candidates = list(range(2, 41))
    
    # Precompute prime factorizations for all candidates
    def factorize(n):
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors
    
    prime_factors = {n: factorize(n) for n in candidates}
    
    # Direct construction using primes as initial best
    direct_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    best_set = direct_primes[:]
    best_size = len(best_set)
    
    def is_dissociated(products_set, candidate):
        """Check if adding candidate keeps all subset products distinct."""
        for p in products_set:
            if p * candidate in products_set:
                return False
        return True
    
    def backtrack(index, current_set, current_products, remaining_candidates):
        """Backtracking search with forward checking."""
        nonlocal best_set, best_size, deadline
        
        if time.time() > deadline - 1:
            return
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
        
        if index >= len(remaining_candidates):
            return
        
        # Try including remaining_candidates[index]
        cand = remaining_candidates[index]
        if is_dissociated(current_products, cand):
            new_products = current_products | {p * cand for p in current_products}
            backtrack(index + 1, current_set + [cand], new_products, remaining_candidates)
        
        # Try excluding remaining_candidates[index]
        backtrack(index + 1, current_set, current_products, remaining_candidates)
    
    # Phase 1: Systematic backtracking with different orderings
    orderings = [
        sorted(candidates),  # increasing
        sorted(candidates, reverse=True),  # decreasing
        sorted(candidates, key=lambda x: len(prime_factors[x])),  # fewer prime factors first
        sorted(candidates, key=lambda x: -len(prime_factors[x])),  # more prime factors first
        sorted(candidates, key=lambda x: max(prime_factors[x].values(), default=0)),  # highest exponent
        sorted(candidates, key=lambda x: x % 2),  # evens first
        sorted(candidates, key=lambda x: x % 3),  # multiples of 3 first
    ]
    
    for ordering in orderings:
        if time.time() > deadline - 3:
            break
        backtrack(0, [], {1}, ordering)
    
    # Phase 2: Randomized restarts with backtracking
    random.seed(42)
    start_phase2 = time.time()
    max_restarts = 200
    
    for restart in range(max_restarts):
        if time.time() > deadline - 2:
            break
        
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        # Use a limited depth backtracking to find good sets
        # Try a greedy approach first, then refine
        current_set = []
        current_products = {1}
        
        for x in shuffled:
            if is_dissociated(current_products, x):
                current_set.append(x)
                current_products.update({p * x for p in current_products})
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
    
    # Phase 3: Local search - try to improve best_set by swapping
    if time.time() < deadline - 3:
        current_set = best_set[:]
        current_products = {1}
        for x in current_set:
            current_products.update({p * x for p in current_products})
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # Try removing one element and adding two others
            for i in range(len(current_set)):
                if time.time() > deadline - 2:
                    break
                
                removed = current_set[i]
                remaining = [x for j, x in enumerate(current_set) if j != i]
                
                # Recompute products without removed
                rem_products = {1}
                for x in remaining:
                    rem_products.update({p * x for p in rem_products})
                
                # Try adding two candidates not in remaining
                added_any = False
                for a in candidates:
                    if a in remaining or a == removed:
                        continue
                    if is_dissociated(rem_products, a):
                        new_products_a = rem_products | {p * a for p in rem_products}
                        
                        for b in candidates:
                            if b in remaining or b == removed or b == a:
                                continue
                            if is_dissociated(new_products_a, b):
                                new_set = remaining + [a, b]
                                new_size = len(new_set)
                                
                                if new_size > len(current_set):
                                    current_set = new_set
                                    current_products = new_products_a | {p * b for p in new_products_a}
                                    improved = True
                                    added_any = True
                                    break
                        
                        if added_any:
                            break
                
                if improved:
                    break
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
    
    # Phase 4: Final targeted search - try to add elements to best_set
    if time.time() < deadline - 1:
        current_set = best_set[:]
        current_products = {1}
        for x in current_set:
            current_products.update({p * x for p in current_products})
        
        for x in candidates:
            if x in current_set:
                continue
            if is_dissociated(current_products, x):
                current_set.append(x)
                current_products.update({p * x for p in current_products})
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
    
    return [int(x) for x in best_set]
