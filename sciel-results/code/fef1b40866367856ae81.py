import time
import random
from itertools import combinations

def construct():
    """
    Find the largest subset of {2, ..., 40} such that all subset products are distinct.
    Uses prime exponent vectors for efficient collision checking.
    """
    deadline = time.time() + 50  # Safety margin (90s limit, keep 40s buffer)
    best_set = []
    best_len = 0
    
    candidates = list(range(2, 41))
    
    # Precompute prime exponent vectors for each candidate
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    prime_index = {p: i for i, p in enumerate(primes)}
    
    def factorize(n):
        """Return prime exponent vector as a tuple."""
        vec = [0] * len(primes)
        for i, p in enumerate(primes):
            while n % p == 0:
                vec[i] += 1
                n //= p
        return tuple(vec)
    
    # Precompute factorizations
    factorizations = {c: factorize(c) for c in candidates}
    
    def vec_add(a, b):
        """Add two prime exponent vectors."""
        return tuple(a[i] + b[i] for i in range(len(primes)))
    
    def is_valid_addition(cand, product_vecs):
        """
        Check if adding cand to the set with existing product vectors 'product_vecs' 
        results in all distinct products.
        Collision if v + vec(cand) == w for some v, w in product_vecs.
        This is equivalent to: for all v in product_vecs, v + vec(cand) is NOT in product_vecs.
        """
        cand_vec = factorizations[cand]
        for v in product_vecs:
            new_v = vec_add(v, cand_vec)
            if new_v in product_vecs:
                return False
        return True
    
    def get_product_vecs(lst):
        """Compute the set of all subset product exponent vectors for lst."""
        vecs = {tuple([0] * len(primes))}  # Empty product
        for x in lst:
            x_vec = factorizations[x]
            new_vecs = [vec_add(v, x_vec) for v in vecs]
            vecs.update(new_vecs)
        return vecs
    
    def try_greedy(order):
        """Build a set greedily based on the given order."""
        current_set = []
        product_vecs = {tuple([0] * len(primes))}
        for cand in order:
            if is_valid_addition(cand, product_vecs):
                current_set.append(cand)
                new_vecs = [vec_add(v, factorizations[cand]) for v in product_vecs]
                product_vecs.update(new_vecs)
        return current_set
    
    def improved_local_search(current_set, time_limit):
        """
        Local search: 
        1. Add if possible.
        2. If stuck, try single swaps (remove i, add j) that enable further additions.
        """
        best = current_set[:]
        product_vecs = get_product_vecs(best)
        remaining = [c for c in candidates if c not in best]
        
        while time.time() < time_limit:
            # Phase 1: Add if possible
            added = False
            for cand in remaining:
                if is_valid_addition(cand, product_vecs):
                    best.append(cand)
                    new_vecs = [vec_add(v, factorizations[cand]) for v in product_vecs]
                    product_vecs.update(new_vecs)
                    remaining.remove(cand)
                    added = True
                    break
            
            if added:
                continue  # Restart loop to try adding more
            
            # Phase 2: Try swaps
            swap_made = False
            for i in range(len(best)):
                if time.time() > time_limit:
                    break
                
                removed = best[i]
                temp_set = best[:i] + best[i+1:]
                temp_vecs = get_product_vecs(temp_set)
                temp_remaining = remaining + [removed]
                
                # Try to add any element from temp_remaining to temp_set
                for cand in temp_remaining:
                    if time.time() > time_limit:
                        break
                    if is_valid_addition(cand, temp_vecs):
                        # Found a valid addition after swap
                        new_set = temp_set + [cand]
                        best = new_set
                        product_vecs = temp_vecs
                        new_vecs = [vec_add(v, factorizations[cand]) for v in product_vecs]
                        product_vecs.update(new_vecs)
                        
                        # Recompute remaining
                        remaining = [c for c in candidates if c not in best]
                        
                        swap_made = True
                        break
                if swap_made:
                    break
            
            if not swap_made:
                break  # No improvements found
        
        return best
    
    def verify_solution(sol):
        """Verify that all subset products are distinct."""
        k = len(sol)
        all_products = set()
        for mask in range(1 << k):
            prod = 1
            for i in range(k):
                if mask & (1 << i):
                    prod *= sol[i]
            if prod in all_products:
                return False
            all_products.add(prod)
        return True
    
    # Strategy: Use multiple orderings and local search
    num_restarts = 500
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Generate diverse orderings
        if r % 5 == 0:
            order = candidates[:]
            random.shuffle(order)
        elif r % 5 == 1:
            # Primes first (they have the most "independent" vectors)
            primes_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
            composites = [c for c in candidates if c not in primes_list]
            random.shuffle(composites)
            order = primes_list + composites
        elif r % 5 == 2:
            # Sort by number of distinct prime factors (ascending)
            def num_distinct_pf(c):
                return sum(1 for e in factorizations[c] if e > 0)
            order = sorted(candidates, key=num_distinct_pf)
            random.shuffle(order)
        elif r % 5 == 3:
            # Sort by total prime factors (ascending)
            def total_pf(c):
                return sum(factorizations[c])
            order = sorted(candidates, key=total_pf)
            random.shuffle(order)
        else:
            # Largest first
            order = sorted(candidates, reverse=True)
            random.shuffle(order[::2])
        
        current_set = try_greedy(order)
        
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set[:]
            
            # Local search
            if time.time() < deadline - 5:
                improved = improved_local_search(best_set, deadline - 5)
                if len(improved) > best_len:
                    best_len = len(improved)
                    best_set = improved
    
    # Fallback
    if not best_set:
        return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        
    # Final verification
    if not verify_solution(best_set):
        return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        
    return best_set
