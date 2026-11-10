import time
import random
from math import gcd

def construct():
    """
    Find the largest subset of {2, ..., 40} such that all subset products are distinct.
    """
    deadline = time.time() + 45  # Safety margin
    best_set = []
    best_len = 0
    
    candidates = list(range(2, 41))
    
    def is_valid_addition(cand, products):
        """Check if adding cand keeps all products distinct.
        Collision exists iff there exist p1, p2 in products such that p1 * cand == p2.
        This is equivalent to: p2 % cand == 0 and p2 // cand in products.
        """
        for p2 in products:
            if p2 % cand == 0:
                if p2 // cand in products:
                    return False
        return True
    
    def try_greedy(order):
        """Try to build a valid set by greedily adding candidates in given order."""
        current_set = []
        products = {1}
        for cand in order:
            if is_valid_addition(cand, products):
                current_set.append(cand)
                new_prods = [p * cand for p in products]
                products.update(new_prods)
        return current_set
    
    def local_search(current_set, remaining_candidates, time_limit):
        """Try to improve the set by adding elements and doing 1-for-1 swaps."""
        if not current_set:
            return current_set
        
        best = current_set[:]
        products = {1}
        for x in best:
            new_prods = [p * x for p in products]
            products.update(new_prods)
        
        remaining = [c for c in remaining_candidates if c not in best]
        
        # Try to add any remaining candidate
        for cand in remaining:
            if time.time() > time_limit:
                break
            if is_valid_addition(cand, products):
                best.append(cand)
                new_prods = [p * cand for p in products]
                products.update(new_prods)
                remaining = [c for c in remaining if c != cand]
        
        # Try 1-for-1 swaps
        if time.time() < time_limit - 3:
            for i in range(len(best)):
                if time.time() > time_limit:
                    break
                removed = best[i]
                temp_set = best[:i] + best[i+1:]
                temp_products = {1}
                for x in temp_set:
                    new_prods = [p * x for p in temp_products]
                    temp_products.update(new_prods)
                
                found = False
                for cand in remaining:
                    if time.time() > time_limit:
                        break
                    if cand == removed:
                        continue
                    if is_valid_addition(cand, temp_products):
                        best = temp_set + [cand]
                        products = temp_products
                        new_prods = [p * cand for p in products]
                        products.update(new_prods)
                        remaining = [c for c in remaining if c != cand]
                        found = True
                        break
                
                if not found:
                    best = temp_set + [removed]
                    products = temp_products
                    new_prods = [p * removed for p in products]
                    products.update(new_prods)
        
        return best
    
    num_restarts = 4000
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        if r % 4 == 0:
            order = candidates[:]
            random.shuffle(order)
        elif r % 4 == 1:
            order = sorted(candidates, reverse=True)
            random.shuffle(order[::2])
        elif r % 4 == 2:
            order = sorted(candidates)
            random.shuffle(order[::2])
        else:
            # Sort by number of distinct prime factors (fewer = "purer")
            def omega(n):
                count = 0
                d = 2
                while d * d <= n:
                    if n % d == 0:
                        count += 1
                        while n % d == 0:
                            n //= d
                    d += 1
                if n > 1:
                    count += 1
                return count
            order = sorted(candidates, key=lambda x: (omega(x), -x))
            random.shuffle(order[::3])
        
        current_set = try_greedy(order)
        
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set[:]
            
            if time.time() < deadline - 8:
                improved = local_search(best_set, candidates, deadline - 8)
                if len(improved) > best_len:
                    best_len = len(improved)
                    best_set = improved
    
    if not best_set:
        return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        
    # Verify the best_set is valid before returning
    k = len(best_set)
    all_products = set()
    for mask in range(1 << k):
        prod = 1
        for i in range(k):
            if mask & (1 << i):
                prod *= best_set[i]
        if prod in all_products:
            return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        all_products.add(prod)
        
    return list(best_set)
