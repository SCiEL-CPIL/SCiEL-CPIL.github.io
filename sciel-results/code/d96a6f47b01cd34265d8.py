import time
import random

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
        This is equivalent to: (products * cand) intersects products.
        """
        # Using set comprehension and intersection is faster than Python loop
        # for large sets due to C implementation.
        scaled = {p * cand for p in products}
        return not (scaled & products)
    
    def try_greedy(order):
        """Try to build a valid set by greedily adding candidates in given order."""
        current_set = []
        products = {1}
        for cand in order:
            if is_valid_addition(cand, products):
                current_set.append(cand)
                # Update products
                products |= {p * cand for p in products}
        return current_set
    
    def local_search(current_set, remaining_candidates, time_limit):
        """Try to improve the set by adding elements and doing 1-for-1 swaps."""
        if not current_set:
            return current_set
        
        best = current_set[:]
        products = {1}
        for x in best:
            products |= {p * x for p in products}
        
        remaining = [c for c in remaining_candidates if c not in best]
        
        # Try to add any remaining candidate
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            new_remaining = []
            for cand in remaining:
                if time.time() > time_limit:
                    break
                if is_valid_addition(cand, products):
                    best.append(cand)
                    products |= {p * cand for p in products}
                    changed = True
                else:
                    new_remaining.append(cand)
            remaining = new_remaining
        
        # Try 1-for-1 swaps
        if time.time() < time_limit - 2:
            improved = True
            while improved and time.time() < time_limit:
                improved = False
                for i in range(len(best)):
                    if time.time() > time_limit:
                        break
                    removed = best[i]
                    temp_set = best[:i] + best[i+1:]
                    temp_products = {1}
                    for x in temp_set:
                        temp_products |= {p * x for p in temp_products}
                    
                    found = False
                    for cand in remaining:
                        if time.time() > time_limit:
                            break
                        if cand == removed:
                            continue
                        if is_valid_addition(cand, temp_products):
                            best = temp_set + [cand]
                            products = temp_products
                            products |= {p * cand for p in products}
                            remaining = [c for c in remaining if c != cand]
                            found = True
                            improved = True
                            break
                    
                    if not found:
                        # Re-add removed if we didn't swap
                        # Actually, if we didn't swap, best is unchanged, 
                        # but we computed temp. We should skip re-adding if no swap.
                        # The current code re-adds and updates products, which is redundant 
                        # if no swap occurred, but ensures state consistency.
                        # However, if no swap, best is same, products is same.
                        # We can just continue.
                        pass
        
        return best
    
    # Precompute omega for sorting
    def omega(n):
        count = 0
        d = 2
        temp = n
        while d * d <= temp:
            if temp % d == 0:
                count += 1
                while temp % d == 0:
                    temp //= d
            d += 1
        if temp > 1:
            count += 1
        return count

    omegas = {c: omega(c) for c in candidates}

    # Increase restarts, rely on time
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        
        if restarts % 5 == 0:
            order = candidates[:]
            random.shuffle(order)
        elif restarts % 5 == 1:
            # Sort by omega (fewer factors first)
            order = sorted(candidates, key=lambda x: (omegas[x], -x))
            random.shuffle(order[::2])
        elif restarts % 5 == 2:
            # Sort by value (larger first)
            order = sorted(candidates, reverse=True)
            random.shuffle(order[::3])
        elif restarts % 5 == 3:
            # Sort by value (smaller first)
            order = sorted(candidates)
            random.shuffle(order[::3])
        else:
            # Random weighted by inverse omega?
            order = candidates[:]
            random.shuffle(order)
        
        current_set = try_greedy(order)
        
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set[:]
            
            if time.time() < deadline - 5:
                improved = local_search(best_set, candidates, deadline - 5)
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
