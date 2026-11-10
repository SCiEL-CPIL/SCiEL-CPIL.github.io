import time
import random
from itertools import combinations

def construct():
    """
    Find the largest subset of {2, ..., 40} such that all subset products are distinct.
    """
    deadline = time.time() + 45  # Safety margin
    best_set = []
    best_len = 0
    
    candidates = list(range(2, 41))
    
    def is_valid_addition(cand, products):
        """
        Check if adding cand to the set with existing products 'products' 
        results in all distinct products.
        Collision if p * cand == q for some p, q in products.
        This is equivalent to: for all p in products, p * cand is NOT in products.
        """
        for p in products:
            if p * cand in products:
                return False
        return True

    def get_products(lst):
        """Compute the set of all subset products for lst."""
        prods = {1}
        for x in lst:
            new_prods = [p * x for p in prods]
            prods.update(new_prods)
        return prods

    def try_greedy(order):
        """Build a set greedily based on the given order."""
        current_set = []
        products = {1}
        for cand in order:
            if is_valid_addition(cand, products):
                current_set.append(cand)
                new_prods = [p * cand for p in products]
                products.update(new_prods)
        return current_set

    def local_search(current_set, remaining, time_limit):
        """
        Attempt to improve the set by adding elements or swapping.
        Returns the improved set.
        """
        best = current_set[:]
        products = get_products(best)
        remaining = [c for c in remaining if c not in best]
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1. Try to add any remaining element
            for cand in remaining:
                if time.time() > time_limit:
                    break
                if is_valid_addition(cand, products):
                    best.append(cand)
                    new_prods = [p * cand for p in products]
                    products.update(new_prods)
                    remaining.remove(cand)
                    improved = True
                    break  # Restart the loop to check again
            
            # 2. If no addition possible, try swaps (remove one, add one) 
            #    to potentially enable a future addition.
            #    This is heuristic: we look for a swap that might lead to a better state.
            #    Since checking "leads to better state" is hard, we just do random kicks.
            if not improved and len(best) > 0:
                # Try to remove a random element and re-greedy fill
                # This is a "kick"
                if time.time() < time_limit - 1:
                    idx = random.randint(0, len(best) - 1)
                    removed = best[idx]
                    new_best = best[:idx] + best[idx+1:]
                    new_remaining = remaining + [removed]
                    
                    # Greedy fill from new_remaining
                    # We need to be careful not to re-add elements that were in best but not removed?
                    # Actually, we just try to build a new set from scratch using the "kick" logic?
                    # No, let's just try to add from new_remaining to the products of new_best.
                    
                    new_products = get_products(new_best)
                    added_any = False
                    for cand in new_remaining:
                        if time.time() > time_limit:
                            break
                        if is_valid_addition(cand, new_products):
                            new_best.append(cand)
                            new_prods = [p * cand for p in new_products]
                            new_products.update(new_prods)
                            added_any = True
                            # If we added something, we might be able to add more, 
                            # but let's just accept this improvement if size is same or larger.
                            # Actually, if we removed one and added one, size is same.
                            # If we removed one and added two, size increases.
                            # Let's just run a full greedy fill on new_remaining after the kick.
                            break
                    
                    # If the new set is better (larger) or equal but different (might allow further growth),
                    # we might switch to it. 
                    # However, a simple kick often just gets stuck in the same local optimum.
                    # A better kick: Remove 1, then try to add 2.
                    
                    # Let's try a specific swap: Remove x, try to add y.
                    # If we can add y, and then z, we win.
                    # This is expensive.
                    
                    pass # The above kick logic is a bit messy. Let's stick to a simpler, effective local search.
                    
        return best

    # Refined Local Search: 
    # 1. Add if possible.
    # 2. If stuck, try all single swaps (remove i, add j). 
    #    If a swap results in a set that allows an addition, take it.
    
    def improved_local_search(current_set, time_limit):
        best = current_set[:]
        products = get_products(best)
        remaining = [c for c in candidates if c not in best]
        
        while time.time() < time_limit:
            # Phase 1: Add if possible
            added = False
            for cand in remaining:
                if is_valid_addition(cand, products):
                    best.append(cand)
                    new_prods = [p * cand for p in products]
                    products.update(new_prods)
                    remaining.remove(cand)
                    added = True
                    break
            
            if added:
                continue # Restart loop to try adding more
            
            # Phase 2: Try swaps
            # For each element in best, try to remove it and add any remaining element.
            # If the new set allows an addition (i.e., is not "maximal"), we might want to move there.
            # But how to check "allows addition" quickly? 
            # We can just perform the swap and then try to add.
            
            swap_made = False
            for i in range(len(best)):
                if time.time() > time_limit:
                    break
                
                removed = best[i]
                temp_set = best[:i] + best[i+1:]
                temp_products = get_products(temp_set)
                temp_remaining = remaining + [removed]
                
                # Try to add any element from temp_remaining to temp_set
                for cand in temp_remaining:
                    if time.time() > time_limit:
                        break
                    if is_valid_addition(cand, temp_products):
                        # We found a valid addition after removing 'removed' and adding 'cand'.
                        # This implies the new set temp_set + [cand] has size len(best).
                        # Is it better? It might allow further additions.
                        # Let's accept this state and restart the search from here.
                        new_set = temp_set + [cand]
                        best = new_set
                        products = temp_products
                        new_prods = [p * cand for p in products]
                        products.update(new_prods)
                        
                        # Update remaining
                        # 'removed' is no longer in best. 'cand' is now in best.
                        # Other elements in temp_remaining that were not added are still remaining.
                        # This bookkeeping is tricky. 
                        # Easier: just recompute remaining from scratch? O(N).
                        remaining = [c for c in candidates if c not in best]
                        
                        swap_made = True
                        break
                if swap_made:
                    break
            
            if not swap_made:
                break # No improvements found
        
        return best

    num_restarts = 1000
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Generate order
        if r % 4 == 0:
            order = candidates[:]
            random.shuffle(order)
        elif r % 4 == 1:
            # Primes first
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
            composites = [c for c in candidates if c not in primes]
            random.shuffle(composites)
            order = primes + composites
        elif r % 4 == 2:
            # Largest first
            order = sorted(candidates, reverse=True)
            random.shuffle(order[::2])
        else:
            # Smallest first
            order = sorted(candidates)
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
        
    # Verification
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
        
    return best_set
