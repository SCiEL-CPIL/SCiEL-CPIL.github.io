import time
import random
from itertools import product

def construct():
    """
    Find the largest subset of {2, ..., 40} such that all subset products are distinct.
    """
    deadline = time.time() + 45  # Safety margin
    best_set = []
    best_len = 0
    
    candidates = list(range(2, 41))
    
    # We can run multiple random restarts
    # The greedy check is O(2^k * k) roughly, which is fine for k up to 20.
    
    # To improve success rate, we can also try deterministic heuristics:
    # Sort by something? Random is usually good for this "subset product distinctness".
    
    num_restarts = 5000  # Can be adjusted based on speed
    
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
            
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        current_set = []
        products = {1}
        
        for cand in shuffled:
            # Check if adding cand keeps all products distinct
            # New products are {p * cand for p in products}
            # If any of these are already in 'products', there's a collision.
            # Also, since cand is distinct from existing elements, 
            # the only collisions can be between new products and old products.
            
            # Optimization: 
            # If cand is 1, it's a problem, but candidates are >= 2.
            
            # Check intersection
            # We can generate the new products and check.
            # To save time, we can break early if a collision is found.
            
            valid = True
            for p in products:
                if p * cand in products:
                    valid = False
                    break
            
            if valid:
                current_set.append(cand)
                # Update products
                # We need to add all p*cand
                # It's important to iterate over a copy or the original set
                # because we are adding to it
                # But wait, if we add to 'products' while iterating, it's bad.
                # We should generate the list of new products first.
                new_prods = [p * cand for p in products]
                products.update(new_prods)
                
                if len(current_set) > best_len:
                    best_len = len(current_set)
                    best_set = current_set[:]
                    
                    # If we reached 18, that's likely optimal (max possible given constraints? 
                    # Actually, can we go higher? 
                    # Max distinct primes is 12. 
                    # Theoretical max size? 
                    # With 12 dimensions, we can fit many vectors. 
                    # But the values are bounded by 40.
                    # Let's just keep searching if we have time.)
                    
    # If no set was found (unlikely), return a valid small set
    if not best_set:
        # Fallback: just primes
        return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        
    # Verify the best_set is valid before returning
    # This is a safety check in case the logic above had a bug
    # We re-check the distinctness of all 2^k subset products
    k = len(best_set)
    all_products = set()
    for mask in range(1 << k):
        prod = 1
        for i in range(k):
            if mask & (1 << i):
                prod *= best_set[i]
        if prod in all_products:
            # Collision found, fallback to primes
            return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        all_products.add(prod)
        
    return best_set
