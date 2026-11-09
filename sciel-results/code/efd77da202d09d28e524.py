import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    candidates = list(range(2, 41))
    
    # Heuristic ordering: Primes first, then others.
    # This increases the chance of finding a valid set early.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    non_primes = [x for x in candidates if x not in primes]
    ordered_candidates = primes + non_primes
    
    def search_local(k_target, current_set, products, start_idx):
        if len(current_set) == k_target:
            return current_set
        if len(current_set) + (len(ordered_candidates) - start_idx) < k_target:
            return None
        
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return "TIMEOUT"
        
        for i in range(start_idx, len(ordered_candidates)):
            c = ordered_candidates[i]
            
            # Optimization: If c is in current_set, skip (not needed due to index logic)
            
            new_products = [p * c for p in products]
            
            # Check collision
            # products is a set
            # If any new product is in products, it's a collision
            # Note: new products are distinct among themselves because products are distinct and c != 0
            if any(np in products for np in new_products):
                continue
            
            new_current_set = current_set + [c]
            new_products_set = products.union(new_products)
            
            result = search_local(k_target, new_current_set, new_products_set, i + 1)
            if result is not None:
                if result == "TIMEOUT":
                    return "TIMEOUT"
                return result
        
        return None

    # Try 18, then 17
    for k in [18, 17]:
        result = search_local(k, [], {1}, 0)
        if result is not None:
            if result == "TIMEOUT":
                break
            # Verify
            res = [int(x) for x in result]
            if len(res) == k and all(2 <= x <= 40 for x in res) and len(set(res)) == k:
                # Quick verify distinct subset products
                prods = set()
                valid = True
                for r in range(k + 1):
                    for comb in itertools.combinations(res, r):
                        prod = 1
                        for x in comb:
                            prod *= x
                        if prod in prods:
                            valid = False
                            break
                        prods.add(prod)
                    if not valid:
                        break
                if valid:
                    return res

    # Fallback
    return [8, 16, 32, 9, 27, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
