"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 40, 'maximum_size': 18}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 90s kill
    
    # Precompute valid candidates: integers from 2 to 40
    candidates = list(range(2, P['n'] + 1))
    
    best_size = 0
    best_set = []
    
    # We want to maximize the size of the set.
    # A randomized greedy approach:
    # 1. Shuffle candidates
    # 2. Iterate through shuffled candidates
    # 3. For each candidate, check if adding it to the current set preserves distinct subset products
    # 4. If yes, add it. If no, skip it.
    
    # To improve the result, we can try multiple random shuffles and keep the best.
    # Also, we can try different ordering heuristics (e.g., larger numbers first, primes first).
    
    def is_valid_subset_products(elements):
        """
        Check if all 2^k subset products of 'elements' are distinct.
        elements is a list of integers.
        Returns True if valid, False otherwise.
        """
        if not elements:
            return True
        
        # Use a set to track products.
        # Start with product 1 (empty subset).
        products = {1}
        
        for x in elements:
            new_products = set()
            for p in products:
                new_p = p * x
                if new_p in products or new_p in new_products:
                    return False
                new_products.add(new_p)
            products.update(new_products)
            
        return True

    # Helper to check if adding 'x' to 'current_set' is valid
    # This can be optimized by checking collisions with existing subset products
    def can_add(current_products, x):
        """
        current_products: set of all subset products of the current set
        x: new element to add
        Returns True if adding x keeps all subset products distinct.
        """
        # The new subset products will be the old ones and {p * x for p in current_products}
        # We need to ensure:
        # 1. No p * x is in current_products (unless p=1 and x is already in set, but x is new)
        #    Actually, if x is not in the set, p*x cannot be in current_products for p>=1?
        #    Wait, if current set is {2, 3}, products are {1, 2, 3, 6}.
        #    If we add 2, it's not distinct. But we assume x is not in the set.
        #    If we add 6, 6 is in products. So 1*6=6 collides with 6.
        # 2. No two new products p1*x and p2*x are equal (impossible if p1!=p2 and x!=0)
        # 3. No new product p*x collides with another new product q*x (same as 2)
        # So the only check needed is: for any p in current_products, p*x is NOT in current_products.
        # And p*x is NOT equal to q*x for any p,q (which is always true for distinct p,q).
        # Wait, is that sufficient?
        # Let S be the set of subset products of current set.
        # New set of products S' = S U {s*x for s in S}.
        # We need S' to have distinct elements.
        # This requires:
        # a) S and {s*x for s in S} are disjoint.
        # b) {s*x for s in S} has distinct elements (true if x != 0 and S has distinct elements).
        # So we just need to check if any s*x is in S.
        
        for s in current_products:
            if (s * x) in current_products:
                return False
        return True

    # We will maintain the set of subset products incrementally.
    
    # Try several random strategies
    # Strategy 1: Random shuffle
    # Strategy 2: Sort by value (ascending)
    # Strategy 3: Sort by value (descending)
    # Strategy 4: Primes first, then others
    
    for trial in range(20):
        if time.time() > deadline:
            break
            
        if trial == 0:
            order = sorted(candidates)
        elif trial == 1:
            order = sorted(candidates, reverse=True)
        elif trial == 2:
            # Primes first
            def is_prime(n):
                if n < 2: return False
                for i in range(2, int(n**0.5)+1):
                    if n % i == 0: return False
                return True
            primes = [x for x in candidates if is_prime(x)]
            others = [x for x in candidates if not is_prime(x)]
            order = primes + others
        else:
            order = candidates[:]
            random.shuffle(order)
            
        current_set = []
        current_products = {1}
        
        for x in order:
            if time.time() > deadline:
                break
                
            if can_add(current_products, x):
                # Add x
                current_set.append(x)
                # Update products
                # New products are {p * x for p in current_products}
                # We must add them to current_products
                new_prods = [p * x for p in current_products]
                # Since can_add passed, these are all distinct from each other and from current_products
                current_products.update(new_prods)
                
        # End of trial
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
        # If we found a very large set, we might stop early?
        # Max possible is 18. If we hit 18, we can return immediately.
        if best_size >= P['maximum_size']:
            break

    # Ensure return type is list of ints
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
