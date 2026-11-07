import time
import random

# EVOLVE-BLOCK-START

def construct():
    """
    Returns a list of integers in 2..40 with distinct subset products.
    Uses randomized greedy search to maximize the size of the set.
    """
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    pool = list(range(2, 41))
    best_set = []
    
    # Helper to check if x can be added to current set with current products
    def can_add(x, products):
        # Check if p * x is in products for any p in products
        # If yes, collision.
        # Note: products is a set of integers.
        for p in products:
            if p * x in products:
                return False
        return True

    # Run multiple randomized greedy passes
    num_passes = 0
    while time.time() < deadline:
        random.shuffle(pool)
        current_set = []
        current_products = {1}
        
        for x in pool:
            if can_add(x, current_products):
                current_set.append(x)
                # Update products
                # new_products = {p * x for p in current_products}
                # current_products = current_products.union(new_products)
                # Optimization: only add new ones
                new_prods = []
                for p in current_products:
                    np_ = p * x
                    if np_ not in current_products:
                        new_prods.append(np_)
                current_products.update(new_prods)
        
        if len(current_set) > len(best_set):
            best_set = current_set.copy()
            if len(best_set) == 18: # Max possible
                break
        num_passes += 1

    # If we haven't found anything better than a known good set, return the known good set
    # The known good set from the prompt has 16 items.
    # If best_set is empty or smaller, we should probably return the hardcoded 16 to be safe,
    # but the search should find at least 16.
    if len(best_set) < 16:
        # Fallback to hardcoded 16
        return [
            2, 4, 16,
            3, 9,
            5, 25,
            7,
            11,
            13,
            17,
            19,
            23,
            29,
            31,
            37
        ]
    
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
