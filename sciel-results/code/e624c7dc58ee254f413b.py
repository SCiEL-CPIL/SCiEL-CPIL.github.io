import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    candidates = list(range(2, 41))
    
    def is_valid_addition(selected, products, c):
        """Check if c can be added to selected with given products."""
        for p in products:
            if p * c in products:
                return False, None
        new_products = {p * c for p in products}
        return True, new_products
    
    def compute_products(selected):
        products = {1}
        for c in selected:
            products = products | {p * c for p in products}
        return products
    
    def greedy(order):
        selected = []
        products = {1}
        for c in order:
            if time.time() > deadline:
                break
            valid, new_prods = is_valid_addition(selected, products, c)
            if valid:
                selected.append(c)
                products = products | new_prods
        return selected, products
    
    # Try multiple orderings
    best_selected = []
    best_products = set()
    
    orders = [
        candidates,  # natural
        [2,3,5,7,11,13,17,19,23,29,31,37] + [c for c in candidates if c not in [2,3,5,7,11,13,17,19,23,29,31,37]],  # primes first
        sorted(candidates, key=lambda x: (x.bit_count(), x)),  # fewer bits first
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        sel, prods = greedy(order)
        if len(sel) > len(best_selected):
            best_selected = sel
            best_products = prods
    
    selected = list(best_selected)
    products = best_products
    
    # Local search: try to add more elements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for c in candidates:
            if c in selected:
                continue
            if time.time() > deadline:
                break
            
            # Try direct addition
            valid, new_prods = is_valid_addition(selected, products, c)
            if valid:
                selected.append(c)
                products = products | new_prods
                improved = True
                continue
            
            # Try swapping: remove one, add c
            for old in selected:
                if time.time() > deadline:
                    break
                trial_selected = [x for x in selected if x != old]
                trial_products = compute_products(trial_selected)
                valid2, new_prods2 = is_valid_addition(trial_selected, trial_products, c)
                if valid2:
                    selected = trial_selected + [c]
                    products = trial_products | new_prods2
                    improved = True
                    break
    
    return [int(x) for x in selected]
