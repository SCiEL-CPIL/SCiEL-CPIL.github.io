import time
import random
from math import gcd
from collections import defaultdict

def construct():
    deadline = time.time() + 55
    candidates = list(range(2, 41))
    
    # Precompute pair conflicts: two numbers conflict if they share a prime factor
    # OR if their ratio is a perfect power relationship that causes subset product collision
    # Actually, the real constraint is: no two subset products can be equal.
    # A necessary condition: no element can be a perfect power of another in the set,
    # but the full constraint is more complex. Let's use the direct product check.
    
    # Direct construction using primes (guaranteed distinct products)
    direct_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    best_set = direct_primes[:]
    best_size = len(best_set)
    
    # Fast validity check: maintain set of all subset products, check if adding x creates collision
    def is_valid_addition(current_set, x, current_products):
        """Check if adding x to current_set keeps all subset products distinct.
        current_products is the set of all 2^k subset products of current_set."""
        for p in current_products:
            if p * x in current_products:
                return False
        return True
    
    def add_to_products(current_products, x):
        """Return new set of subset products after adding x."""
        new_prods = {p * x for p in current_products}
        current_products.update(new_prods)
        return current_products
    
    # Precompute: for each candidate, which other candidates conflict with it
    # Two candidates a, b conflict if {1, a, b, a*b} has a collision, i.e., a==b (impossible) or a*b==1 (impossible)
    # Actually for just two elements, products are 1, a, b, ab. These are always distinct for distinct a,b >= 2.
    # The real conflicts arise with 3+ elements. So pair-level precomputation isn't straightforward.
    # Instead, let's focus on making the incremental check faster.
    
    random.seed(42)
    
    # Strategy 1: Randomized greedy with multiple restarts
    # Use incremental product set for O(2^k) check per addition
    for restart in range(5000):
        if time.time() > deadline - 1.0:
            break
        
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        current_set = []
        current_products = {1}
        
        for x in shuffled:
            if is_valid_addition(current_set, x, current_products):
                current_set.append(x)
                add_to_products(current_products, x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            if best_size >= 17:
                # Found a good solution, do intensive local search
                break
    
    # Strategy 2: Intensive local search on the best set found
    if time.time() < deadline - 3:
        local_best = best_set[:]
        local_size = best_size
        local_products = {1}
        for x in local_best:
            add_to_products(local_products, x)
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # Try adding each candidate not in set
            for x in candidates:
                if x in local_best:
                    continue
                if is_valid_addition(local_best, x, local_products):
                    local_best.append(x)
                    add_to_products(local_products, x)
                    improved = True
                    if len(local_best) > local_size:
                        local_size = len(local_best)
                        best_set = local_best[:]
                        best_size = local_size
                    break
            
            # Try 1-for-1 swaps
            if not improved:
                for i, old_x in enumerate(local_best[:]):
                    trial_set = [x for j, x in enumerate(local_best) if j != i]
                    trial_products = {1}
                    for elem in trial_set:
                        add_to_products(trial_products, elem)
                    
                    for new_x in candidates:
                        if new_x in trial_set:
                            continue
                        if is_valid_addition(trial_set, new_x, trial_products):
                            local_best = trial_set + [new_x]
                            local_products = {1}
                            for elem in local_best:
                                add_to_products(local_products, elem)
                            improved = True
                            if len(local_best) > local_size:
                                local_size = len(local_best)
                                best_set = local_best[:]
                                best_size = local_size
                            break
                    if improved:
                        break
            
            # Try 2-for-1 swaps (remove 2, add 1) - can only shrink, skip
            # Try 1-for-2 swaps (remove 1, add 2) - can grow
            if not improved:
                for i, old_x in enumerate(local_best[:]):
                    trial_set = [x for j, x in enumerate(local_best) if j != i]
                    trial_products = {1}
                    for elem in trial_set:
                        add_to_products(trial_products, elem)
                    
                    found = False
                    for new_x1 in candidates:
                        if new_x1 in trial_set or found:
                            continue
                        if is_valid_addition(trial_set, new_x1, trial_products):
                            trial_set2 = trial_set + [new_x1]
                            trial_products2 = set(trial_products)
                            add_to_products(trial_products2, new_x1)
                            for new_x2 in candidates:
                                if new_x2 in trial_set2 or found:
                                    continue
                                if is_valid_addition(trial_set2, new_x2, trial_products2):
                                    local_best = trial_set2 + [new_x2]
                                    local_products = {1}
                                    for elem in local_best:
                                        add_to_products(local_products, elem)
                                    improved = True
                                    found = True
                                    if len(local_best) > local_size:
                                        local_size = len(local_best)
                                        best_set = local_best[:]
                                        best_size = local_size
                                    break
                    if improved:
                        break
    
    # Strategy 3: Systematic greedy with different orderings
    if time.time() < deadline - 5:
        orderings = [
            sorted(candidates),
            sorted(candidates, reverse=True),
            sorted(candidates, key=lambda x: (len([d for d in range(2, x) if x % d == 0]), x)),
            sorted(candidates, key=lambda x: -x),
            list(range(40, 1, -1)),
        ]
        
        for ordering in orderings:
            if time.time() > deadline - 3:
                break
            current_set = []
            current_products = {1}
            for x in ordering:
                if is_valid_addition(current_set, x, current_products):
                    current_set.append(x)
                    add_to_products(current_products, x)
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
    
    # Strategy 4: More randomized restarts with remaining time
    start_time = time.time()
    max_time = max(0, deadline - time.time() - 1.0)
    while time.time() - start_time < max_time:
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        current_set = []
        current_products = {1}
        
        for x in shuffled:
            if is_valid_addition(current_set, x, current_products):
                current_set.append(x)
                add_to_products(current_products, x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
    
    return [int(x) for x in best_set]
