import time
import random
from itertools import combinations

def is_valid(vals):
    """
    Check if the sorted list of integers has distinct products for all non-empty consecutive blocks.
    """
    if not vals:
        return True
    
    products = set()
    n = len(vals)
    
    # Generate all consecutive block products
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= vals[j]
            if prod in products:
                return False
            products.add(prod)
            
    return True

def _all_products(vals):
    """Return set of all consecutive block products for a sorted list."""
    products = set()
    n = len(vals)
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= vals[j]
            products.add(prod)
    return products

def _can_add(vals, x):
    """Check if x can be added to vals while maintaining distinct products."""
    # x must be > all elements in vals for strict increase
    if vals and x <= vals[-1]:
        return False
    
    new_vals = vals + [x]
    products = _all_products(vals)
    n = len(vals)
    
    # Check all blocks ending at x
    prod = 1
    for i in range(n, -1, -1):
        prod *= new_vals[i]
        if prod in products:
            return False
        products.add(prod)
    
    return True

def _greedy_pass(start_val=2, end_val=129):
    """Greedy construction from start_val to end_val."""
    values, products = [], set()
    for x in range(start_val, end_val):
        if not values or x > values[-1]:
            tails = []
            prod = x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            tails.append(x)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
    return values

def _prime_power_candidates():
    """Generate candidate numbers that are prime powers (likely to have unique factorizations)."""
    candidates = set()
    # Include all primes up to 128
    for p in range(2, 129):
        is_prime = True
        for i in range(2, int(p**0.5) + 1):
            if p % i == 0:
                is_prime = False
                break
        if is_prime:
            candidates.add(p)
    
    # Include prime powers: p^k where p^k <= 128
    for p in range(2, 129):
        is_prime = True
        for i in range(2, int(p**0.5) + 1):
            if p % i == 0:
                is_prime = False
                break
        if is_prime:
            power = p
            while power <= 128:
                candidates.add(power)
                power *= p
    
    return sorted(candidates)

def _local_search(vals, deadline):
    """Local search: try to add missing numbers, then try swaps."""
    best_set = set(vals)
    best_len = len(vals)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add any missing number
        for x in range(2, 129):
            if x not in best_set:
                new_set = best_set | {x}
                if is_valid(sorted(new_set)):
                    best_set = new_set
                    best_len += 1
                    improved = True
                    break
        
        if improved:
            continue
            
        # Phase 2: Try swapping one element to enable adding another
        if time.time() > deadline - 5.0:
            break
            
        found_swap = False
        for x in list(best_set):
            if found_swap:
                break
            for y in range(2, 129):
                if y in best_set or y == x:
                    continue
                
                new_set = best_set - {x} | {y}
                if not is_valid(sorted(new_set)):
                    continue
                
                for z in range(2, 129):
                    if z in new_set:
                        continue
                    final_set = new_set | {z}
                    if is_valid(sorted(final_set)):
                        best_set = final_set
                        best_len += 1
                        improved = True
                        found_swap = True
                        break
            if found_swap:
                break
        
        # Phase 3: Try removing one element to enable adding two
        if not improved and time.time() < deadline - 10.0:
            for x in list(best_set):
                reduced_set = best_set - {x}
                if len(reduced_set) < best_len - 1:
                    continue
                    
                # Try to add two elements to reduced_set
                for y in range(2, 129):
                    if y in reduced_set:
                        continue
                    set_with_y = reduced_set | {y}
                    if not is_valid(sorted(set_with_y)):
                        continue
                    
                    for z in range(2, 129):
                        if z in set_with_y:
                            continue
                        final_set = set_with_y | {z}
                        if is_valid(sorted(final_set)) and len(final_set) > best_len:
                            best_set = final_set
                            best_len = len(final_set)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    
    return sorted(best_set)

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s margin
    
    # Strategy 1: Standard greedy
    values1 = _greedy_pass(2, 129)
    best = _local_search(values1, deadline)
    
    # Strategy 2: Prime-power focused greedy
    candidates = _prime_power_candidates()
    values2 = []
    products = set()
    for x in candidates:
        if not values2 or x > values2[-1]:
            tails = []
            prod = x
            for a in reversed(values2):
                prod *= a
                tails.append(prod)
            tails.append(x)
            if not products.intersection(tails):
                values2.append(x)
                products.update(tails)
    
    if len(values2) > len(best):
        best = _local_search(values2, deadline)
    
    # Strategy 3: Multi-start with random subsets
    random.seed(42)
    all_candidates = list(range(2, 129))
    
    for trial in range(20):
        if time.time() > deadline - 10.0:
            break
        
        # Random subset of candidates, sorted
        subset_size = random.randint(10, 20)
        random_subset = random.sample(all_candidates, subset_size)
        random_subset.sort()
        
        # Greedy from this subset
        values3 = []
        products = set()
        for x in random_subset:
            if not values3 or x > values3[-1]:
                tails = []
                prod = x
                for a in reversed(values3):
                    prod *= a
                    tails.append(prod)
                tails.append(x)
                if not products.intersection(tails):
                    values3.append(x)
                    products.update(tails)
        
        # Extend with greedy
        for x in range(2, 129):
            if x not in values3 and (not values3 or x > values3[-1]):
                tails = []
                prod = x
                for a in reversed(values3):
                    prod *= a
                    tails.append(prod)
                tails.append(x)
                if not products.intersection(tails):
                    values3.append(x)
                    products.update(tails)
        
        if len(values3) > len(best):
            best = _local_search(values3, deadline)
    
    return best
