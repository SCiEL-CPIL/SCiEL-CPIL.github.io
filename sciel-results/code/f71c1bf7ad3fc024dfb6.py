import time
import random
from math import gcd
from collections import defaultdict

def construct():
    deadline = time.time() + 55
    candidates = list(range(2, 41))
    
    # Direct construction with primes as initial best
    direct_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    best_set = direct_primes[:]
    best_size = len(best_set)
    
    # Fast validity check using incremental product tracking
    # A set S is valid iff all 2^|S| subset products are distinct
    # When adding x to S: collision iff x * p == q for some p, q in products(S)
    # i.e., x is in {q/p : p,q in products(S), p divides q}
    
    def is_valid_with_products(current_products, x):
        """Check if x can be added given the current set of subset products."""
        for p in current_products:
            if p * x in current_products:
                return False
        return True
    
    def add_to_products(current_products, x):
        """Add x to the set, returning new product set."""
        new_prods = {p * x for p in current_products}
        current_products.update(new_prods)
        return current_products
    
    # Backtracking search with forward checking
    # State: (current_set, current_products, remaining_candidates)
    # Prune: if even taking all remaining can't beat best_size, stop
    
    def backtrack(current_set, current_products, remaining, depth_limit=None):
        """Try to extend current_set using elements from remaining."""
        nonlocal best_set, best_size
        
        # Pruning: even if we take all remaining, can we beat best?
        max_possible = len(current_set) + len(remaining)
        if max_possible <= best_size:
            return
        
        # Try to add each candidate
        for i, x in enumerate(remaining):
            if not is_valid_with_products(current_products, x):
                continue
            
            # Tentatively add x
            new_products = set(current_products)
            add_to_products(new_products, x)
            
            new_set = current_set + [x]
            new_remaining = remaining[:i] + remaining[i+1:]
            
            # Update best if improved
            if len(new_set) > best_size:
                best_size = len(new_set)
                best_set = new_set[:]
            
            # Recurse if we can potentially improve further
            if len(new_set) + len(new_remaining) > best_size:
                backtrack(new_set, new_products, new_remaining)
    
    # Greedy with multiple orderings to get good initial solutions
    def greedy_search(ordering):
        current_set = []
        current_products = {1}
        
        for x in ordering:
            if is_valid_with_products(current_products, x):
                current_set.append(x)
                add_to_products(current_products, x)
        
        return current_set
    
    random.seed(42)
    
    # Try different orderings
    orderings = [
        sorted(candidates),
        sorted(candidates, reverse=True),
        candidates[:],
    ]
    
    for ordering in orderings:
        result = greedy_search(ordering)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
    
    # Randomized greedy restarts
    for restart in range(5000):
        if time.time() > deadline - 5:
            break
        
        shuffled = candidates[:]
        random.shuffle(shuffled)
        result = greedy_search(shuffled)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
    
    # Backtracking search from best greedy solution
    if time.time() < deadline - 3:
        # Start from best_set and try to extend
        current_products = {1}
        for elem in best_set:
            add_to_products(current_products, elem)
        
        remaining = [x for x in candidates if x not in best_set]
        
        # Sort remaining by some heuristic (fewer conflicts first)
        # Try backtracking with time limit
        start_bt = time.time()
        backtrack(best_set[:], current_products, remaining)
        
    # Local search: try swaps to find better solutions
    if time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            
            # Try adding any candidate
            for x in candidates:
                if x in best_set:
                    continue
                
                current_products = {1}
                for elem in best_set:
                    add_to_products(current_products, elem)
                
                if is_valid_with_products(current_products, x):
                    best_set.append(x)
                    best_size += 1
                    improved = True
                    break
            
            # Try swaps
            if not improved:
                for i, old_x in enumerate(best_set):
                    trial_set = [x for j, x in enumerate(best_set) if j != i]
                    
                    for new_x in candidates:
                        if new_x in trial_set:
                            continue
                        
                        current_products = {1}
                        for elem in trial_set:
                            add_to_products(current_products, elem)
                        
                        if is_valid_with_products(current_products, new_x):
                            best_set = trial_set + [new_x]
                            improved = True
                            break
                    if improved:
                        break
    
    # More randomized search with time remaining
    start_time = time.time()
    max_time = max(0, deadline - time.time() - 1.0)
    while time.time() - start_time < max_time:
        shuffled = candidates[:]
        random.shuffle(shuffled)
        result = greedy_search(shuffled)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
    
    # Final verification
    return [int(x) for x in best_set]
