import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s limit
    
    # Candidate numbers
    candidates = list(range(2, 41))
    
    # Heuristic order: try to include primes and small composites?
    # Actually, let's just sort by value.
    # Or maybe random restarts?
    
    best_set = []
    
    def is_valid(S, P):
        # S is list of ints, P is set of products
        # Check if adding x is valid
        pass # Not used directly in this structure
        
    def backtrack(idx, current_set, current_products):
        nonlocal best_set
        if time.time() > deadline:
            return
        
        if len(current_set) > len(best_set):
            best_set = list(current_set)
            
        if idx == len(candidates):
            return
            
        # Pruning
        if len(current_set) + (len(candidates) - idx) <= len(best_set):
            return
            
        c = candidates[idx]
        
        # Try adding c
        new_products = set()
        valid = True
        for p in current_products:
            np = p * c
            if np in current_products or np in new_products:
                valid = False
                break
            new_products.add(np)
            
        if valid:
            current_products.update(new_products)
            current_set.append(c)
            backtrack(idx + 1, current_set, current_products)
            current_set.pop()
            current_products.difference_update(new_products)
            
        # Try skipping c
        backtrack(idx + 1, current_set, current_products)

    # Initial guess: primes
    primes = [a for a in range(2, 41) if all(a % d for d in range(2, int(a**0.5)+1))]
    # Sort candidates to improve pruning?
    # Smaller numbers first might be better for finding large sets early?
    # Or larger numbers first?
    # Let's try sorting by value.
    candidates.sort()
    
    current_set = []
    current_products = {1}
    
    backtrack(0, current_set, current_products)
    
    return best_set
