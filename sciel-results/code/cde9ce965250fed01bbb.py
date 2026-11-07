"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

_START_TIME = time.time()
_DEADLINE = _START_TIME + 55.0  # Safety margin for 60s limit

def construct():
    """
    Returns a list of distinct integers in 2..40 such that all 2^k subset products are distinct.
    Maximizes the number of selected integers.
    
    Strategy:
    1. Exhaustive backtracking search over subsets of {2, ..., 40}.
    2. Enforce increasing order to avoid permutations.
    3. Maintain the set of all subset products of the current selection.
    4. Prune branches where the current size + remaining candidates < best found size.
    5. Prune branches where the candidate cannot be added without causing product collisions.
    6. This guarantees finding the optimal set size.
    """
    
    candidates = list(range(2, 41))
    n = len(candidates)
    
    best_set = []
    best_size = 0
    # Pre-compute primes to help with initial lower bound or heuristic ordering if needed,
    # but for exact search, we just need the candidates.
    
    # We can speed up by checking if we can even beat the current best.
    # Initial best can be the set of primes, which is always valid.
    def _is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0: return False
        return True
        
    primes = [x for x in candidates if _is_prime(x)]
    best_set = list(primes)
    best_size = len(primes)
    
    # To make the search faster, we can order candidates.
    # Usually, numbers with fewer prime factors are "safer" to add? 
    # Or maybe we should just go in increasing order and rely on pruning.
    # Actually, a smarter ordering might help prune earlier.
    # Let's stick to increasing order for simplicity and symmetry breaking, 
    # but the branch-and-bound on size is the key.
    
    # We represent the set of products as a set of integers.
    # Since products can be large, we use Python's arbitrary precision integers.
    
    def backtrack(start_idx, current_set, current_prods, current_size):
        nonlocal best_set, best_size
        
        # Deadline check
        if time.time() > _DEADLINE:
            return
            
        # Update best if current is better
        if current_size > best_size:
            best_size = current_size
            best_set = list(current_set)
            
        # Pruning: if we can't beat the best size, return
        # Max possible size = current_size + (n - start_idx)
        if current_size + (n - start_idx) <= best_size:
            return
            
        # Try adding each candidate from start_idx onwards
        for i in range(start_idx, n):
            if time.time() > _DEADLINE:
                return
                
            x = candidates[i]
            
            # Check if x can be added to current_prods
            # New products will be {p * x for p in current_prods}
            # Collision if any p * x is in current_prods
            # Note: p1*x == p2*x implies p1==p2, so new products are distinct among themselves.
            # We only need to check against existing products.
            
            can_add = True
            # Optimization: If x is in current_set, skip (handled by start_idx and distinctness)
            # But we are picking distinct candidates from the list, so x is not in current_set.
            
            # Check collisions
            # We can break early if a collision is found
            for p in current_prods:
                if p * x in current_prods:
                    can_add = False
                    break
                    
            if can_add:
                # Create new set of products
                # new_prods = current_prods union {p * x for p in current_prods}
                # We don't want to modify current_prods in place for backtracking.
                # However, creating a new set every time is expensive.
                # Given the small size (max ~15-20 elements), the set of products has size 2^k.
                # If k=15, 2^15 = 32768 products.
                # Creating a set of 32768 ints is fast enough.
                
                new_prods = set(current_prods)
                for p in current_prods:
                    new_prods.add(p * x)
                    
                current_set.append(x)
                backtrack(i + 1, current_set, new_prods, current_size + 1)
                current_set.pop()
                
            # If we've already found a set of size n (impossible) or other pruning...
            # The main pruning is the size bound.
            
    # Initial call
    # Start with empty set
    initial_prods = {1}
    backtrack(0, [], initial_prods, 0)
    
    # Ensure types are correct
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
