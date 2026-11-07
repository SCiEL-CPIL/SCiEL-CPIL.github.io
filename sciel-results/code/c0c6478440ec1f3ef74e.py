"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    candidates = list(range(40, 1, -1))
    
    # Try to find the largest possible set size
    # We know the size is at least what the old greedy found (likely ~10-12)
    # Let's try to find 15, then 14, etc.
    
    def backtrack(start_idx, current_set, current_prods, target_size):
        if len(current_set) == target_size:
            return current_set.copy()
        
        # Pruning: if not enough candidates left
        if len(current_set) + (len(candidates) - start_idx) < target_size:
            return None
            
        for i in range(start_idx, len(candidates)):
            x = candidates[i]
            # Check if x can be added
            # new_prods = {p * x for p in current_prods}
            # Check intersection
            if any((p * x) in current_prods for p in current_prods):
                continue
            
            # If no collision, add it
            new_prods = current_prods | {p * x for p in current_prods}
            res = backtrack(i + 1, current_set + [x], new_prods, target_size)
            if res is not None:
                return res
        return None

    # Try sizes from high to low
    # Max possible is 39, but realistically much lower.
    # Start from a reasonable high bound.
    for target in range(15, 1, -1):
        # Initial products: {1}
        initial_prods = {1}
        result = backtrack(0, [], initial_prods, target)
        if result is not None:
            return result
            
    # Fallback to a simple valid set if backtracking fails or times out (unlikely for small N)
    # But we must return a list.
    # A set of primes always works?
    # Primes in 2..40: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37. (12 primes)
    # Subset products of primes are distinct by Fundamental Theorem of Arithmetic.
    # So 12 is always achievable.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    return primes

# EVOLVE-BLOCK-END
