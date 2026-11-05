"""Simulated Annealing approach for maximizing sum of 1/(512-a) for pairwise coprime set."""
# EVOLVE-BLOCK-START
import math
import random

def construct():
    MAX_VAL = 511
    N = 512
    
    # Precompute smallest prime factor or prime factors for fast coprimality check
    # We need to check if gcd(a, b) == 1.
    # Storing sets of prime factors is efficient for intersection checks.
    factors = [set() for _ in range(MAX_VAL + 1)]
    for i in range(2, MAX_VAL + 1):
        if not factors[i]: # i is prime
            for j in range(i, MAX_VAL + 1, i):
                factors[j].add(i)
    
    # Helper to check if a is coprime to all elements in the set S
    # S is a list or set of integers
    def is_coprime_to_all(a, S):
        fa = factors[a]
        if not fa: # a=1 is coprime to everything
            return True
        for b in S:
            if fa & factors[b]:
                return False
        return True

    def get_score(S):
        return sum(1.0 / (N - a) for a in S)

    # Initialize with a random valid set or greedy
    # Let's start with a greedy solution to have a good starting point, 
    # but use SA to explore.
    S = set()
    # Greedy initialization: pick largest a first? Or random?
    # Greedy by value (largest a) is a strong seed.
    candidates = list(range(MAX_VAL, 0, -1))
    for a in candidates:
        if is_coprime_to_all(a, S):
            S.add(a)
    
    current_score = get_score(S)
    best_S = set(S)
    best_score = current_score
    
    # Simulated Annealing parameters
    T_start = 1.0
    T_end = 1e-4
    alpha = 0.995 # cooling rate
    max_iterations = 50000
    # We need to be careful with time. 50k iterations should be fast in Python if optimized.
    
    for i in range(max_iterations):
        T = T_start * (alpha ** i)
        if T < T_end:
            break
            
        # Generate a neighbor
        # Move types:
        # 1. Add a random valid element
        # 2. Remove a random element
        # 3. Swap: Remove one, Add one
        
        move_type = random.random()
        new_S = set(S)
        
        if move_type < 0.4:
            # Add
            # Find a random candidate not in S that is coprime to all in S
            # To be efficient, we can pick a random few and check
            added = False
            # Try up to 10 random candidates
            for _ in range(10):
                a = random.randint(1, MAX_VAL)
                if a not in new_S and is_coprime_to_all(a, new_S):
                    new_S.add(a)
                    added = True
                    break
            if not added:
                continue # No valid move, stay
                
        elif move_type < 0.7:
            # Remove
            if len(new_S) > 1: # Keep at least 1
                a = random.choice(list(new_S))
                new_S.remove(a)
        else:
            # Swap
            if len(new_S) == 0:
                continue
            a_to_remove = random.choice(list(new_S))
            new_S.remove(a_to_remove)
            added = False
            for _ in range(10):
                a_to_add = random.randint(1, MAX_VAL)
                if a_to_add not in new_S and is_coprime_to_all(a_to_add, new_S):
                    new_S.add(a_to_add)
                    added = True
                    break
            if not added:
                # If we couldn't add, revert removal? 
                # Standard SA: if move is invalid, reject. 
                # Here, removal is always valid, but swap implies adding.
                # If add fails, the state is just "removed". 
                # This is effectively a "Remove" move. That's fine.
                pass

        new_score = get_score(new_S)
        delta = new_score - current_score
        
        # Acceptance
        if delta > 0 or random.random() < math.exp(delta / T):
            S = new_S
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_S = set(S)
                
    return list(best_S)
# EVOLVE-BLOCK-END
