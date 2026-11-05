"""Rewritten program using Local Search (Hill Climbing) to improve upon the greedy seed."""
# EVOLVE-BLOCK-START
def construct():
    import math
    import random
    
    MAX_VAL = 511
    N = 512
    
    # Precompute prime factors for all numbers 1..511
    factors = [set() for _ in range(MAX_VAL + 1)]
    for i in range(2, MAX_VAL + 1):
        if not factors[i]: # i is prime
            for j in range(i, MAX_VAL + 1, i):
                factors[j].add(i)
    
    # Helper to check if a number is coprime to a set of numbers
    def is_coprime_to_set(a, S):
        fa = factors[a]
        for b in S:
            if fa & factors[b]: # intersection is non-empty
                return False
        return True

    def get_score(S):
        return sum(1.0 / (N - a) for a in S)

    # 1. Generate initial greedy solution
    S = []
    for a in range(MAX_VAL, 0, -1):
        if is_coprime_to_set(a, S):
            S.append(a)
    
    current_score = get_score(S)
    
    # 2. Local Search: Add and Swap moves
    # We will try to improve the set by adding valid numbers or swapping out one for better combinations
    
    max_iterations = 2000
    no_improve_count = 0
    
    for _ in range(max_iterations):
        improved = False
        
        # Move Type 1: Add a single number
        # Find the best candidate not in S that is coprime to all in S
        best_add = None
        best_add_val = 0
        for a in range(MAX_VAL, 0, -1):
            if a not in S:
                if is_coprime_to_set(a, S):
                    val = 1.0 / (N - a)
                    if val > best_add_val:
                        best_add_val = val
                        best_add = a
        
        if best_add is not None:
            S.append(best_add)
            current_score += best_add_val
            improved = True
            # After adding, we might be able to add more, so continue loop
            continue
            
        # Move Type 2: Swap/Refill
        # Try removing each element in S and refilling greedily
        best_swap_set = None
        best_swap_score = current_score
        
        # Sort S to make removals deterministic or just iterate
        for a in list(S):
            S_temp = [x for x in S if x != a]
            score_temp = current_score - (1.0 / (N - a))
            
            # Greedily fill S_temp
            # We want to add numbers that are coprime to S_temp
            # Sort candidates by value descending
            # To speed up, we can just iterate all a from MAX_VAL down
            added_any = False
            for b in range(MAX_VAL, 0, -1):
                if b in S_temp:
                    continue
                if is_coprime_to_set(b, S_temp):
                    S_temp.append(b)
                    score_temp += 1.0 / (N - b)
                    added_any = True
            
            if score_temp > best_swap_score:
                best_swap_score = score_temp
                best_swap_set = S_temp
                improved = True # We found a better set, we will apply it after loop

        if best_swap_set is not None:
            S = best_swap_set
            current_score = best_swap_score
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 50:
                break
                
    return S
# EVOLVE-BLOCK-END
