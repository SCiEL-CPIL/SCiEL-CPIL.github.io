"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import gcd
    import itertools
    
    n = 512
    # Precompute gcd conflicts
    # Two numbers a, b in 1..511 conflict if gcd(a,b) > 1
    
    # Greedy initial solution: pick from 511 down
    chosen_set = set()
    for a in range(511, 0, -1):
        if all(gcd(a, b) == 1 for b in chosen_set):
            chosen_set.add(a)
    
    chosen_list = list(chosen_set)
    
    def score(s):
        return sum(1.0 / (n - a) for a in s)
    
    current_score = score(chosen_list)
    
    # Local search: try to add, swap
    improved = True
    while improved:
        improved = False
        
        # Try adding any excluded number
        for a in range(511, 0, -1):
            if a not in chosen_set:
                if all(gcd(a, b) == 1 for b in chosen_set):
                    chosen_set.add(a)
                    new_score = score(chosen_set)
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                    else:
                        chosen_set.discard(a)
        
        # Try single swaps: remove c, add e
        chosen_list = list(chosen_set)
        for c in chosen_list:
            for e in range(511, 0, -1):
                if e not in chosen_set:
                    # Check if e is coprime with all chosen except c
                    if all(gcd(e, b) == 1 for b in chosen_set if b != c):
                        old_score = current_score
                        chosen_set.discard(c)
                        chosen_set.add(e)
                        new_score = score(chosen_set)
                        if new_score > old_score:
                            current_score = new_score
                            improved = True
                        else:
                            chosen_set.discard(e)
                            chosen_set.add(c)
    
    return list(chosen_set)
# EVOLVE-BLOCK-END
