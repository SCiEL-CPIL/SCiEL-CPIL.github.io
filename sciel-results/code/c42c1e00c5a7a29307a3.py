import random
from itertools import combinations

def construct():
    n = 16
    all_triples = list(combinations(range(n), 3))
    
    # We need to check if adding a triple creates a K(2,2,2)
    # A K(2,2,2) consists of 8 triples from 3 disjoint pairs
    # Precompute for each triple, what K(2,2,2)s it could be part of
    
    # Actually, let's use a different approach:
    # For each set of 3 disjoint pairs, check if all 8 transversal triples are present
    # When we add a new triple, we only need to check K(2,2,2)s that include this triple
    
    # A triple {a,b,c} is part of a K(2,2,2) if we can find pairs {a,a'}, {b,b'}, {c,c'} 
    # such that a', b', c' are distinct and different from a,b,c
    # The 8 triples are all combinations of choosing from each pair
    
    # For efficiency, let's maintain the set of present triples
    present = set()
    result = []
    
    # Precompute: for each triple, list all possible K(2,2,2) configurations it belongs to
    # A K(2,2,2) is determined by 3 disjoint pairs. For triple (a,b,c),
    # we need a' != a,b,c; b' != a,b,c; c' != a,b,c; and a',b',c' all distinct
    # and also a' != b, a' != c, b' != a, b' != c, c' != a, c' != b (which is implied by distinctness from a,b,c and each other)
    
    # This is getting complex. Let me just do a direct check.
    
    # Alternative: use the fact that the seed gives 105. Let me try to find a better structured solution.
    
    # What about: take all triples from a 5x5 grid minus one point? No, that's 24 points.
    
    # Let me try: partition 16 into two sets of 8, and take all triples that have at least 2 points from the first set.
    # Count: C(8,2)*8 + C(8,3) = 28*8 + 56 = 224 + 56 = 280
    # Is this valid? Three disjoint pairs: if all 6 points are in the second set, no triples. 
    # If 4 points in first set, 2 in second: the pairs must be distributed. 
    # This is getting complicated. Let me just try the greedy random approach with a timeout safeguard.
    
    random.seed(42)
    random.shuffle(all_triples)
    
    present = set()
    result = []
    
    # For checking K(2,2,2), when we add triple t=(a,b,c), we need to check
    # if there exist a',b',c' such that all 8 triples from pairs {a,a'},{b,b'},{c,c'} are present
    # The 8 triples are: (a,b,c), (a,b,c'), (a,b',c), (a,b',c'), (a',b,c), (a',b,c'), (a',b',c), (a',b',c')
    
    # For efficiency, precompute for each triple, the list of "completions" - i.e., for triple (a,b,c),
    # the possible (a',b',c') that would form a K(2,2,2)
    
    # Actually, let me just check directly. For each candidate triple, check all possible K(2,2,2) it could complete.
    
    # Number of possible K(2,2,2) containing a given triple (a,b,c):
    # a' can be any of 13 points (not a,b,c), b' any of 12 (not a,b,c,a'), c' any of 11 (not a,b,c,a',b')
    # But we also need {a,a'},{b,b'},{c,c'} to be disjoint, which is automatically satisfied if a',b',c' are distinct and not in {a,b,c}
    # So ~13*12*11 = 1716 checks per triple, times 560 triples = ~960K checks. Should be fast enough.
    
    for t in all_triples:
        a, b, c = t
        can_add = True
        for a_prime in range(n):
            if a_prime in (a, b, c):
                continue
            for b_prime in range(n):
                if b_prime in (a, b, c, a_prime):
                    continue
                for c_prime in range(n):
                    if c_prime in (a, b, c, a_prime, b_prime):
                        continue
                    # Check if all 8 triples are present (including the new one)
                    # The 8 triples:
                    triples_to_check = [
                        (a, b, c),
                        (a, b, c_prime),
                        (a, b_prime, c),
                        (a, b_prime, c_prime),
                        (a_prime, b, c),
                        (a_prime, b, c_prime),
                        (a_prime, b_prime, c),
                        (a_prime, b_prime, c_prime),
                    ]
                    # Check if all are in present or equal to t
                    all_present = True
                    for tt in triples_to_check:
                        tt_sorted = tuple(sorted(tt))
                        if tt_sorted != t and tt_sorted not in present:
                            all_present = False
                            break
                    if all_present:
                        can_add = False
                        break
                if not can_add:
                    break
            if not can_add:
                break
        if can_add:
            present.add(tuple(sorted(t)))
            result.append(list(t))
    
    return result
