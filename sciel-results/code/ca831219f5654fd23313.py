import random
from itertools import combinations
from math import lcm, gcd

def construct():
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute forbidden triples and map pairs to conflicting third elements
    # forbidden_by_pair[key] contains c if (a,b,c) is a forbidden triple
    forbidden_by_pair = {}
    
    # To handle Python versions where math.lcm might not exist (though 3.9+ is standard)
    # We assume math.lcm exists as per standard modern Python.
    # If we want to be safe:
    # def _lcm(a, b):
    #     return a * b // gcd(a, b)
    
    for a, b, c in combinations(numbers, 3):
        # Check condition
        # lcm(a,b) == lcm(a,c) == lcm(b,c)
        if lcm(a, b) == lcm(a, c) == lcm(b, c):
            # Add to map
            forbidden_by_pair.setdefault(tuple(sorted((a, b))), set()).add(c)
            forbidden_by_pair.setdefault(tuple(sorted((a, c))), set()).add(b)
            forbidden_by_pair.setdefault(tuple(sorted((b, c))), set()).add(a)
            
    empty_set = frozenset()
    best_set = set()
    
    # Run randomized greedy multiple times
    # 500 restarts to be safe within time limits
    for _ in range(500):
        random.shuffle(numbers)
        current_set = []
        
        for x in numbers:
            valid = True
            # Check against all pairs in current_set
            # Optimization: If current_set is large, this is the bottleneck
            for a, b in combinations(current_set, 2):
                key = tuple(sorted((a, b)))
                # Check if x conflicts with this pair
                if x in forbidden_by_pair.get(key, empty_set):
                    valid = False
                    break
            
            if valid:
                current_set.append(x)
                
        if len(current_set) > len(best_set):
            best_set = set(current_set)
            
    if not best_set:
        return [1]
        
    return list(best_set)
