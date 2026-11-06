import time
import random
from math import lcm

def construct():
    start_time = time.time()
    deadline = 85.0
    
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute LCMs
    lcm_cache = {}
    def get_lcm(a, b):
        if a > b: a, b = b, a
        key = (a, b)
        if key not in lcm_cache:
            lcm_cache[key] = lcm(a, b)
        return lcm_cache[key]
        
    # Identify all bad triples and map them to elements
    triple_map = {x: [] for x in elements}
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            l_ab = get_lcm(a, b)
            for c in range(b + 1, N + 1):
                l_ac = get_lcm(a, c)
                if l_ac != l_ab:
                    continue
                l_bc = get_lcm(b, c)
                if l_bc == l_ab:
                    triple_map[a].append((b, c))
                    triple_map[b].append((a, c))
                    triple_map[c].append((a, b))
                    
    best_S = set()
    best_size = 0
    
    # Run many randomized greedy trials
    # We can also try different orderings (e.g., by value, by degree)
    
    # To diversify, we can use different random seeds
    # And maybe different initial subsets?
    # Just shuffling is a good start.
    
    num_trials = 5000
    for trial in range(num_trials):
        if time.time() > deadline - 1:
            break
            
        # Use trial as seed for reproducibility in debugging, but random is fine
        random.seed(trial)
        elems = elements.copy()
        random.shuffle(elems)
        
        S = set()
        
        for x in elems:
            compatible = True
            for a, b in triple_map[x]:
                if a in S and b in S:
                    compatible = False
                    break
                    
            if compatible:
                S.add(x)
                
        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()
            
    return list(best_S)
