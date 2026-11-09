import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    n = 16
    all_triples = list(combinations(range(n), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # 1. Precompute all K(2,2,2) configurations
    # A K(2,2,2) is defined by 3 disjoint pairs.
    # We iterate over all sets of 6 points, then all pairings of those 6 points.
    # Number of configs: C(16,6) * 15 = 120120
    
    configs = [] # List of lists of 8 triple indices
    # To save memory/time, we might not need to store all if we can generate on fly, 
    # but 120k configs * 8 ints is small.
    
    # Map from triple index to list of config indices containing it
    triple_to_configs = [[] for _ in range(num_triples)]
    
    # We need a fast way to get the 8 triples for a given pairing of 6 points.
    # Let the 6 points be p0..p5.
    # Pairings are permutations of (0,1,2,3,4,5) modulo symmetries? 
    # There are 15 distinct pairings of 6 items.
    # We can hardcode the 15 pairings of indices 0..5.
    
    pairings = [
        ((0,1),(2,3),(4,5)),
        ((0,1),(2,4),(3,5)),
        ((0,1),(2,5),(3,4)),
        ((0,2),(1,3),(4,5)),
        ((0,2),(1,4),(3,5)),
        ((0,2),(1,5),(3,4)),
        ((0,3),(1,2),(4,5)),
        ((0,3),(1,4),(2,5)),
        ((0,3),(1,5),(2,4)),
        ((0,4),(1,2),(3,5)),
        ((0,4),(1,3),(2,5)),
        ((0,4),(1,5),(2,3)),
        ((0,5),(1,2),(3,4)),
        ((0,5),(1,3),(2,4)),
        ((0,5),(1,4),(2,3))
    ]
    
    # Generate configs
    for six_points in combinations(range(n), 6):
        # six_points is a tuple of 6 distinct points
        # Map index 0-5 to actual points
        pts = six_points
        for pairing in pairings:
            p1 = (pts[pairing[0][0]], pts[pairing[0][1]])
            p2 = (pts[pairing[1][0]], pts[pairing[1][1]])
            p3 = (pts[pairing[2][0]], pts[pairing[2][1]])
            
            # Generate the 8 transversals
            transversals = []
            for a in p1:
                for b in p2:
                    for c in p3:
                        # Ensure sorted tuple for lookup
                        t = tuple(sorted((a, b, c)))
                        transversals.append(t)
            
            # Convert to indices
            t_indices = [triple_to_idx[t] for t in transversals]
            
            # Store config
            config_idx = len(configs)
            configs.append(t_indices)
            
            # Update reverse mapping
            for t_idx in t_indices:
                triple_to_configs[t_idx].append(config_idx)
    
    # 2. Greedy Construction
    def greedy_construct(seed):
        random.seed(seed)
        selected = set()
        forbidden = set()
        counts = [0] * len(configs)
        
        candidates = list(range(num_triples))
        random.shuffle(candidates)
        
        for t_idx in candidates:
            if time.time() > deadline - 2:
                break
                
            if t_idx in forbidden:
                continue
                
            # Check if adding this triple creates a K(2,2,2)
            # It does if any config containing it already has 7 selected.
            # We track counts. If counts[k] == 7, the 8th is forbidden.
            # So we just check if t_idx is in forbidden.
            # Wait, the logic above says:
            # "If counts[k] == 7, mark the missing one as forbidden."
            # So if t_idx is in forbidden, we skip.
            # If not, we add it.
            
            selected.add(t_idx)
            
            # Update counts for all configs containing t_idx
            for k in triple_to_configs[t_idx]:
                counts[k] += 1
                if counts[k] == 7:
                    # Find the missing triple in this config
                    # The config has 8 triples. 7 are in selected.
                    # We need to find the one NOT in selected.
                    # Since we just added t_idx, and 6 others were added before, 
                    # and 1 is missing.
                    # We can iterate through the 8 triples of config k.
                    for t_candidate in configs[k]:
                        if t_candidate not in selected:
                            forbidden.add(t_candidate)
                            break
                            
        return list(selected)

    # Run multiple restarts
    best_result = []
    best_size = 0
    num_restarts = 20 # Adjust based on speed
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        result = greedy_construct(seed=i)
        if len(result) > best_size:
            best_size = len(result)
            best_result = result
            
    # Convert indices back to triples
    return [list(all_triples[idx]) for idx in best_result]
