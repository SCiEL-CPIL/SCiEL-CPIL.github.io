import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11
    
    # Generate weight-3 supports
    supports = list(combinations(range(D), 3))
    N_sup = len(supports)
    
    # Build conflict graph: edge if intersection size >= 2
    # Use bitmasks
    sup_masks = []
    for s in supports:
        mask = 0
        for idx in s:
            mask |= (1 << idx)
        sup_masks.append(mask)
        
    # Precompute conflicts
    # conflict[i] is set of j such that |sup[i] & sup[j]| >= 2
    conflicts = [set() for _ in range(N_sup)]
    
    for i in range(N_sup):
        if time.time() > deadline - 30.0:
            break
        for j in range(i + 1, N_sup):
            inter_count = bin(sup_masks[i] & sup_masks[j]).count('1')
            if inter_count >= 2:
                conflicts[i].add(j)
                conflicts[j].add(i)
                
    # Find Maximum Independent Set in the conflict graph
    # This is equivalent to Max Packing
    best_set = set()
    
    # Strategy: Greedy with random restarts
    # Sort by degree (conflicts) ascending? Low degree is "safer"
    # Actually, for packing, we want to pick items that don't block many others.
    # Degree in conflict graph = number of items it conflicts with.
    # Low degree = conflicts with few others = good candidate.
    
    import random
    random.seed(42)
    
    max_found = 0
    
    for attempt in range(200):
        if time.time() > deadline - 10.0:
            break
            
        # Randomize order slightly or use deterministic greedy
        # Let's try greedy picking lowest degree first
        # Calculate dynamic degrees? Or static?
        # Static degree is fine for first pass.
        
        # Shuffled list for randomization
        order = list(range(N_sup))
        random.shuffle(order)
        
        # Sort by static degree (conflicts count)
        # Lower degree first
        order.sort(key=lambda x: len(conflicts[x]))
        
        selected = set()
        for idx in order:
            if idx in selected:
                continue
            # Check if compatible with all selected
            # i.e., no conflict with any selected
            compatible = True
            for s in selected:
                if s in conflicts[idx]:
                    compatible = False
                    break
            if compatible:
                selected.add(idx)
                
        if len(selected) > max_found:
            max_found = len(selected)
            best_set = set(selected)
            
    # If we didn't find 15, try a simple local search / swap?
    # Given the small size, 200 restarts should find 15 if it exists.
    # The packing number is 15.
    
    # Generate vectors
    vectors = []
    for i in best_set:
        sup = supports[i]
        # 8 sign variations
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                for s3 in [1, -1]:
                    v = np.zeros(D, dtype=np.int8)
                    v[sup[0]] = s1
                    v[sup[1]] = s2
                    v[sup[2]] = s3
                    vectors.append(v)
                    
    if not vectors:
        # Fallback
        v = np.zeros(D, dtype=np.int8)
        v[0]=1; v[1]=1; v[2]=1
        vectors = [v]
        
    return np.array(vectors, dtype=np.int8)
