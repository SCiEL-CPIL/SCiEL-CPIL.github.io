import itertools
import numpy as np
import time

D = 11

def construct():
    start_time = time.time()
    deadline = 140  # 10s margin
    
    # Generate all 4-sparse vectors with signs
    # Index: (tuple of 4 indices, tuple of 4 signs)
    # We need to select a subset such that no 3-subset-sign is repeated.
    
    all_vectors = []
    for indices in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(indices, signs):
                v[idx] = s
            all_vectors.append(v)
            
    # Total 5280 vectors.
    # We need to pick max subset.
    # Constraint: For any pair, they cannot share a 3-subset with same signs.
    # This is equivalent to: The set of (3-subset, signs) used by each vector must be disjoint from others.
    
    # Map each 3-subset-sign to a set of vectors that use it.
    # Or simpler: Greedy selection.
    # Shuffle vectors? Or sort by some heuristic?
    
    selected = []
    used_slots = set() # (indices_tuple, signs_tuple) for size 3
    
    # To make it fast, we can pre-calculate the 3-slots for each vector.
    # 5280 vectors * 4 slots = 21120 slots.
    # 1320 unique slots.
    
    vec_slots = []
    for v in all_vectors:
        indices = [i for i in range(D) if v[i] != 0]
        signs = tuple(v[i] for i in indices)
        slots = []
        for j in range(4):
            # Remove j-th element
            idx_3 = tuple(x for k, x in enumerate(indices) if k != j)
            sig_3 = tuple(s for k, s in enumerate(signs) if k != j)
            slots.append((idx_3, sig_3))
        vec_slots.append(slots)
        
    # Randomize order to avoid bias?
    indices_order = list(range(len(all_vectors)))
    np.random.shuffle(indices_order)
    
    for i in indices_order:
        if time.time() > deadline:
            break
        if len(selected) >= 330:
            break
            
        slots = vec_slots[i]
        # Check if any slot is already used
        conflict = False
        for slot in slots:
            if slot in used_slots:
                conflict = True
                break
        
        if not conflict:
            selected.append(all_vectors[i])
            for slot in slots:
                used_slots.add(slot)
                
    if not selected:
        # Fallback to seed
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return np.array(selected, dtype=np.int64)
