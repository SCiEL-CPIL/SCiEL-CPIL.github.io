import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140

    def time_left():
        return deadline - time.time()

    # 1. Generate candidates (Weight 4)
    supports = list(itertools.combinations(range(D), 4))
    candidates = []
    support_to_indices = {}
    
    idx = 0
    for s in supports:
        support_to_indices[s] = []
        for signs in itertools.product((-1, 1), repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for i, val in zip(s, signs):
                v[i] = val
            candidates.append(v)
            support_to_indices[s].append(idx)
            idx += 1
            
    candidates = np.array(candidates, dtype=np.int8)
    N = len(candidates)
    
    # 2. Conflict Matrix
    dots = candidates.astype(np.int16) @ candidates.astype(np.int16).T
    conflict = (dots > 2).astype(np.uint8)
    np.fill_diagonal(conflict, 0)
    
    # 3. Initialize with 3-(11,4,1) packing
    triples = list(itertools.combinations(range(D), 3))
    used_triples = set()
    packing_supports = []
    
    # Shuffle to get different packings? Fixed for reproducibility?
    # Let's use a fixed seed for the packing generation to be consistent.
    rng_pack = random.Random(0)
    shuffled_supports = supports[:]
    rng_pack.shuffle(shuffled_supports)
    
    for s in shuffled_supports:
        if time_left() < 10:
            break
        s_triples = list(itertools.combinations(s, 3))
        if any(t in used_triples for t in s_triples):
            continue
        packing_supports.append(s)
        for t in s_triples:
            used_triples.add(t)
            
    initial_set = set()
    for s in packing_supports:
        initial_set.update(support_to_indices[s])
        
    current_set = initial_set
    best_set = set(initial_set)
    best_size = len(best_set)
    
    # Maintain in_set mask for fast compatibility checks
    in_set = np.zeros(N, dtype=bool)
    for i in current_set:
        in_set[i] = True
        
    all_indices = list(range(N))
    rng = random.Random(42)
    
    steps = 0
    
    while time_left() > 5:
        steps += 1
        
        # 1. Try to Add
        idx = rng.choice(all_indices)
        
        if not in_set[idx]:
            # Check compatibility
            # conflict[idx] is a row of uint8
            # We need to check if any j in current_set (where in_set[j]=1) has conflict
            # If sum(conflict[idx, in_set]) == 0, then compatible
            if conflict[idx, in_set].sum() == 0:
                current_set.add(idx)
                in_set[idx] = True
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                continue
        
        # 2. Try to Swap
        # Pick v in current_set
        # To pick randomly from set, we need a list.
        # Maintaining a list of current_set elements is good.
        # But set is fine for small sizes if we convert.
        # Let's maintain a list `current_list` parallel to `in_set`?
        # Or just `list(current_set)` which is O(K). K~600. 10000 steps -> 6M ops. OK.
        
        if current_set:
            v = rng.choice(list(current_set))
            u = rng.choice(all_indices)
            
            if u not in current_set and u != v:
                # Check if u is compatible with current_set - {v}
                # Create a temporary mask?
                # Or just check conflict[u, in_set] and subtract conflict[u, v]
                # If conflict[u, v] is 1, then conflict[u, in_set] includes it.
                # We want to check if conflict[u, in_set] == conflict[u, v] (which is 0 or 1)
                # i.e., u only conflicts with v (or nobody) in the current set.
                
                conf_with_set = conflict[u, in_set].sum()
                conf_with_v = conflict[u, v]
                
                # If u is compatible with all others, then conf_with_set should be equal to conf_with_v
                # because v is in the set.
                # If conf_with_set == conf_with_v, then u has no other conflicts.
                if conf_with_set == conf_with_v:
                    # Swap
                    current_set.discard(v)
                    in_set[v] = False
                    current_set.add(u)
                    in_set[u] = True
                    # Size unchanged
                    
        # 3. Random Removal to escape local minima
        if steps % 500 == 0 and current_set:
             if rng.random() < 0.2:
                 v = rng.choice(list(current_set))
                 current_set.discard(v)
                 in_set[v] = False

    result_indices = list(best_set)
    if not result_indices:
        return np.zeros((0, D), dtype=np.int64)
        
    result = candidates[result_indices].astype(np.int64)
    
    # Verify
    if len(result) > 1:
        max_norm_sq = np.max(np.sum(result ** 2, axis=1))
        valid = True
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                d_sq = np.sum((result[i] - result[j]) ** 2)
                if d_sq < max_norm_sq:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
             vectors = []
             for i, j in itertools.combinations(range(D), 2):
                 for si in (-1, 1):
                     for sj in (-1, 1):
                         v = np.zeros(D, dtype=np.int64)
                         v[i] = si
                         v[j] = sj
                         vectors.append(v)
             return np.array(vectors, dtype=np.int64)
             
    return result

# EVOLVE-BLOCK-END
