import itertools
import random

def construct():
    vertices = list(range(16))
    all_triples = list(itertools.combinations(vertices, 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    # Precompute conflicts
    # conflicts[i] is a list of lists of indices. Each inner list represents a set of 7 triples.
    conflicts = [[] for _ in range(n_triples)]
    
    for i, t in enumerate(all_triples):
        u, v, w = t
        remaining = [x for x in vertices if x not in t]
        conf_list = []
        for combo in itertools.combinations(remaining, 3):
            x, y, z = combo
            for perm in itertools.permutations(combo):
                px, py, pz = perm
                others = [
                    (u, v, pz),
                    (u, py, w),
                    (u, py, pz),
                    (px, v, w),
                    (px, v, pz),
                    (px, py, w),
                    (px, py, pz)
                ]
                others_indices = [triple_to_idx[tuple(sorted(o))] for o in others]
                conf_list.append(others_indices)
        conflicts[i] = conf_list

    # Global frequency dictionary for weight updates
    freq = [0.0] * n_triples
    
    best_size = 0
    best_set = set()
    
    num_restarts = 200
    
    for restart in range(num_restarts):
        available = set(range(n_triples))
        current_set = set()
        
        while available:
            # Calculate weights for available triples
            # Lower freq -> Higher weight
            best_idx = -1
            best_weight = -1
            
            for i in available:
                w = 1.0 / (1.0 + freq[i]) + random.random() * 0.01
                if w > best_weight:
                    best_weight = w
                    best_idx = i
            
            if best_idx == -1:
                break
                
            # Check safety for best_idx
            safe = True
            blocking_conflict = None
            for cs in conflicts[best_idx]:
                is_subset = True
                for j in cs:
                    if j not in current_set:
                        is_subset = False
                        break
                if is_subset:
                    safe = False
                    blocking_conflict = cs
                    break
            
            if safe:
                current_set.add(best_idx)
                available.discard(best_idx)
                
                for cs in conflicts[best_idx]:
                    for j in cs:
                        if j in available:
                            freq[j] += 0.5
            else:
                available.discard(best_idx)
                
                if blocking_conflict:
                    for j in blocking_conflict:
                        freq[j] += 1.0
                    freq[best_idx] += 0.5

        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()

    return [list(all_triples[i]) for i in best_set]
