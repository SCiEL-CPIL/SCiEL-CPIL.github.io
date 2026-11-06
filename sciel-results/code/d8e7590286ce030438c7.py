import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n = 16
    points = list(range(n))
    all_triples = list(itertools.combinations(points, 3))
    
    # Precompute bitmasks and pairs for each triple
    triple_masks = [0] * len(all_triples)
    triple_pairs = []
    for i, t in enumerate(all_triples):
        mask = 0
        a, b, c = t
        mask |= (1 << a)
        mask |= (1 << b)
        mask |= (1 << c)
        triple_masks[i] = mask
        triple_pairs.append([(a, b), (a, c), (b, c)])
        
    # Precompute masks for subsets of each size for independence check
    masks_by_size = [[] for _ in range(n + 1)]
    for r in range(1, n + 1):
        for subset in itertools.combinations(points, r):
            mask = 0
            for p in subset:
                mask |= (1 << p)
            masks_by_size[r].append(mask)

    def get_alpha(edge_masks):
        # Check subsets from size n down to 1
        for k in range(n, 0, -1):
            for s_mask in masks_by_size[k]:
                is_independent = True
                for e in edge_masks:
                    if (e & s_mask) == e:
                        is_independent = False
                        break
                if is_independent:
                    return k
        return 1

    best_alpha = n + 1
    best_triples = []
    
    indices = list(range(len(all_triples)))
    
    # Randomized Greedy Construction
    while time.time() < deadline:
        random.shuffle(indices)
        used_pairs = set()
        current_edges = []
        current_masks = []
        
        for t_idx in indices:
            pairs = triple_pairs[t_idx]
            # Check if any pair is already used
            if pairs[0] in used_pairs or pairs[1] in used_pairs or pairs[2] in used_pairs:
                continue
            
            # Add triple
            used_pairs.add(pairs[0])
            used_pairs.add(pairs[1])
            used_pairs.add(pairs[2])
            
            t = all_triples[t_idx]
            current_edges.append(t)
            current_masks.append(triple_masks[t_idx])
            
        # Evaluate independence number
        alpha = get_alpha(current_masks)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = current_edges.copy()
            # Optional: if best_alpha is very low, we could break early
            # but let's keep searching for a bit to see if we can go lower
            if best_alpha <= 5:
                # 5 is likely the theoretical minimum or close to it
                # We can break or continue. Let's continue to be safe.
                pass

    return best_triples
