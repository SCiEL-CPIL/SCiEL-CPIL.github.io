import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    points = list(range(16))
    all_triples = list(combinations(points, 3))
    
    triple_pairs = []
    for t in all_triples:
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
        triple_pairs.append(pairs)
        
    best_alpha = 16
    best_triples = []
    
    def has_independent_set_size(triple_list, k):
        masks = []
        for t in triple_list:
            mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            masks.append(mask)
            
        for subset in combinations(points, k):
            mask = 0
            for p in subset:
                mask |= (1 << p)
                
            is_ind = True
            for m in masks:
                if (m & mask) == m:
                    is_ind = False
                    break
            if is_ind:
                return True
        return False

    def get_alpha(triple_list, max_k=16):
        # Find largest k such that has_independent_set_size is True
        # We can binary search? No, monotonic.
        # If size k is independent, size k-1 is independent.
        # So we can check from max_k down.
        for k in range(max_k, 0, -1):
            if has_independent_set_size(triple_list, k):
                return k
        return 0

    # Initial best: seed
    seed_triples = [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    best_alpha = get_alpha(seed_triples)
    best_triples = seed_triples
    
    num_restarts = 5000
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        indices = list(range(len(all_triples)))
        random.shuffle(indices)
        
        current_triples = []
        used_pairs = set()
        
        for idx in indices:
            t = all_triples[idx]
            pairs = triple_pairs[idx]
            
            conflict = False
            for p in pairs:
                if p in used_pairs:
                    conflict = True
                    break
                
            if conflict:
                continue
                
            current_triples.append(t)
            for p in pairs:
                used_pairs.add(p)
                
        if not current_triples:
            continue
            
        # Check if this is better than best_alpha
        # If there is an independent set of size best_alpha, it's not better (or equal)
        # We want strictly smaller alpha.
        if has_independent_set_size(current_triples, best_alpha):
            continue
            
        # It might be better. Find the actual alpha.
        # It must be < best_alpha.
        # Check best_alpha - 1
        alpha = best_alpha - 1
        while alpha > 0 and has_independent_set_size(current_triples, alpha):
            alpha -= 1
        # Wait, if has_independent_set_size(alpha) is True, then the max is at least alpha.
        # We know max < best_alpha.
        # So we check best_alpha - 1.
        # If True, max >= best_alpha - 1.
        # We want the LARGEST independent set.
        # So if has_independent_set_size(best_alpha - 1) is True, the alpha is at least best_alpha - 1.
        # It could be best_alpha - 1.
        # We need to check if it's larger? No, we know it's < best_alpha.
        # So if has_independent_set_size(best_alpha - 1) is True, the alpha is exactly best_alpha - 1?
        # No, it could be best_alpha - 1.
        # But could it be best_alpha? No, we checked that.
        # So if has_independent_set_size(best_alpha - 1) is True, then alpha >= best_alpha - 1.
        # Since alpha < best_alpha, alpha must be best_alpha - 1.
        
        if has_independent_set_size(current_triples, best_alpha - 1):
            new_alpha = best_alpha - 1
        else:
            # Check lower
            new_alpha = best_alpha - 1
            while new_alpha > 0 and not has_independent_set_size(current_triples, new_alpha):
                new_alpha -= 1
            # If new_alpha becomes 0, it means no independent set of size 1? Impossible.
            # So this loop will find the largest k < best_alpha - 1 that is independent.
            
        if new_alpha < best_alpha:
            best_alpha = new_alpha
            best_triples = current_triples

    return best_triples
