import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    all_triple_masks = [ (1 << a) | (1 << b) | (1 << c) for a, b, c in all_triples ]
    
    # Map triple tuple to mask? Or just use index?
    # We will store triples as (tuple, mask) or just use indices into all_triples?
    # Using indices is memory efficient but lookup is slower.
    # Let's store the actual triple tuple and its mask in a list of tuples?
    # Or just two parallel lists.
    
    # Pre-compute vertex masks sorted by popcount desc
    vertex_masks = list(range(1, 1 << n))
    vertex_masks.sort(key=lambda x: x.bit_count(), reverse=True)
    
    def get_alpha(triple_masks_list):
        """
        triple_masks_list: list of integer masks for the triples in the hypergraph.
        """
        if not triple_masks_list:
            return n
        
        # Check independence for each mask in descending size order
        for mask in vertex_masks:
            is_independent = True
            for tm in triple_masks_list:
                if (mask & tm) == tm:
                    is_independent = False
                    break
            if is_independent:
                return mask.bit_count()
        return 0

    def is_linear_mask(t_mask, existing_masks):
        """Check if t_mask is linear with existing_masks."""
        for em in existing_masks:
            if (t_mask & em).bit_count() > 1:
                return False
        return True

    def random_greedy():
        triples = []
        triple_masks = []
        shuffled_indices = list(range(len(all_triples)))
        random.shuffle(shuffled_indices)
        
        for idx in shuffled_indices:
            t = all_triples[idx]
            tm = all_triple_masks[idx]
            if is_linear_mask(tm, triple_masks):
                triples.append(t)
                triple_masks.append(tm)
        return triples, triple_masks

    best_alpha = n
    best_triples = []
    
    while time.time() < deadline:
        triples, triple_masks = random_greedy()
        alpha = get_alpha(triple_masks)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples[:]
            
        # Local Search
        if best_alpha < 8:
            current_alpha = alpha
            for _ in range(50):
                if time.time() > deadline:
                    break
                
                if not triple_masks:
                    break
                
                idx = random.randint(0, len(triple_masks) - 1)
                removed_triple = triples[idx]
                removed_mask = triple_masks[idx]
                
                triples.pop(idx)
                triple_masks.pop(idx)
                
                # Pick random new triple
                new_idx = random.randint(0, len(all_triples) - 1)
                new_triple = all_triples[new_idx]
                new_mask = all_triple_masks[new_idx]
                
                if is_linear_mask(new_mask, triple_masks):
                    triples.append(new_triple)
                    triple_masks.append(new_mask)
                    
                    new_alpha = get_alpha(triple_masks)
                    
                    if new_alpha < current_alpha:
                        current_alpha = new_alpha
                        if current_alpha < best_alpha:
                            best_alpha = current_alpha
                            best_triples = triples[:]
                    else:
                        triples.pop()
                        triple_masks.pop()
                        triples.append(removed_triple)
                        triple_masks.append(removed_mask)
                else:
                    triples.append(removed_triple)
                    triple_masks.append(removed_mask)
                    
    return best_triples
