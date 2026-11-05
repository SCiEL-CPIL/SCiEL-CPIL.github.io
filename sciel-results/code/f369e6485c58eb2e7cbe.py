import random
import itertools

def calculate_independence_number_and_sets(triples, k_target=None, max_sets=100):
    """
    Calculate the independence number. If k_target is provided, returns up to max_sets
    independent sets of that size. If k_target is None, returns the number and one set of that size.
    """
    if not triples:
        return 16, [list(range(16))] if k_target == 16 else []
        
    triple_masks = []
    for t in triples:
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks.append(mask)
    
    def is_independent(mask):
        for tm in triple_masks:
            if (tm & mask) == tm:
                return False
        return True

    # Find alpha
    alpha = 0
    sets_found = []
    
    # We search from largest k down
    # Optimization: If we just need alpha, we can stop at first found.
    # If we need sets, we collect them.
    
    for k in range(16, 0, -1):
        found_any = False
        for subset in itertools.combinations(range(16), k):
            sub_mask = 0
            for p in subset:
                sub_mask |= (1 << p)
            if is_independent(sub_mask):
                if k_target is None or k == k_target:
                    found_any = True
                    if max_sets > 0:
                        sets_found.append(list(subset))
                        if len(sets_found) >= max_sets:
                            return k, sets_found
                else:
                    # We found a set of size k, but we are looking for k_target?
                    # If k_target is larger than current k, we continue.
                    # If k_target is smaller, we skip.
                    # But we are iterating down. If we pass k_target, we stop.
                    pass
        
        if found_any:
            if k_target is None:
                return k, sets_found[:1] # Return just one for alpha calc
            elif k == k_target:
                return k, sets_found
            else:
                # k < k_target, so no sets of size k_target exist
                return k, []
                
    return 0, []

def greedy_construct_optimized(seed):
    random.seed(seed)
    
    # Precompute all triples and their masks
    all_triples = []
    triple_masks_map = {}
    for t in itertools.combinations(range(16), 3):
        mask = 0
        for p in t:
            mask |= (1 << p)
        all_triples.append(t)
        triple_masks_map[t] = mask
        
    # Shuffle to break symmetry
    random.shuffle(all_triples)
    
    selected_triples = []
    used_pairs = set()
    
    for _ in range(50): # Max 40 triples, 50 is safe
        # Find valid candidates
        valid_candidates = []
        for t in all_triples:
            pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
            valid = True
            for p in pairs:
                if p in used_pairs:
                    valid = False
                    break
            if valid:
                valid_candidates.append(t)
                
        if not valid_candidates:
            break
            
        # Get current alpha and sample independent sets
        # We want to pick a triple that hits many large independent sets
        # Let's get alpha first
        alpha, sample_sets = calculate_independence_number_and_sets(selected_triples, k_target=None, max_sets=1)
        
        # If alpha is 0 or 1, we are done? No, alpha 1 means no independent set of size 2? 
        # Wait, independence number is the size of the LARGEST set.
        # If alpha is 5, we want to hit sets of size 5.
        
        if alpha <= 2:
             # Just pick any valid candidate
             t = random.choice(valid_candidates)
        else:
             # Sample some independent sets of size alpha (or alpha-1 if alpha is hard?)
             # Let's try to get a few sets of size alpha
             _, sample_sets = calculate_independence_number_and_sets(selected_triples, k_target=alpha, max_sets=50)
             
             if not sample_sets:
                 # Fallback: try alpha - 1
                 _, sample_sets = calculate_independence_number_and_sets(selected_triples, k_target=alpha-1, max_sets=50)
             
             if not sample_sets:
                 t = random.choice(valid_candidates)
             else:
                 # Score each candidate by how many sample sets it intersects (is contained in)
                 best_t = None
                 best_score = -1
                 
                 # To speed up, we can precompute masks for sample sets
                 sample_masks = []
                 for s in sample_sets:
                     m = 0
                     for p in s:
                         m |= (1 << p)
                     sample_masks.append(m)
                     
                 for t in valid_candidates:
                     tm = triple_masks_map[t]
                     score = 0
                     for sm in sample_masks:
                         if (tm & sm) == tm:
                             score += 1
                     if score > best_score:
                         best_score = score
                         best_t = t
                 
                 if best_t is None:
                     best_t = random.choice(valid_candidates)
                 t = best_t
        
        # Add t
        selected_triples.append(list(t))
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
        for p in pairs:
            used_pairs.add(p)
            
    return selected_triples

def construct():
    best_triples = []
    best_score = 17
    
    # Run multiple attempts with different seeds
    for seed in range(20):
        triples = greedy_construct_optimized(seed)
        # Calculate final score
        score, _ = calculate_independence_number_and_sets(triples, k_target=None, max_sets=0)
        if score < best_score:
            best_score = score
            best_triples = triples
            
    return best_triples
