import time
import itertools
import random

# Precompute masks for each popcount
# This is done once at module load
MasksByPopcount = {}
for r in range(17):
    masks = []
    # Optimization: Generate combinations directly?
    # itertools.combinations(range(16), r)
    # Then convert to mask.
    for combo in itertools.combinations(range(16), r):
        m = 0
        for p in combo:
            m |= (1 << p)
        masks.append(m)
    MasksByPopcount[r] = masks

def _independence_number(triple_masks):
    """
    triple_masks: list of integers, each representing a triple as a bitmask.
    Returns the size of the largest independent set.
    """
    if not triple_masks:
        return 16
    
    # Sort triple masks? 
    # If we sort by something, does it help? 
    # Maybe sort by the number of times they appear in random subsets? No.
    # Just use as is.
    
    for r in range(16, 0, -1):
        for mask in MasksByPopcount[r]:
            independent = True
            for tm in triple_masks:
                if (mask & tm) == tm:
                    independent = False
                    break
            if independent:
                return r
    return 0

def _construct_greedy(seed):
    random.seed(seed)
    V = list(range(16))
    all_triples = list(itertools.combinations(V, 3))
    random.shuffle(all_triples)
    
    used_pairs = set()
    triples = []
    triple_masks = []
    
    for t in all_triples:
        a, b, c = t
        mask = (1 << a) | (1 << b) | (1 << c)
        
        ok = True
        # Check linearity
        if (a, b) in used_pairs or (b, a) in used_pairs: ok = False
        elif (a, c) in used_pairs or (c, a) in used_pairs: ok = False
        elif (b, c) in used_pairs or (c, b) in used_pairs: ok = False
        
        if ok:
            triples.append([a, b, c])
            triple_masks.append(mask)
            used_pairs.add((a, b))
            used_pairs.add((b, a))
            used_pairs.add((a, c))
            used_pairs.add((c, a))
            used_pairs.add((b, c))
            used_pairs.add((c, b))
            
    return triples, triple_masks

def _local_search(triples, triple_masks, deadline):
    # Convert to sets for faster pair checking? 
    # We need to check linearity for new triples.
    # Precompute all possible triples and their masks.
    V = list(range(16))
    all_triples = list(itertools.combinations(V, 3))
    # Map triple tuple to mask
    triple_to_mask = {t: (1<<t[0])|(1<<t[1])|(1<<t[2]) for t in all_triples}
    
    # Current state
    current_triples = set(tuple(t) for t in triples)
    current_masks = list(triple_masks)
    
    # Precompute pairs used
    used_pairs = set()
    for t in current_triples:
        a, b, c = t
        used_pairs.add((a, b)); used_pairs.add((b, a))
        used_pairs.add((a, c)); used_pairs.add((c, a))
        used_pairs.add((b, c)); used_pairs.add((c, b))
        
    current_alpha = _independence_number(current_masks)
    
    # Local search loop
    no_improve_count = 0
    while time.time() < deadline:
        improved = False
        
        # Try to add a triple
        # Sample random triples
        for _ in range(20):
            if time.time() > deadline: break
            t_in = random.choice(all_triples)
            if t_in in current_triples:
                continue
                
            a, b, c = t_in
            if (a,b) in used_pairs or (b,a) in used_pairs or \
               (a,c) in used_pairs or (c,a) in used_pairs or \
               (b,c) in used_pairs or (c,b) in used_pairs:
                continue
                
            # It is linear. Try adding it.
            new_triples = set(current_triples)
            new_triples.add(t_in)
            new_masks = current_masks + [triple_to_mask[t_in]]
            
            new_alpha = _independence_number(new_masks)
            
            if new_alpha < current_alpha:
                current_triples = new_triples
                current_masks = new_masks
                used_pairs.add((a,b)); used_pairs.add((b,a))
                used_pairs.add((a,c)); used_pairs.add((c,a))
                used_pairs.add((b,c)); used_pairs.add((c,b))
                current_alpha = new_alpha
                improved = True
                no_improve_count = 0
                break
            else:
                no_improve_count += 1
                
        if improved:
            continue
            
        # Try to swap
        for _ in range(20):
            if time.time() > deadline: break
            t_out = random.choice(list(current_triples))
            t_in = random.choice(all_triples)
            if t_in == t_out:
                continue
                
            # Check if t_in is linear with current_triples - {t_out}
            # This is expensive. 
            # Optimization: Only try if t_in is "close" to being linear?
            # Or just check.
            
            # Check linearity against all triples except t_out
            linear = True
            a, b, c = t_in
            # Check pairs
            # If a pair of t_in is used by t_out, it might be free now.
            # But if it's used by another triple, it's blocked.
            
            # Simple check:
            for t_other in current_triples:
                if t_other == t_out:
                    continue
                if len(set(t_in) & set(t_other)) > 1:
                    linear = False
                    break
            
            if not linear:
                continue
                
            new_triples = set(current_triples)
            new_triples.remove(t_out)
            new_triples.add(t_in)
            new_masks = [triple_to_mask[t] for t in new_triples]
            
            new_alpha = _independence_number(new_masks)
            
            if new_alpha < current_alpha:
                current_triples = new_triples
                current_masks = new_masks
                # Rebuild used_pairs
                used_pairs = set()
                for t in current_triples:
                    x, y, z = t
                    used_pairs.add((x,y)); used_pairs.add((y,x))
                    used_pairs.add((x,z)); used_pairs.add((z,x))
                    used_pairs.add((y,z)); used_pairs.add((z,y))
                current_alpha = new_alpha
                improved = True
                no_improve_count = 0
                break
            else:
                no_improve_count += 1
                
        if not improved:
            no_improve_count += 1
            if no_improve_count > 100:
                # Restart local search? Or break?
                # Break to let outer loop restart with new seed
                break
                
    return list(current_triples), current_alpha

def construct():
    start = time.time()
    DEADLINE = 115.0
    
    best_alpha = 16
    best_sys = []
    
    # Initial greedy
    triples, masks = _construct_greedy(0)
    alpha = _independence_number(masks)
    if alpha < best_alpha:
        best_alpha = alpha
        best_sys = triples
        
    # Local search
    seed = 1
    while time.time() - start < DEADLINE:
        triples, masks = _construct_greedy(seed)
        # Refine
        refined_triples, refined_alpha = _local_search(triples, masks, DEADLINE)
        
        if refined_alpha < best_alpha:
            best_alpha = refined_alpha
            best_sys = refined_triples
            
            if best_alpha == 4:
                break
                
        seed += 1

    if not best_sys:
        best_sys = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        
    return best_sys
