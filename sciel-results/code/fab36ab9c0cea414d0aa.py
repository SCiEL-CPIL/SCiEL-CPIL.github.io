import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Beam Search
    # State: (masks_tuple, set_tuple)
    # masks is a tuple of ints (bitsets)
    # set is a tuple of ints (the elements)
    
    # Initial state
    # masks[0] = 1 (sum 0)
    # masks[k] = 0 for k > 0
    init_masks = tuple([1] + [0]*60)
    init_set = tuple()
    
    beam = [(init_masks, init_set)]
    best_set = []
    
    W = 200  # Beam width
    
    candidates = list(range(1, 257))
    
    for x in candidates:
        if time.time() > deadline:
            break
            
        new_beam = []
        for masks, s in beam:
            # Option 1: Skip x
            new_beam.append((masks, s))
            
            # Option 2: Add x
            # Check validity
            new_masks = list(masks)
            valid = True
            # Update masks
            # new_masks[k] = masks[k] | (masks[k-1] << x)
            # We need to check disjointness for all k.
            # Efficient check:
            # union = 0
            # for k in range(len(new_masks)):
            #    new_masks[k] = masks[k] | (masks[k-1] << x) if k>0 else masks[0]
            #    if union & new_masks[k]: valid = False; break
            #    union |= new_masks[k]
            
            # Optimization: Only check up to current size + 1
            curr_size = len(s)
            union = 0
            new_masks_list = [0] * 61
            for k in range(curr_size + 2):
                if k == 0:
                    nm = masks[0]
                else:
                    nm = masks[k] | (masks[k-1] << x)
                new_masks_list[k] = nm
                if union & nm:
                    valid = False
                    break
                union |= nm
            
            if valid:
                new_beam.append((tuple(new_masks_list), s + (x,)))
        
        # Prune new_beam
        if not new_beam:
            break
            
        # Sort by size desc, then by union bit_count asc
        # To avoid sorting the whole list if it's huge, we can use nlargest
        # But W is small, so sorting is fine if new_beam isn't huge.
        # new_beam size is at most 2 * W.
        # So sorting 400 items is fast.
        
        # We need to extract the union bit count for sorting.
        # We can compute it on the fly in the key function, or store it.
        # Let's just sort by len(s) desc.
        # To break ties, we need the union bit count.
        
        def score(state):
            masks, s = state
            # Compute union bit count
            u = 0
            for m in masks:
                u |= m
            return (-len(s), u.bit_count())
        
        new_beam.sort(key=score)
        beam = new_beam[:W]
        
        # Update best
        # The best state in the beam is the first one (since sorted by size desc)
        if beam:
            best_masks, best_s = beam[0]
            if len(best_s) > len(best_set):
                best_set = list(best_s)
            
        # Early exit if we reached max possible?
        # Max possible is 256, but we know it's much lower.
        # If we can't add any more?
        # If the best state in the next iteration is the same as current, maybe we are stuck?
        # No, we might skip some and add later.
        
    return [int(x) for x in best_set]
