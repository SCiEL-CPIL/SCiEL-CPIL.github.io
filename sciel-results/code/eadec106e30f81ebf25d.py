import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Greedy Baseline
    def get_greedy():
        seq = []
        prods = set()
        sufs = [1]
        for x in range(2, 129):
            # Check collision
            new_prods = [x * p for p in sufs]
            if not prods.intersection(new_prods):
                seq.append(x)
                prods.update(new_prods)
                # Update sufs: new suffixes are 1, x, x*s_k, ...
                # old sufs: 1, s_k, ...
                # new sufs: 1, x, x*s_k, ...
                # So new_sufs = [1] + new_prods
                sufs = [1] + new_prods
        return seq, prods, sufs

    greedy_seq, _, _ = get_greedy()
    best_seq = greedy_seq
    best_len = len(greedy_seq)

    # Beam Search
    # State: (seq_tuple, prods_set, sufs_list, last_val)
    # We use tuple for seq to make it hashable if we wanted dedup, but we just keep top K.
    # Actually, list is fine for seq if we don't dedup.
    
    beam = [ ([], set(), [1], 1) ]
    
    K = 200 # Beam width
    
    for x in range(2, 129):
        if time.time() > deadline:
            break
            
        new_beam = []
        
        for state in beam:
            seq, prods, sufs, last_val = state
            
            # Pruning: Max possible length
            # Current len + remaining numbers (128 - x + 1)
            if len(seq) + (129 - x) <= best_len:
                # Even if we take all remaining, we can't beat best_len
                # But we might tie? We want to maximize length.
                # If we can't beat, and we are only interested in strictly better, we can prune.
                # However, if we tie, we might want to keep it for secondary scores?
                # The task says "Maximize length". If lengths are equal, score is same?
                # "The evaluator measures the raw value". Value is likely length.
                # So we only care about strictly greater length.
                # But wait, if we prune a state that ties, we might miss a path that later beats it?
                # No, if len(seq) + remaining <= best_len, then max possible final length <= best_len.
                # So it cannot beat best_len. We can prune.
                continue

            # Option 1: Skip x
            # We add the current state to new_beam
            # Note: We are not modifying state, so we can just append it.
            new_beam.append(state)
            
            # Option 2: Take x
            if x > last_val:
                # Check collision
                # New products: x * p for p in sufs
                # sufs[0] is 1, so x*1 = x is the first new product.
                # We need to check if any x*p is in prods.
                
                # Optimization: Check intersection
                # Creating a list of new prods is O(L).
                # Intersection is O(min(len(prods), len(new_prods))).
                
                # We can early break if we find a collision.
                # But set intersection is implemented in C, so it's fast.
                # Let's create the list.
                new_prods = [x * p for p in sufs]
                
                if not prods.intersection(new_prods):
                    # Valid
                    new_seq = seq + [x]
                    new_prods_set = prods.union(new_prods)
                    new_sufs = [1] + new_prods
                    new_last = x
                    new_beam.append((new_seq, new_prods_set, new_sufs, new_last))
                    
                    # Update global best
                    if len(new_seq) > best_len:
                        best_len = len(new_seq)
                        best_seq = new_seq
        
        if not new_beam:
            break
            
        # Sort new_beam
        # Key: (-len, last_val)
        # We want longer sequences first.
        # Among same length, smaller last_val is better (more room).
        new_beam.sort(key=lambda s: (-len(s[0]), s[3]))
        
        # Keep top K
        beam = new_beam[:K]

    return best_seq
