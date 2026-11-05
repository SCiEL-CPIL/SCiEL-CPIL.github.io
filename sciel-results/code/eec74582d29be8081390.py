import time

def construct():
    start_time = time.time()
    # Use 85s to be safe within the 90s limit
    deadline = start_time + 85.0

    # Beam Search
    # State: (last_picked, seq_list, prod_set)
    
    beam = []
    for x in range(2, 129):
        beam.append((x, [x], {x}))
        
    best_seq = []
    best_len = 0
    
    for _, seq, _ in beam:
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
            
    beam_width = 5000
    
    while time.time() < deadline - 5.0:
        next_beam = []
        
        for last_picked, seq, prods in beam:
            # Optimization: if len(seq) + (128 - last_picked) <= best_len, skip?
            # No, this is a lower bound on the *possible* length, not a guarantee.
            # But if we are in Beam Search, we keep the most promising.
            # The pruning `len(seq) + (128 - last_picked) <= best_len` is valid for DFS.
            # In Beam Search, if a state *cannot* beat best_len, we can drop it.
            if len(seq) + (128 - last_picked) <= best_len:
                continue
                
            for x in range(last_picked + 1, 129):
                if x in prods:
                    continue
                
                valid = True
                prod = x
                new_prods = [x]
                for a in reversed(seq):
                    prod *= a
                    new_prods.append(prod)
                    if prod in prods:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                next_beam.append((x, seq + [x], prods.union(new_prods)))
                
        if not next_beam:
            break
            
        if len(next_beam) > beam_width:
            # Sort by last_picked (smaller is better), then by len(prods) (smaller is better)
            next_beam.sort(key=lambda s: (s[0], len(s[2])))
            next_beam = next_beam[:beam_width]
            
        beam = next_beam
        
        current_len = len(beam[0][1])
        if current_len > best_len:
            best_len = current_len
            # Update best_seq
            # Pick the one with smallest last_picked?
            # beam[0] is the best sorted one.
            best_seq = beam[0][1][:]
            
    return best_seq
