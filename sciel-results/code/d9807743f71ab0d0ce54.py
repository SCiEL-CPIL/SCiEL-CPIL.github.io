import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 90s limit
    
    # 1. Run Greedy to get a good initial lower bound
    def run_greedy():
        values, products = [], set()
        for x in range(2, 129):
            tails, prod = [x], x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        return values, products
    
    best_seq, best_prods = run_greedy()
    best_len = len(best_seq)
    
    # 2. Beam Search
    # State: seq_tuple -> prods_set
    # We start with the empty state
    beam = {(): set()}
    
    # Beam width parameter
    # Inspiration 1 used 300. Let's try 600 to see if it helps within time limits.
    BEAM_WIDTH = 600
    
    for x in range(2, 129):
        if time.time() > deadline:
            break
            
        next_beam = {}
        
        for seq, prods in beam.items():
            # Option 1: Skip x
            # The state (seq, prods) remains valid.
            # We add it to next_beam.
            # If it's already there (from another path? No, paths are unique by seq), 
            # we just keep it.
            if seq not in next_beam:
                next_beam[seq] = prods
            else:
                # This shouldn't happen if we deduplicate by seq in the beam
                pass
            
            # Option 2: Include x
            # Since we iterate x in increasing order, and seq elements are < x (from previous steps),
            # seq + (x,) is strictly increasing.
            
            # Compute tails
            # tails = [x, x*seq[-1], x*seq[-1]*seq[-2], ...]
            tails = [x]
            prod = x
            for a in reversed(seq):
                prod *= a
                tails.append(prod)
                
            # Check collision
            # Optimization: Early exit
            has_collision = False
            for t in tails:
                if t in prods:
                    has_collision = True
                    break
            
            if not has_collision:
                new_seq = seq + (x,)
                new_prods = set(prods)
                new_prods.update(tails)
                
                # Add to next_beam
                # new_seq is unique for this step because x is unique and we append it.
                # It cannot collide with an existing key in next_beam unless 
                # we had duplicate seqs in the input beam, which we prevent.
                next_beam[new_seq] = new_prods
        
        # Prune beam
        # Sort by length desc, then by sum of elements asc
        # Rationale: Longer is better. If tied, smaller sum might be better 
        # (heuristic from Inspiration 1).
        
        if not next_beam:
            break
            
        sorted_seqs = sorted(next_beam.keys(), key=lambda s: (-len(s), sum(s) if s else 0))
        
        # Update best
        if sorted_seqs:
            top_seq = sorted_seqs[0]
            if len(top_seq) > best_len:
                best_len = len(top_seq)
                best_seq = list(top_seq)
                best_prods = next_beam[top_seq]
                
        # Keep top W
        beam = {s: next_beam[s] for s in sorted_seqs[:BEAM_WIDTH]}
        
    return list(best_seq)
