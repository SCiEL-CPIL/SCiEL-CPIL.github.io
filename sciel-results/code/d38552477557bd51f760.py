import sys

def construct():
    # Optimized Beam Search
    K = 3000  # Increased beam width
    
    # State: (seq_tuple, prod_set, suf_prods_list)
    # Start with empty sequence
    beam = [ ((), set(), []) ]
    
    best_len = 0
    best_seq = ()
    
    for x in range(2, 129):
        new_beam = []
        
        for seq, prods, sufs in beam:
            # Option 1: Exclude x
            # Prune if even adding all remaining numbers won't beat best_len
            if len(seq) + (128 - x) < best_len:
                continue
            new_beam.append((seq, prods, sufs))
            
            # Option 2: Include x
            # Fast check: if x is in prods, collision
            if x in prods:
                continue
                
            # Calculate new tails
            new_tails = [x]
            for p in sufs:
                new_tails.append(x * p)
                
            # Check collision for other tails
            collision = False
            for t in new_tails[1:]:
                if t in prods:
                    collision = True
                    break
            
            if not collision:
                new_prods = prods.union(new_tails)
                new_sufs = [x] + [x * p for p in sufs]
                new_seq = seq + (x,)
                
                if len(new_seq) > best_len:
                    best_len = len(new_seq)
                    best_seq = new_seq
                    
                new_beam.append((new_seq, new_prods, new_sufs))
        
        if not new_beam:
            break
            
        # Sort by length descending, then last element ascending, then sum of last 3
        def sort_key(s):
            seq = s[0]
            last = seq[-1] if seq else 0
            sum_last3 = sum(seq[-3:]) if len(seq) >= 3 else sum(seq)
            return (-len(seq), last, sum_last3)
            
        new_beam.sort(key=sort_key)
        
        # Keep top K
        beam = new_beam[:K]
        
        # Early stopping
        max_current_len = max(len(s[0]) for s in beam)
        if max_current_len + (128 - x) < best_len:
            break

    return list(best_seq)
