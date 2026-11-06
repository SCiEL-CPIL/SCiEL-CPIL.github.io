import sys

def construct():
    # Beam search
    K = 100  # Beam width
    
    # State: (seq_tuple, prod_set, suf_prods_list)
    # Start with empty sequence
    beam = [ ((), set(), []) ]
    
    best_len = 0
    best_seq = ()
    
    for x in range(2, 129):
        new_beam = []
        
        for seq, prods, sufs in beam:
            # Option 1: Exclude x
            # We can prune if even adding all remaining numbers won't beat best_len
            # Max possible length from this state: len(seq) + (128 - x)
            if len(seq) + (128 - x) < best_len:
                continue
            new_beam.append((seq, prods, sufs))
            
            # Option 2: Include x
            # Calculate new tails
            # New tails are x, x*sufs[0], x*sufs[1], ...
            # sufs[0] is the last element of seq.
            # If seq is empty, sufs is empty, new tails is [x]
            
            new_tails = [x]
            for p in sufs:
                new_tails.append(x * p)
                
            # Check collision
            # If any new tail is in prods, collision
            # Note: new_tails are distinct from each other because x > 1 and sufs are distinct multipliers?
            # sufs are products of suffixes. Since seq is strictly increasing, sufs are strictly increasing?
            # s_k > s_{k-1} > ...
            # s_k > s_{k-1}s_k? No.
            # Example: 2, 3. sufs = [3, 6].
            # x=4. new_tails = [4, 12, 24]. Distinct.
            # Generally, since x >= 2, multiplying by x increases the value.
            # So new_tails are distinct.
            
            if not prods.intersection(new_tails):
                new_prods = prods.union(new_tails)
                new_sufs = [x] + [x * p for p in sufs]
                new_seq = seq + (x,)
                
                if len(new_seq) > best_len:
                    best_len = len(new_seq)
                    best_seq = new_seq
                    
                new_beam.append((new_seq, new_prods, new_sufs))
        
        # Prune new_beam
        if not new_beam:
            break
            
        # Sort by length descending
        # Tie-breaker: let's use the sum of the sequence ascending?
        # Or just keep the first K.
        # To ensure we explore diverse paths, maybe randomize?
        # No, deterministic is better for reproducibility.
        # Let's sort by length desc, then by the last element asc?
        
        new_beam.sort(key=lambda s: (-len(s[0]), s[0][-1] if s[0] else 0))
        
        # Keep top K
        beam = new_beam[:K]
        
        # Early stopping if we can't beat best_len
        # Max possible length in beam: max(len(s[0]) for s in beam)
        # If max_len + (128 - x) < best_len, we can stop?
        # No, best_len is updated in the loop.
        # But if the max length in the beam + remaining numbers < best_len, we can break.
        max_current_len = max(len(s[0]) for s in beam)
        if max_current_len + (128 - x) < best_len:
            break

    return list(best_seq)
