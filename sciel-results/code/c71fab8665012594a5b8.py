import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # State: (sequence_tuple, all_products_set, suffix_products_list)
    # We store suffix products as a list to allow quick multiplication by x.
    # Actually, suffix products of [a,b,c] are [c, bc, abc].
    # If we append x, new suffix products are [x, xc, xcb, xcba].
    # So we just need the old suffix products to generate new ones.
    
    initial_beam = ((), set(), [])
    beams = [initial_beam]
    best_seq = ()
    best_len = 0
    
    # We will expand beams.
    # To avoid infinite loops or useless expansion, we stop if no beam grows in a full iteration?
    # Or just until time runs out.
    
    beam_width = 5000
    
    while time.time() < deadline:
        new_beams = []
        for seq, prods, suffixes in beams:
            last = seq[-1] if seq else 1
            # Try all x > last
            for x in range(last + 1, 129):
                # Check time
                if time.time() > deadline:
                    break
                
                # Calculate new suffix products
                # Old suffixes: [s_k, s_{k-1}s_k, ..., s_1...s_k]
                # New suffixes: [x, x*s_k, x*s_{k-1}s_k, ...]
                new_suffixes = [x]
                for p in suffixes:
                    new_suffixes.append(x * p)
                
                # Check for collisions
                # New products must not be in old prods
                # And new products are distinct (guaranteed if old suffixes distinct and x>0)
                # So just check intersection with old prods.
                if not set(new_suffixes).intersection(prods):
                    new_prods = prods.union(new_suffixes)
                    new_seq = seq + (x,)
                    new_len = len(new_seq)
                    
                    if new_len > best_len:
                        best_len = new_len
                        best_seq = new_seq
                        
                    new_beams.append((new_seq, new_prods, new_suffixes))
        
        if not new_beams:
            break
            
        # Select top beam_width beams
        # Sort by length desc, then sum asc
        # To make it faster, we can bucket by length?
        # Given max length is small (<30), sorting is fine.
        
        # Optimization: if we have many beams, sorting 5000*100 items might be slow.
        # We can use heapq.nlargest?
        # Key: (len, -sum) -> we want max len, min sum.
        # So sort key: (len, -sum) and take largest?
        # Or sort by (-len, sum) and take smallest?
        
        # Let's just sort.
        new_beams.sort(key=lambda x: (len(x[0]), -sum(x[0])), reverse=True)
        # Wait, reverse=True on tuple (len, -sum).
        # We want highest len. Then lowest sum (highest -sum).
        # So sorting by (len, -sum) in descending order works.
        
        beams = new_beams[:beam_width]
        
        # If all beams are of max possible length or no growth, we might want to stop?
        # Actually, if no beam in new_beams is longer than the max length in beams, 
        # we might be stuck in a loop of same length?
        # But we keep top W. If we can't extend any, new_beams is empty, loop breaks.
        
    return list(best_seq)
