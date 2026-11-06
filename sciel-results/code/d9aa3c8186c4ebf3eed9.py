import time

def _compute_products(seq):
    products = set()
    n = len(seq)
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            products.add(prod)
    return products

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    beam_width = 500 # Adjust based on speed
    best_seq = []
    best_len = 0
    
    # State: (sequence, product_set)
    # Start with empty
    beam = [([], set())]
    
    for v in range(2, 129):
        new_beam = []
        for seq, prods in beam:
            # Check if adding v is valid
            # New products are:
            # 1. v
            # 2. v * a_i for all suffixes of seq? No.
            # Consecutive blocks ending at v:
            # v
            # v * seq[-1]
            # v * seq[-1] * seq[-2]
            # ...
            # v * prod(seq)
            
            # We need to check if any of these new products are in `prods` or in each other.
            # Note: products within the new suffixes are unique if v is coprime? No.
            # But the products of the new suffixes are distinct from each other because they are multiples of v by distinct suffix products of seq.
            # Wait, if seq has suffix products S1, S2, ..., Sk (where Sk = prod(seq)),
            # new products are v*S1, v*S2, ..., v*Sk.
            # Are v*S_i distinct? Yes, because S_i are distinct (since seq elements > 1, suffix products are strictly increasing in magnitude? No, but distinct).
            # Actually, suffix products of a strictly increasing sequence of integers > 1 are distinct.
            # So the new products are distinct from each other.
            # We just need to check if any new product is in `prods`.
            
            new_prods = set(prods)
            suffix_prod = 1
            # Iterate backwards to generate suffix products
            # The new consecutive blocks ending at v are:
            # [v], [seq[-1], v], [seq[-2], seq[-1], v], ...
            
            # Let's generate the suffix products of seq.
            # Suffix product of seq[k:] is P_k.
            # New products are v * P_k for all k, plus v itself (which is v * 1).
            
            # We can compute these on the fly.
            valid = True
            # Check v
            if v in new_prods:
                valid = False
            else:
                new_prods.add(v)
                
            # Check v * suffix_prod for each suffix
            # Suffixes: seq[-1:], seq[-2:], ...
            # We need to be careful not to recompute everything from scratch if possible, 
            # but given the small size, it's okay.
            
            # Let's just compute the full set of new products and check intersection.
            # For efficiency, let's just do the check.
            
            # Optimization: 
            # The new products are { v * prod(seq[i:]) for i in range(len(seq)) } U {v}
            # Wait, prod(seq[i:]) includes the case i=len(seq) which is 1? No, range(len) goes 0..n-1.
            # Suffix starting at i: seq[i], ..., seq[n-1].
            # Product is P_i.
            # New block is seq[i], ..., seq[n-1], v. Product is P_i * v.
            # Also the block [v] has product v.
            
            # So new products = { v * P_i for i in 0..n-1 } U { v }
            
            # We need to check if any of these are in `prods`.
            
            # Let's generate them.
            current_suffix_prod = 1
            # We need P_i. 
            # P_{n-1} = seq[n-1]
            # P_{n-2} = seq[n-2] * seq[n-1]
            # ...
            
            # We can iterate backwards.
            # But we also need to add them to new_prods to check against each other?
            # As established, they are distinct.
            # So we just check against `prods`.
            
            # However, we also need to update `new_prods` for the next steps?
            # Yes, the state for the next step needs the full product set.
            
            # Let's compute the list of new products.
            new_products_to_add = [v]
            suffix_prod = 1
            for i in range(len(seq)-1, -1, -1):
                suffix_prod *= seq[i]
                new_products_to_add.append(suffix_prod * v)
                
            # Check intersection with existing prods
            if any(p in prods for p in new_products_to_add):
                valid = False
            else:
                # Update new_prods
                new_prods.update(new_products_to_add)
                
            if valid:
                new_seq = seq + [v]
                new_beam.append((new_seq, new_prods))
                
        if not new_beam:
            break
            
        # Sort and truncate
        # Heuristic: Prefer longer sequences.
        # If tie, prefer sequences with "simpler" product sets? 
        # Or just keep the first ones?
        # Let's sort by length descending.
        new_beam.sort(key=lambda x: -len(x[0]))
        
        # To maintain diversity, we might want to avoid keeping too many similar sequences.
        # But simple truncation is a good start.
        
        beam = new_beam[:beam_width]
        
        # Update best
        if beam:
            current_best = beam[0][0]
            if len(current_best) > best_len:
                best_seq = current_best
                best_len = best_len
        
        if time.time() - start_time > time_limit:
            break
            
    return best_seq
