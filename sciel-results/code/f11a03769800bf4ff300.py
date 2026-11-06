import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    num_primes = len(primes)
    
    # Calculate bit positions
    # p=2 (idx 0): 3 bits (0,1,2)
    # p=3 (idx 1): 3 bits (3,4,5)
    # p=5 (idx 2): 2 bits (6,7)
    # p>=7 (idx >=3): 1 bit each
    # idx 3 -> bit 8
    # idx 4 -> bit 9
    # ...
    # idx i -> bit 8 + (i - 3) = i + 5
    
    def get_bitmask(n):
        temp = n
        mask = 0
        for i, p in enumerate(primes):
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    count += 1
                    temp //= p
                if i == 0:
                    mask |= (count << 0)
                elif i == 1:
                    mask |= (count << 3)
                elif i == 2:
                    mask |= (count << 6)
                else:
                    mask |= (count << (i + 5))
        return mask

    bitmasks = {i: get_bitmask(i) for i in range(2, 129)}
    
    # Beam search
    # State: (seq_tuple, prefixes_tuple, prods_set)
    # prefixes_tuple[0] is always 0
    
    beam = [(((),), (0,), set())]
    
    beam_width = 50
    
    for n in range(2, 129):
        if time.time() > deadline:
            break
            
        new_beam = []
        v_n = bitmasks[n]
        
        for seq, pvecs, pset in beam:
            # Option 1: Exclude n
            new_beam.append((seq, pvecs, pset))
            
            # Option 2: Include n
            last_pvec = pvecs[-1]
            v_new = last_pvec + v_n
            
            valid = True
            new_prod_list = []
            
            # Check all suffixes ending at n
            # Products are v_new - pvecs[i]
            for i in range(len(pvecs)):
                diff = v_new - pvecs[i]
                
                if diff in pset:
                    valid = False
                    break
                
                new_prod_list.append(diff)
            
            if valid:
                new_seq = seq + (n,)
                new_pvecs = pvecs + (v_new,)
                # Update pset
                new_pset = pset.union(new_prod_list)
                new_beam.append((new_seq, new_pvecs, new_pset))
        
        # Select top beam_width states
        # Sort by length of seq (descending)
        # To break ties, we might want to prefer states with "smaller" products or something?
        # For now, just length.
        
        # Deduplicate
        # If two states have the same seq, they are identical.
        # We can use a set of seq tuples to deduplicate.
        seen_seqs = set()
        unique_new_beam = []
        
        # Sort first to ensure we keep the "best" duplicates if any logic differed, 
        # but here duplicates are identical states.
        new_beam.sort(key=lambda x: len(x[0]), reverse=True)
        
        for state in new_beam:
            if state[0] in seen_seqs:
                continue
            seen_seqs.add(state[0])
            unique_new_beam.append(state)
            if len(unique_new_beam) == beam_width:
                break
                
        beam = unique_new_beam
        
        # Early termination
        if beam and len(beam[0][0]) == 127:
            return list(beam[0][0])

    # Return best found
    if beam:
        return list(beam[0][0])
    else:
        return []
