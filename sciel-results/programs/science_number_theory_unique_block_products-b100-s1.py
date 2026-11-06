import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Primes up to 127
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    num_primes = len(primes)
    
    # Precompute factor vectors
    def get_vec(n):
        v = [0] * num_primes
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
        return tuple(v)
        
    vecs = {i: get_vec(i) for i in range(2, 129)}
    zero_vec = (0,) * num_primes

    # Global ID registry for vectors to speed up set operations
    vec_to_id = {zero_vec: 0}
    next_id = 1
    
    def get_id(vec):
        nonlocal next_id
        if vec in vec_to_id:
            return vec_to_id[vec]
        vec_to_id[vec] = next_id
        next_id += 1
        return vec_to_id[vec]

    # Beam Search
    # State: (seq, prefix_vecs, prod_ids)
    # seq: list of ints
    # prefix_vecs: list of tuples (V0, V1, ..., Vk)
    # prod_ids: set of int IDs representing all block products
    
    # Initial state
    beam = [([], [zero_vec], {0})] # Wait, product of empty block? 
    # The problem says "nonempty consecutive block".
    # So initial product set is empty.
    # But prefix V0 is zero.
    # If we add n, product is V1 - V0 = v_n.
    # So initial prod_ids should be empty set().
    beam = [([], [zero_vec], set())]
    
    beam_width = 300 # Increased width due to faster set ops
    
    for n in range(2, 129):
        if time.time() > deadline:
            break
            
        new_beam = []
        v_n = vecs[n]
        
        for seq, pvecs, pset in beam:
            # Option 1: Exclude n
            # We keep the state as is.
            # To avoid duplicates, we rely on the fact that if two paths lead to the same seq, 
            # they are identical states. But beam search might have duplicates?
            # We sort and take top K, so duplicates are handled if we deduplicate.
            # However, "Exclude n" from two different parents might lead to same seq?
            # No, the parent seq determines the child seq.
            # If parent A and parent B have same seq, they are the same state.
            # So we just add.
            new_beam.append((seq, pvecs, pset))
            
            # Option 2: Include n
            last_pvec = pvecs[-1]
            # Vector addition
            v_new_list = [last_pvec[i] + v_n[i] for i in range(num_primes)]
            v_new = tuple(v_new_list)
            
            valid = True
            new_prod_ids = []
            
            # Check all suffixes ending at n
            # Products are v_new - pvecs[i]
            for i in range(len(pvecs)):
                diff_list = [v_new[j] - pvecs[i][j] for j in range(num_primes)]
                diff_tup = tuple(diff_list)
                diff_id = get_id(diff_tup)
                
                if diff_id in pset:
                    valid = False
                    break
                
                new_prod_ids.append(diff_id)
            
            if valid:
                new_seq = seq + [n]
                new_pvecs = pvecs + [v_new]
                # Update pset
                # Using set union with list of ints is fast
                new_pset = pset.union(new_prod_ids)
                new_beam.append((new_seq, new_pvecs, new_pset))
        
        # Select top beam_width states from new_beam
        # Sort by length of seq (descending)
        new_beam.sort(key=lambda x: len(x[0]), reverse=True)
        
        # Deduplicate? 
        # If two states have the same seq, they have the same pvecs and pset.
        # We can use a set of tuples to deduplicate, but seq is a list.
        # We can just keep them. The sort will group them.
        # Taking top W will pick the best ones.
        
        beam = new_beam[:beam_width]
        
        # Early termination
        if beam and len(beam[0][0]) == 127:
            return beam[0][0]

    # Return best found
    if beam:
        return beam[0][0]
    else:
        return []
