import time
import copy

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

    # Beam Search
    # State: (seq, prefix_vecs, prod_set)
    # We want to maximize len(seq)
    
    # Initial state
    # seq: []
    # prefix_vecs: [zero_vec] (V0)
    # prod_set: set()
    
    beam = [([], [zero_vec], set())]
    beam_width = 100 # Tune this if needed, 100 is safe
    
    for n in range(2, 129):
        if time.time() > deadline:
            break
            
        new_beam = []
        v_n = vecs[n]
        
        for seq, pvecs, pset in beam:
            # Option 1: Exclude n
            # We keep the state as is.
            # To avoid duplicates in beam, we might need to deduplicate, but states are unique by seq.
            # However, if two paths lead to the same seq, they have same pset.
            # We can just add it.
            new_beam.append((seq, pvecs, pset))
            
            # Option 2: Include n
            # Check validity
            # New prefix: V_new = pvecs[-1] + v_n
            # New products: V_new - pvecs[i] for i in 0..len(pvecs)-1
            
            # Optimization: We can check conflicts as we go.
            # But we need to construct the new pset to add to beam.
            
            # Let's calculate new products first to see if valid.
            # We need V_new.
            last_pvec = pvecs[-1]
            # Vector addition
            v_new_list = [last_pvec[i] + v_n[i] for i in range(num_primes)]
            v_new = tuple(v_new_list)
            
            valid = True
            new_prods = []
            
            # Check all suffixes ending at n
            # The products are v_new - pvecs[i]
            for i in range(len(pvecs)):
                # diff = v_new - pvecs[i]
                diff_list = [v_new[j] - pvecs[i][j] for j in range(num_primes)]
                diff_tup = tuple(diff_list)
                
                if diff_tup in pset:
                    valid = False
                    break
                # We also need to ensure new_prods are distinct among themselves?
                # If diff_tup1 == diff_tup2 for i1 != i2, then:
                # V_new - V_i1 = V_new - V_i2 => V_i1 = V_i2.
                # But prefix vectors in a valid sequence are distinct?
                # If V_i1 = V_i2 with i1 < i2, then product of block (i1+1..i2) is 0 vector (value 1).
                # But all numbers are >= 2, so products are >= 2.
                # So prefix vectors must be distinct.
                # Thus new_prods are automatically distinct if pvecs are distinct.
                # Are pvecs distinct? Yes, because all elements > 1, so vectors strictly increase in some component?
                # Not necessarily strictly increase in every component, but the sum of positive vectors is unique?
                # Actually, if we have a valid sequence so far, all block products are distinct and > 1.
                # This implies all prefix differences are distinct and non-zero.
                # This implies prefix vectors are distinct.
                # So new_prods are distinct.
                
                new_prods.append(diff_tup)
            
            if valid:
                # Create new state
                new_seq = seq + [n]
                new_pvecs = pvecs + [v_new]
                # Update pset
                # pset is a set. We can create a new set by copying and adding.
                # Copying a set of size L^2 is O(L^2).
                # This is the bottleneck.
                # For L=100, size=5000. Copying 5000 tuples * 100 beams * 127 steps = 63 million ops.
                # This might be slow in Python.
                
                # Optimization: Use a persistent data structure? Or just accept the cost?
                # 63M simple tuple copies/hashes might take ~5-10 seconds.
                # We have 85s budget. It should be fine.
                
                new_pset = pset.union(new_prods)
                new_beam.append((new_seq, new_pvecs, new_pset))
        
        # Select top beam_width states from new_beam
        # Sort by length of seq (descending)
        # If tie, we can just keep them.
        # To keep diversity, maybe we should not just sort by length?
        # But length is the objective.
        
        new_beam.sort(key=lambda x: len(x[0]), reverse=True)
        
        # Keep top W
        beam = new_beam[:beam_width]
        
        # Early termination if all beams have length 127? No, max possible is 127.
        # If we have a beam of length 127, we can return it immediately?
        # No, we must check if it's valid (it is, by construction).
        # But can we get 127?
        # If we do, we can return.
        
        if beam and len(beam[0][0]) == 127:
            return beam[0][0]

    # Return best found
    if beam:
        # The beam is already sorted by length
        return beam[0][0]
    else:
        return []
