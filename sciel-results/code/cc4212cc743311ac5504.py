import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Primes up to 127
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    num_primes = len(primes)
    
    # Calculate shifts for packing vectors into a single integer
    # We need enough bits for the maximum possible exponent in a product of length 127
    # Max exponent for p_i is 127 * floor(log_p_i(128))
    shifts = [0] * num_primes
    current_shift = 0
    for i, p in enumerate(primes):
        max_exp = 127
        # More precise max exp:
        # 2: 127*7=889 -> 10 bits
        # 3: 127*4=508 -> 9 bits
        # 5: 127*2=254 -> 8 bits
        # 7: 127*1=127 -> 7 bits
        # others: 127*1=127 -> 7 bits
        if p == 2:
            bits = 10
        elif p == 3:
            bits = 9
        elif p == 5:
            bits = 8
        elif p == 7:
            bits = 7
        else:
            bits = 7
            
        shifts[i] = current_shift
        current_shift += bits
        
    # Precompute packed vectors for each number
    def get_packed_vec(n):
        packed = 0
        temp = n
        for i, p in enumerate(primes):
            if temp % p == 0:
                exp = 0
                while temp % p == 0:
                    exp += 1
                    temp //= p
                packed |= (exp << shifts[i])
        return packed
        
    packed_vecs = {i: get_packed_vec(i) for i in range(2, 129)}
    
    # Beam Search
    # State: (seq, prefix_packs, prod_set)
    # prefix_packs: list of packed ints [P0, P1, ..., PL]
    # prod_set: set of packed ints representing all block products
    
    beam = [([], [0], set())]
    beam_width = 200 # Increased width due to speedup
    
    for n in range(2, 129):
        if time.time() > deadline:
            break
            
        new_beam = []
        v_n = packed_vecs[n]
        
        for seq, ppacks, pset in beam:
            # Option 1: Exclude n
            # State remains the same
            new_beam.append((seq, ppacks, pset))
            
            # Option 2: Include n
            last_ppack = ppacks[-1]
            new_last_ppack = last_ppack + v_n
            
            valid = True
            new_prods = []
            
            # Check all suffixes ending at n
            # Products are new_last_ppack - ppacks[i]
            for i in range(len(ppacks)):
                diff = new_last_ppack - ppacks[i]
                # diff is non-negative because ppacks are cumulative sums of non-negative vectors
                if diff in pset:
                    valid = False
                    break
                new_prods.append(diff)
            
            if valid:
                new_seq = seq + [n]
                new_ppacks = ppacks + [new_last_ppack]
                new_pset = pset.union(new_prods)
                new_beam.append((new_seq, new_ppacks, new_pset))
        
        # Select top beam_width states
        # Sort by length of seq (descending)
        new_beam.sort(key=lambda x: len(x[0]), reverse=True)
        
        # Keep top W
        beam = new_beam[:beam_width]
        
        if beam and len(beam[0][0]) == 127:
            return beam[0][0]

    if beam:
        return beam[0][0]
    else:
        return []
