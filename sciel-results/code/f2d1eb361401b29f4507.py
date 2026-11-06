import time

# EVOLVE-BLOCK-START
def construct():
    # Primes up to 128
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
    num_primes = len(primes)
    
    # Precompute exponent vectors for each number 2..128
    vecs = {}
    for x in range(2, 129):
        v = [0] * num_primes
        n = x
        for i, p in enumerate(primes):
            while n % p == 0:
                v[i] += 1
                n //= p
        vecs[x] = tuple(v)
        
    def try_add(seq, prods, x):
        """
        Tries to add x to the end of seq.
        Returns (success, new_prods_set)
        """
        x_vec = vecs[x]
        new_suffixes = [x_vec]
        running_prod = x_vec
        
        for v in reversed(seq):
            running_prod = tuple(a + b for a, b in zip(running_prod, v))
            new_suffixes.append(running_prod)
            
        # Check collision
        for p in new_suffixes:
            if p in prods:
                return False, None
                
        # Create new set
        new_prods = set(prods)
        new_prods.update(new_suffixes)
        return True, new_prods

    best_seq = []
    best_len = 0
    start_time = time.time()
    
    # Beam Search
    # Start with a reasonable K
    K_values = [50, 100, 200, 500, 1000, 2000, 5000]
    
    for K in K_values:
        if time.time() - start_time > 80:
            break
            
        beam = [([], set())]
        
        for x in range(2, 129):
            new_beam = []
            for seq, prods in beam:
                # Option 1: Skip x
                new_beam.append((seq, prods))
                
                # Option 2: Add x
                success, new_prods = try_add(seq, prods, x)
                if success:
                    new_seq = seq + [x]
                    new_beam.append((new_seq, new_prods))
            
            # Sort by length descending
            new_beam.sort(key=lambda s: len(s[0]), reverse=True)
            
            # Keep top K
            beam = new_beam[:K]
            
        # Update best
        if beam:
            current_best = max(beam, key=lambda s: len(s[0]))
            if len(current_best[0]) > best_len:
                best_len = len(current_best[0])
                best_seq = current_best[0]
                
    return best_seq
# EVOLVE-BLOCK-END
