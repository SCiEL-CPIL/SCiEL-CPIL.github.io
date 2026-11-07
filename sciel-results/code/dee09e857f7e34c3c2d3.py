import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    def factorize(n):
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors

    all_nums = list(range(2, 41))
    exp_vectors = {}
    for n in all_nums:
        f = factorize(n)
        vec = [0] * num_primes
        for p, e in f.items():
            vec[prime_to_idx[p]] = e
        exp_vectors[n] = tuple(vec)
        
    zero_vec = tuple([0] * num_primes)
    
    # Precompute sum of vectors for faster addition?
    # Since dimension is small, tuple addition is okay.
    
    def add_vecs(a, b):
        return tuple(a[i] + b[i] for i in range(num_primes))

    best_set = []
    best_size = 0
    
    # Initial heuristic: try to build a large set greedily
    # Sort by "complexity" or just random
    
    for _ in range(10000): # Upper bound on iterations, will break on time
        if time.time() > deadline:
            break
            
        random.shuffle(all_nums)
        current_set = []
        current_sums = {zero_vec}
        
        for n in all_nums:
            v = exp_vectors[n]
            # Check if v can be added
            # Condition: No s in current_sums such that s + v is in current_sums
            # This ensures all new sums (s+v) are distinct from old sums (s)
            # And since old sums were distinct, new sums are distinct from each other?
            # If s1 + v == s2 + v => s1 == s2. Yes.
            # So we just need to check intersection of current_sums and (current_sums + v)
            
            valid = True
            # Optimization: if len(current_sums) is small, this is fast
            for s in current_sums:
                if add_vecs(s, v) in current_sums:
                    valid = False
                    break
                    
            if valid:
                current_set.append(n)
                # Update current_sums
                # New sums are old sums + v
                # We create a list of new sums and add to set
                new_sums = [add_vecs(s, v) for s in current_sums]
                current_sums.update(new_sums)
                
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
                    
                    # If we reached a high count, we can try to break early if time is tight?
                    # No, keep searching for max.
                    
    # Final check to ensure best_set is valid (paranoia)
    # And convert to list of ints
    if best_set:
        # Verify best_set one last time
        # Reconstruct sums for best_set
        # This is just a safety check
        pass 
        
    return [int(x) for x in best_set]
