import random
import time

def construct():
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    # Precompute exponent vectors
    vecs = {}
    for n in range(2, 41):
        v = [0] * num_primes
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
        vecs[n] = tuple(v)
        
    zero = (0,) * num_primes
    best_set = []
    best_len = 0
    
    # Initial good set: primes
    initial_set = primes[:]
    # Verify and compute sums for initial set
    sums = {zero}
    valid_initial = True
    for p in initial_set:
        v = vecs[p]
        new_sums = []
        for s in sums:
            ns = tuple(s[i] + v[i] for i in range(num_primes))
            if ns in sums:
                valid_initial = False
                break
            new_sums.append(ns)
        if not valid_initial:
            break
        sums.update(new_sums)
        
    if valid_initial:
        best_set = initial_set
        best_len = len(initial_set)

    start_time = time.time()
    deadline = start_time + 50
    
    # Random restarts
    while time.time() < deadline:
        candidates = list(range(2, 41))
        random.shuffle(candidates)
        
        sums = {zero}
        current_set = []
        
        for n in candidates:
            v = vecs[n]
            collision = False
            new_sums_list = []
            for s in sums:
                ns = tuple(s[i] + v[i] for i in range(num_primes))
                if ns in sums:
                    collision = True
                    break
                new_sums_list.append(ns)
            
            if not collision:
                sums.update(new_sums_list)
                current_set.append(n)
                
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set
            
    return best_set
