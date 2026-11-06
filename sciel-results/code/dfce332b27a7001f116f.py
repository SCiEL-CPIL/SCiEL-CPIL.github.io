import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    all_values = list(range(2, 129))
    
    # Identify primes and composites
    primes = []
    composites = []
    for i in all_values:
        is_prime = True
        for p in range(2, int(i**0.5) + 1):
            if i % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
        else:
            composites.append(i)
            
    primes.sort()
    composites.sort()
    
    def is_valid(seq):
        """Check if all subarray products are distinct."""
        n = len(seq)
        prods = set()
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                if p in prods:
                    return False
                prods.add(p)
        return True

    def greedy_insert(base_seq, candidates):
        """
        Try to insert candidates into base_seq one by one.
        Candidates are assumed to be sorted or in a specific order.
        Returns the resulting sequence.
        """
        seq = list(base_seq)
        seq_set = set(seq)
        
        for c in candidates:
            if c in seq_set:
                continue
            # Find sorted position
            pos = 0
            while pos < len(seq) and seq[pos] < c:
                pos += 1
            
            # Try inserting at pos
            new_seq = seq[:pos] + [c] + seq[pos:]
            if is_valid(new_seq):
                seq = new_seq
                seq_set.add(c)
        return seq

    best_seq = list(primes)
    best_len = len(primes)
    
    random.seed(42)
    
    # Initial deterministic pass
    current = greedy_insert(primes, composites)
    if len(current) > best_len:
        best_seq = current
        best_len = len(current)
        
    # Main loop: Alternate between Randomized Greedy and Local Search
    while time.time() < deadline:
        # 1. Local Search on best_seq
        # Try to swap composites to improve length
        if time.time() > deadline - 10: break
        
        seq = list(best_seq)
        comps_in_seq = [x for x in seq if x in composites]
        missing_comps = [x for x in composites if x not in seq]
        
        improved = False
        # Try some swaps
        # Limit attempts to avoid wasting time if no improvement
        attempts = 200
        while attempts > 0 and time.time() < deadline - 5:
            attempts -= 1
            if not comps_in_seq or not missing_comps:
                break
                
            out_c = random.choice(comps_in_seq)
            in_c = random.choice(missing_comps)
            
            # Remove out_c
            new_seq = [x for x in seq if x != out_c]
            # Insert in_c at sorted position
            pos = 0
            while pos < len(new_seq) and new_seq[pos] < in_c:
                pos += 1
            new_seq = new_seq[:pos] + [in_c] + new_seq[pos:]
            
            if is_valid(new_seq):
                if len(new_seq) > best_len:
                    best_seq = new_seq
                    best_len = len(new_seq)
                    improved = True
                    # Update lists for next iteration
                    comps_in_seq = [x for x in best_seq if x in composites]
                    missing_comps = [x for x in composites if x not in best_seq]
                    seq = list(best_seq)
                else:
                    # Same length. Accept with some probability to explore?
                    # Or just skip. Length is the primary metric.
                    pass
        
        # 2. Randomized Greedy from Primes
        # This explores different "shapes" of the sequence
        if time.time() > deadline - 5: break
        
        shuffled_comps = composites[:]
        random.shuffle(shuffled_comps)
        current = greedy_insert(primes, shuffled_comps)
        if len(current) > best_len:
            best_seq = current
            best_len = len(current)
            
        # 3. Greedy Insertion of missing composites into current best_seq
        # This allows us to build on a good local search result
        if time.time() > deadline - 5: break
        
        missing = [x for x in composites if x not in best_seq]
        if missing:
             # Shuffle missing to try different orders
             random.shuffle(missing)
             current = greedy_insert(best_seq, missing)
             if len(current) > best_len:
                 best_seq = current
                 best_len = len(current)

    return best_seq
