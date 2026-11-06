import time
import sys
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    # Precompute prime exponent vectors for numbers 2-128
    # Primes up to 128
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 
              53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 
              113, 127]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    def get_exp_vector(n):
        """Get prime exponent vector for n."""
        vec = [0] * num_primes
        for i, p in enumerate(primes):
            while n % p == 0:
                vec[i] += 1
                n //= p
        return tuple(vec)
    
    # Precompute exponent vectors for all numbers 2-128
    exp_vecs = {}
    for x in range(2, 129):
        exp_vecs[x] = get_exp_vector(x)
    
    # Precompute zero vector
    zero_vec = tuple([0] * num_primes)
    
    def vec_add(a, b):
        """Add two exponent vectors."""
        return tuple(ai + bi for ai, bi in zip(a, b))
    
    # Fast greedy initialization
    def _fast_greedy():
        values = []
        products = set()
        for x in range(2, 129):
            # Calculate new suffix products as exponent vectors
            # If we append x to values, the new products are:
            # x, x*values[-1], x*values[-1]*values[-2], ...
            tails = []
            current_vec = exp_vecs[x]
            valid = True
            for a in reversed(values):
                current_vec = vec_add(current_vec, exp_vecs[a])
                if current_vec in products:
                    valid = False
                    break
                tails.append(current_vec)
            
            if not valid:
                continue
            
            # Check if any tail is already in products
            if products.intersection(tails):
                continue
            
            values.append(x)
            products.update(tails)
        
        return values
    
    best_seq = _fast_greedy()
    best_len = len(best_seq)
    
    # Beam search with dynamic width
    def _beam_search():
        nonlocal best_seq, best_len
        start_bs = time.time()
        
        # State: (seq_tuple, prods_set, sufs_tuple)
        # sufs is the tuple of suffix product exponent vectors
        # For sequence [a, b, c], sufs = (c, b*c, a*b*c)
        
        beam = [ ((), set(), ()) ]
        
        # Dynamic beam width: start wide, narrow later
        # We'll track the step and adjust width
        for step in range(127):  # Max 127 elements
            if time.time() > deadline:
                break
            
            # Dynamic width: wider early, narrower later
            if step < 5:
                width = 200
            elif step < 15:
                width = 100
            elif step < 30:
                width = 50
            else:
                width = 30
            
            next_beam = []
            
            for seq, prods, sufs in beam:
                if time.time() > deadline:
                    break
                
                last = seq[-1] if seq else 1
                
                # Try all x > last
                for x in range(last + 1, 129):
                    # Calculate new suffix products
                    # New sufs: (x, x*sufs[0], x*sufs[1], ...)
                    # But we need to add exp_vecs[x] to each suffix
                    x_vec = exp_vecs[x]
                    
                    new_sufs = []
                    current = x_vec
                    valid = True
                    for s in sufs:
                        current = vec_add(current, s)
                        if current in prods:
                            valid = False
                            break
                        new_sufs.append(current)
                    
                    if not valid:
                        continue
                    
                    # Check if any new suffix is in prods
                    new_sufs_set = set(new_sufs)
                    if prods.intersection(new_sufs_set):
                        continue
                    
                    new_seq = seq + (x,)
                    new_prods = prods.union(new_sufs_set)
                    new_sufs_tuple = tuple(new_sufs)
                    
                    next_beam.append((new_seq, new_prods, new_sufs_tuple))
            
            if not next_beam:
                break
            
            # Prune next_beam to width
            # Heuristic: Keep states with smallest last element and highest potential
            # Potential = number of x > last that could potentially be added
            def score_state(state):
                seq, prods, sufs = state
                last = seq[-1] if seq else 1
                # Potential: smaller last is better, longer seq is better
                potential = (128 - last) * 10 + len(seq)
                return -potential  # Minimize for sorting
            
            next_beam.sort(key=score_state)
            beam = next_beam[:width]
            
            # Track best
            if beam:
                best_in_beam = max(beam, key=lambda s: len(s[0]))
                if len(best_in_beam[0]) > best_len:
                    best_len = len(best_in_beam[0])
                    best_seq = list(best_in_beam[0])
        
        return best_seq
    
    # Run beam search
    if time.time() < deadline:
        bs_result = _beam_search()
        if len(bs_result) > len(best_seq):
            best_seq = bs_result
    
    # Ensure we have a valid result
    if not best_seq:
        # Fallback
        values = []
        products = set()
        for x in range(2, 129):
            tails = []
            current_vec = exp_vecs[x]
            valid = True
            for a in reversed(values):
                current_vec = vec_add(current_vec, exp_vecs[a])
                if current_vec in products:
                    valid = False
                    break
                tails.append(current_vec)
            if valid and not products.intersection(tails):
                values.append(x)
                products.update(tails)
        best_seq = values
    
    return best_seq

if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)
