import time
import sys
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill

    # Primes up to 127
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    num_primes = len(primes)
    
    # Precompute factor vectors for 2..128
    # Vector is a tuple of exponents for each prime
    def get_vec(n):
        v = [0] * num_primes
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
            if temp == 1:
                break
        return tuple(v)
        
    vecs = {i: get_vec(i) for i in range(2, 129)}
    zero_vec = (0,) * num_primes
    
    # Precompute which numbers are "light" (fewer prime factors, lower total exponent)
    # These are more likely to be addable without causing collisions
    def vec_weight(v):
        return sum(v)
    
    lightness = {x: vec_weight(vecs[x]) for x in range(2, 129)}

    # State: (seq, prefixes, sums)
    # seq: list of ints
    # prefixes: list of tuples (prefix sum vectors)
    # sums: set of tuples (all subarray sum vectors)
    
    # Initial state
    # prefixes = [0]
    # sums = {}
    beam = [([], [zero_vec], set())]
    
    best_seq = []
    best_len = 0
    
    # Adaptive beam width based on time remaining and progress
    # Start wide, narrow as we approach deadline
    
    for x in range(2, 129):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
            
        # Estimate remaining iterations
        remaining_x = 128 - x + 1
        
        # Determine beam width: wider when we have more time and more numbers to process
        # Use a smooth schedule
        time_frac = elapsed / deadline
        if time_frac < 0.3:
            beam_width = 400
        elif time_frac < 0.5:
            beam_width = 250
        elif time_frac < 0.7:
            beam_width = 150
        elif time_frac < 0.85:
            beam_width = 80
        else:
            beam_width = 40
            
        new_beam = []
        v_x = vecs[x]
        
        for state in beam:
            seq, prefixes, sums = state
            last_prefix = prefixes[-1]
            
            # Option 1: Skip x
            new_beam.append(state)
            
            # Option 2: Take x
            # Calculate new prefix sum
            new_prefix_val = [last_prefix[i] + v_x[i] for i in range(num_primes)]
            new_prefix = tuple(new_prefix_val)
            
            # Calculate new subarray sums
            # New sums are new_prefix - p for all p in prefixes
            new_sums = set()
            valid = True
            for p in prefixes:
                diff_val = [new_prefix[i] - p[i] for i in range(num_primes)]
                diff_tup = tuple(diff_val)
                if diff_tup in sums:
                    valid = False
                    break
                new_sums.add(diff_tup)
            
            if valid:
                # Create new state
                new_seq = seq + [x]
                new_prefixes = prefixes + [new_prefix]
                new_sums_set = sums.union(new_sums)
                
                new_beam.append((new_seq, new_prefixes, new_sums_set))
                
                # Update best
                if len(new_seq) > best_len:
                    best_len = len(new_seq)
                    best_seq = new_seq
        
        if not new_beam:
            break
            
        # Score each state using a heuristic that combines:
        # 1. Current length (primary)
        # 2. "Flexibility" - estimated remaining capacity
        #    Based on how many of the remaining numbers (x+1 to 128) 
        #    are "light" enough to potentially be added
        def state_score(state):
            seq, prefixes, sums = state
            L = len(seq)
            
            # Estimate remaining addable numbers
            # A number y is potentially addable if its vector doesn't create
            # a collision with existing sums. We approximate by checking
            # if the number is "light" (low total exponent).
            # Also, we need to account for the fact that adding a number
            # increases the number of prefixes, making future additions harder.
            
            # Simple heuristic: count how many remaining numbers have low weight
            # and are greater than the last element in seq (if any)
            last_elem = seq[-1] if seq else 1
            remaining_count = 0
            for y in range(x + 1, 129):
                if y > last_elem and lightness[y] <= 2:
                    remaining_count += 1
            
            # The heuristic: L + remaining_count * 0.3
            # This rewards states that can potentially add more light numbers
            return L + remaining_count * 0.3
        
        # Sort by heuristic score (descending)
        new_beam.sort(key=state_score, reverse=True)
        
        # Keep top K
        beam = new_beam[:beam_width]
        
        # Early exit if we can't beat best_len
        # The best possible length from any state in beam is at most
        # max_len_in_beam + (128 - x)
        if beam:
            max_len_in_beam = len(beam[0][0])
            if max_len_in_beam + (128 - x) <= best_len:
                break

    return best_seq

# Test locally if needed
if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)
