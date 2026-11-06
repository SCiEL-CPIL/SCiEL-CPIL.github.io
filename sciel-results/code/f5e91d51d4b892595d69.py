"""Optimized 32-element set maximizing distinct 3-AP differences in [1,512]."""

def construct():
    import time
    import random
    
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin for 120s limit
    
    # Bitset representation: x in 1..512 maps to bit (x-1)
    # S is an integer where bit i is 1 if (i+1) is in the set
    
    def bitset_from_set(S):
        b = 0
        for x in S:
            b |= (1 << (x - 1))
        return b
    
    def set_from_bitset(b):
        s = []
        i = 0
        while b:
            if b & 1:
                s.append(i + 1)
            b >>= 1
            i += 1
        return s
    
    def popcount(b):
        return bin(b).count('1')
    
    def score_bitset(b):
        """Count distinct d > 0 such that a, a+d, a+2d all in set."""
        count = 0
        for d in range(1, 257):
            if (b & (b >> d) & (b >> (2 * d))) != 0:
                count += 1
        return count
    
    # Initial random set of 32 distinct integers from 1..512
    random.seed(42)
    initial_set = random.sample(range(1, 513), 32)
    current_b = bitset_from_set(initial_set)
    current_score = score_bitset(current_b)
    
    best_b = current_b
    best_score = current_score
    
    # Simulated annealing
    T = 10.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.999
    
    while T > T_min:
        if time.time() > deadline - 10.0:
            break
        
        # Generate neighbor: swap one element
        # Pick a random element in the set and a random element not in the set
        in_set = set_from_bitset(current_b)
        not_in_set = [x for x in range(1, 513) if x not in set(in_set)]
        
        if not in_set or not not_in_set:
            break
        
        x_out = random.choice(in_set)
        x_in = random.choice(not_in_set)
        
        # Create new bitset
        new_b = current_b & ~(1 << (x_out - 1))
        new_b |= (1 << (x_in - 1))
        
        new_score = score_bitset(new_b)
        
        # Accept or reject
        if new_score >= current_score:
            current_b = new_b
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_b = current_b
        else:
            delta = new_score - current_score
            # Accept with probability exp(delta / T)
            if random.random() < pow(2.718281828, delta / T):
                current_b = new_b
                current_score = new_score
        
        T *= cooling_rate
    
    # Final local search polish
    if time.time() < deadline - 5.0:
        for _ in range(5000):
            if time.time() > deadline - 2.0:
                break
            in_set = set_from_bitset(best_b)
            not_in_set = [x for x in range(1, 513) if x not in set(in_set)]
            
            x_out = random.choice(in_set)
            x_in = random.choice(not_in_set)
            
            new_b = best_b & ~(1 << (x_out - 1))
            new_b |= (1 << (x_in - 1))
            
            new_score = score_bitset(new_b)
            if new_score > best_score:
                best_score = new_score
                best_b = new_b
    
    return set_from_bitset(best_b)
