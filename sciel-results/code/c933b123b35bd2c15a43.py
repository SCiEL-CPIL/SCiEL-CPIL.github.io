import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    offset = 25000
    mask = (1 << 50001) - 1
    
    def greedy(order):
        S = []
        sums = {}
        
        for x in order:
            if time.time() > deadline:
                return S
            
            valid = True
            bits = 1 << offset
            non_empty_zero = 0
            
            # Check if x is mean of subset of S
            if S:
                for t in S:
                    d = t - x
                    # Update bits
                    shifted = (bits << d) & mask
                    # Check if -d was reachable in old bits
                    if (bits >> (offset - d)) & 1:
                        non_empty_zero |= (1 << offset)
                    bits |= shifted
                
                if (non_empty_zero >> offset) & 1:
                    valid = False
            
            # Check if x violates existing a in S
            if valid:
                for a in S:
                    target = a - x
                    if -25000 <= target <= 25000:
                        if (sums[a] >> (offset + target)) & 1:
                            valid = False
                            break
            
            if valid:
                S.append(x)
                
                # Update sums for existing a
                for a in S[:-1]:
                    d = x - a
                    old_bits = sums[a]
                    sums[a] = (old_bits | (old_bits << d)) & mask
                
                # Initialize sums[x]
                sums[x] = bits

        return S

    best_set = []
    
    # Try natural order
    s1 = greedy(range(1, 513))
    if len(s1) > len(best_set):
        best_set = s1
        
    # Try reverse order
    s2 = greedy(range(512, 0, -1))
    if len(s2) > len(best_set):
        best_set = s2
        
    # Try random orders
    while time.time() < deadline:
        order = list(range(1, 513))
        random.shuffle(order)
        s_rand = greedy(order)
        if len(s_rand) > len(best_set):
            best_set = s_rand
            
    return best_set
