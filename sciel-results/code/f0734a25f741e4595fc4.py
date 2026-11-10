def construct():
    import math
    
    # Initial set B: all numbers from 129 to 256
    B = set(range(129, 257))
    A = []
    
    # Precompute multiples for each a in 2..128 that fall in B
    # multiples_in_B[a] = set of b in B such that b % a == 0
    multiples_in_B = {a: set(b for b in B if b % a == 0) for a in range(2, 129)}
    
    # Current B as a set to allow efficient removal
    current_B = set(B)
    
    # Greedy loop: pick the a with highest marginal gain
    while True:
        best_a = -1
        best_gain = 0.0
        
        for a in range(2, 129):
            if a in A:
                continue
            
            # Check LCM constraint with existing A
            valid = True
            for x in A:
                if math.lcm(a, x) <= 256:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Calculate marginal gain
            # Gain: 1/a
            # Cost: sum of 1/b for b in multiples_in_B[a] that are still in current_B
            cost = 0.0
            for b in multiples_in_B[a]:
                if b in current_B:
                    cost += 1.0 / b
            
            gain = 1.0 / a - cost
            
            if gain > best_gain:
                best_gain = gain
                best_a = a
        
        if best_a == -1 or best_gain <= 0:
            break
        
        # Add best_a to A
        A.append(best_a)
        # Remove its multiples from current_B
        for b in multiples_in_B[best_a]:
            current_B.remove(b)
            
    return A + list(current_B)
