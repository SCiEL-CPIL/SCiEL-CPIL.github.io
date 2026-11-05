import time
import random

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Greedy initialization
    S = 0  # Bitset, bit i corresponds to number i+1
    size = 0
    ap_count = 0
    
    # Helper to check if adding x (0-indexed) creates APs
    # Returns number of APs created
    def count_new_aps(x, S):
        count = 0
        # x is 0-indexed. Number is x+1.
        # AP: a, a+d, a+2d, a+3d
        # Cases where x is one of the terms.
        
        # Case 1: x is a (1st term)
        # x, x+d, x+2d, x+3d
        # Need x+d, x+2d, x+3d in S
        # d can be 1 to (N-1-x)//3
        max_d = (N - 1 - x) // 3
        for d in range(1, max_d + 1):
            b1 = (1 << (x + d))
            b2 = (1 << (x + 2*d))
            b3 = (1 << (x + 3*d))
            if (S & b1) and (S & b2) and (S & b3):
                count += 1
                
        # Case 2: x is a+d (2nd term)
        # x-d, x, x+d, x+2d
        # Need x-d, x+d, x+2d in S
        # d can be 1 to min(x, (N-1-x)//2)
        max_d = min(x, (N - 1 - x) // 2)
        for d in range(1, max_d + 1):
            b1 = (1 << (x - d))
            b2 = (1 << (x + d))
            b3 = (1 << (x + 2*d))
            if (S & b1) and (S & b2) and (S & b3):
                count += 1
                
        # Case 3: x is a+2d (3rd term)
        # x-2d, x-d, x, x+d
        # Need x-2d, x-d, x+d in S
        # d can be 1 to min(x//2, (N-1-x)//1)
        max_d = min(x // 2, N - 1 - x)
        for d in range(1, max_d + 1):
            b1 = (1 << (x - 2*d))
            b2 = (1 << (x - d))
            b3 = (1 << (x + d))
            if (S & b1) and (S & b2) and (S & b3):
                count += 1
                
        # Case 4: x is a+3d (4th term)
        # x-3d, x-2d, x-d, x
        # Need x-3d, x-2d, x-d in S
        # d can be 1 to x//3
        max_d = x // 3
        for d in range(1, max_d + 1):
            b1 = (1 << (x - 3*d))
            b2 = (1 << (x - 2*d))
            b3 = (1 << (x - d))
            if (S & b1) and (S & b2) and (S & b3):
                count += 1
                
        return count

    # Helper to count APs involving x in current S (for removal)
    # This is the same as count_new_aps but S is the set WITHOUT x?
    # No, if we remove x, the APs that are destroyed are those that included x.
    # The other 3 elements are in S.
    # So it's the same logic, just checking if the other 3 are in S.
    # Since x is in S, and we are checking if removing x breaks APs,
    # we check if the other 3 are in S.
    def count_old_aps(x, S):
        # S contains x. We want to know how many APs in S contain x.
        # The other 3 elements must be in S.
        # Logic is identical to count_new_aps.
        return count_new_aps(x, S)

    # Greedy Fill
    for x in range(N):
        if not (S >> x) & 1:
            new_aps = count_new_aps(x, S)
            if new_aps == 0:
                S |= (1 << x)
                size += 1

    best_S = S
    best_size = size
    best_ap = ap_count
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.995
    M = 100 # Penalty weight
    
    # Precompute list of indices for random selection
    indices = list(range(N))
    
    while time.time() < deadline:
        if T < 0.1:
            break
            
        # Random move
        x = random.choice(indices)
        is_set = (S >> x) & 1
        
        if is_set:
            # Try to remove x
            old_aps = count_old_aps(x, S)
            delta_size = -1
            delta_aps = -old_aps
            delta_E = -delta_size + M * delta_aps # Minimize E = -size + M*aps
            # E = -size + M*aps. 
            # New E = -(size-1) + M*(aps-old_aps) = -size+1 + M*aps - M*old_aps
            # Delta E = 1 - M*old_aps
            if delta_E < 0 or random.random() < pow(2.71828, -delta_E / T):
                S &= ~(1 << x)
                size += delta_size
                ap_count += delta_aps
                if ap_count == 0 and size > best_size:
                    best_S = S
                    best_size = size
        else:
            # Try to add x
            new_aps = count_new_aps(x, S)
            delta_size = 1
            delta_aps = new_aps
            # Delta E = -1 + M*new_aps
            delta_E = -delta_size + M * delta_aps
            if delta_E < 0 or random.random() < pow(2.71828, -delta_E / T):
                S |= (1 << x)
                size += delta_size
                ap_count += delta_aps
                if ap_count == 0 and size > best_size:
                    best_S = S
                    best_size = size
                    
        T *= cooling

    # Convert bitset to list
    result = []
    for i in range(N):
        if (best_S >> i) & 1:
            result.append(i + 1)
            
    return sorted(result)
