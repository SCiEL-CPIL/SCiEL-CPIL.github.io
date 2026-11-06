import time
import random
import math

def construct():
    """
    Constructs a subset of [1, 256] with no 4-term arithmetic progression
    that maximizes the number of 3-term arithmetic progressions.
    """
    N = 256
    start_time = time.time()
    time_limit = 55.0  # Safety margin
    
    # Helper functions for incremental updates
    # State: in_set (list of bools, index 1..N), count_3, count_4
    
    def count_3aps_full(in_set):
        c = 0
        for a in range(1, N + 1):
            if not in_set[a]:
                continue
            # a is start
            d = 1
            while a + 2 * d <= N:
                if in_set[a + d] and in_set[a + 2 * d]:
                    c += 1
                d += 1
            # a is middle
            d = 1
            while a - d >= 1 and a + d <= N:
                if in_set[a - d] and in_set[a + d]:
                    c += 1
                d += 1
            # a is end
            d = 1
            while a - 2 * d >= 1:
                if in_set[a - d] and in_set[a - 2 * d]:
                    c += 1
                d += 1
        return c

    def count_4aps_full(in_set):
        c = 0
        for a in range(1, N + 1):
            d = 1
            while a + 3 * d <= N:
                if in_set[a + d] and in_set[a + 2 * d] and in_set[a + 3 * d]:
                    c += 1
                d += 1
        return c

    def delta_add(x, in_set):
        """Calculate change in count_3 and count_4 when adding x."""
        d3 = 0
        d4 = 0
        # x as start of 3-AP
        d = 1
        while x + 2 * d <= N:
            if in_set[x + d] and in_set[x + 2 * d]:
                d3 += 1
            d += 1
        # x as middle of 3-AP
        d = 1
        while x - d >= 1 and x + d <= N:
            if in_set[x - d] and in_set[x + d]:
                d3 += 1
            d += 1
        # x as end of 3-AP
        d = 1
        while x - 2 * d >= 1:
            if in_set[x - d] and in_set[x - 2 * d]:
                d3 += 1
            d += 1
            
        # x as part of 4-AP
        # x is 1st term
        d = 1
        while x + 3 * d <= N:
            if in_set[x + d] and in_set[x + 2 * d] and in_set[x + 3 * d]:
                d4 += 1
            d += 1
        # x is 2nd term
        d = 1
        while x - d >= 1 and x + 2 * d <= N:
            if in_set[x - d] and in_set[x + d] and in_set[x + 2 * d]:
                d4 += 1
            d += 1
        # x is 3rd term
        d = 1
        while x - 2 * d >= 1 and x + d <= N:
            if in_set[x - 2 * d] and in_set[x - d] and in_set[x + d]:
                d4 += 1
            d += 1
        # x is 4th term
        d = 1
        while x - 3 * d >= 1:
            if in_set[x - 3 * d] and in_set[x - 2 * d] and in_set[x - d]:
                d4 += 1
            d += 1
        return d3, d4

    def delta_remove(x, in_set):
        """Calculate change in count_3 and count_4 when removing x.
        Note: in_set[x] is True at this point."""
        d3 = 0
        d4 = 0
        # x as start of 3-AP
        d = 1
        while x + 2 * d <= N:
            if in_set[x + d] and in_set[x + 2 * d]:
                d3 += 1
            d += 1
        # x as middle of 3-AP
        d = 1
        while x - d >= 1 and x + d <= N:
            if in_set[x - d] and in_set[x + d]:
                d3 += 1
            d += 1
        # x as end of 3-AP
        d = 1
        while x - 2 * d >= 1:
            if in_set[x - d] and in_set[x - 2 * d]:
                d3 += 1
            d += 1
            
        # x as part of 4-AP
        # x is 1st term
        d = 1
        while x + 3 * d <= N:
            if in_set[x + d] and in_set[x + 2 * d] and in_set[x + 3 * d]:
                d4 += 1
            d += 1
        # x is 2nd term
        d = 1
        while x - d >= 1 and x + 2 * d <= N:
            if in_set[x - d] and in_set[x + d] and in_set[x + 2 * d]:
                d4 += 1
            d += 1
        # x is 3rd term
        d = 1
        while x - 2 * d >= 1 and x + d <= N:
            if in_set[x - 2 * d] and in_set[x - d] and in_set[x + d]:
                d4 += 1
            d += 1
        # x is 4th term
        d = 1
        while x - 3 * d >= 1:
            if in_set[x - 3 * d] and in_set[x - 2 * d] and in_set[x - d]:
                d4 += 1
            d += 1
        return -d3, -d4

    # 1. Greedy Initialization
    in_set = [False] * (N + 1)
    for x in range(1, N + 1):
        # Check if adding x creates a 4-AP
        can_add = True
        # x is 1st
        d = 1
        while x + 3 * d <= N:
            if in_set[x + d] and in_set[x + 2 * d] and in_set[x + 3 * d]:
                can_add = False
                break
            d += 1
        if can_add:
            # x is 2nd
            d = 1
            while x - d >= 1 and x + 2 * d <= N:
                if in_set[x - d] and in_set[x + d] and in_set[x + 2 * d]:
                    can_add = False
                    break
                d += 1
        if can_add:
            # x is 3rd
            d = 1
            while x - 2 * d >= 1 and x + d <= N:
                if in_set[x - 2 * d] and in_set[x - d] and in_set[x + d]:
                    can_add = False
                    break
                d += 1
        if can_add:
            # x is 4th
            d = 1
            while x - 3 * d >= 1:
                if in_set[x - 3 * d] and in_set[x - 2 * d] and in_set[x - d]:
                    can_add = False
                    break
                d += 1
        
        if can_add:
            in_set[x] = True

    # Calculate initial scores
    count_3 = count_3aps_full(in_set)
    count_4 = count_4aps_full(in_set)
    
    # 2. Simulated Annealing
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    penalty = 1000.0
    score = count_3 - penalty * count_4
    
    best_score = score
    best_set = in_set[:]
    
    # Precompute random numbers or use random module
    # We will run iterations until time limit
    
    while time.time() - start_time < time_limit:
        x = random.randint(1, N)
        
        if in_set[x]:
            # Try removing x
            d3, d4 = delta_remove(x, in_set)
            new_score = score + d3 - penalty * d4
            delta_score = new_score - score
        else:
            # Try adding x
            d3, d4 = delta_add(x, in_set)
            new_score = score + d3 - penalty * d4
            delta_score = new_score - score
        
        # Metropolis criterion
        if delta_score > 0 or random.random() < math.exp(delta_score / T):
            # Accept
            if in_set[x]:
                in_set[x] = False
            else:
                in_set[x] = True
            count_3 += d3
            count_4 += d4
            score = new_score
            
            if score > best_score:
                best_score = score
                best_set = in_set[:]
        
        T *= cooling
        if T < T_min:
            T = T_min
            
    # 3. Repair if necessary (ensure no 4-APs)
    # The SA might end with count_4 > 0 if penalty wasn't strong enough or T was high
    # But with penalty 1000, it's unlikely to stay in a 4-AP state for long.
    # However, to be safe, we enforce validity on the best_set found.
    
    in_set = best_set[:]
    count_4 = count_4aps_full(in_set)
    while count_4 > 0:
        # Find a 4-AP
        found = False
        for a in range(1, N + 1):
            d = 1
            while a + 3 * d <= N:
                if in_set[a] and in_set[a + d] and in_set[a + 2 * d] and in_set[a + 3 * d]:
                    # Found 4-AP: a, a+d, a+2d, a+3d
                    # Remove the element that contributes least to 3-APs
                    candidates = [a, a + d, a + 2 * d, a + 3 * d]
                    min_contrib = float('inf')
                    best_remove = -1
                    for c in candidates:
                        # Count 3-APs involving c
                        contrib = 0
                        # c as start
                        dd = 1
                        while c + 2 * dd <= N:
                            if in_set[c + dd] and in_set[c + 2 * dd]:
                                contrib += 1
                            dd += 1
                        # c as middle
                        dd = 1
                        while c - dd >= 1 and c + dd <= N:
                            if in_set[c - dd] and in_set[c + dd]:
                                contrib += 1
                            dd += 1
                        # c as end
                        dd = 1
                        while c - 2 * dd >= 1:
                            if in_set[c - dd] and in_set[c - 2 * dd]:
                                contrib += 1
                            dd += 1
                        if contrib < min_contrib:
                            min_contrib = contrib
                            best_remove = c
                    in_set[best_remove] = False
                    found = True
                    break
            if found:
                break
        if not found:
            break
        count_4 = count_4aps_full(in_set)
        
    # Final check and return
    final_count_4 = count_4aps_full(in_set)
    if final_count_4 > 0:
        # Fallback: just return the greedy set if repair failed (shouldn't happen)
        # Or use the original greedy set
        pass
        
    return [i for i in range(1, N + 1) if in_set[i]]

