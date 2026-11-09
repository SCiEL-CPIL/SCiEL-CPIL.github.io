import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 256
    # Precompute reciprocals
    recip = [0.0] * (N + 1)
    for i in range(1, N + 1):
        recip[i] = 1.0 / i
        
    # Precompute suffix sums of reciprocals for upper bound estimation
    # suffix_sum[i] = sum(1/j for j in range(i, N+1))
    suffix_sum = [0.0] * (N + 2)
    for i in range(N, 0, -1):
        suffix_sum[i] = suffix_sum[i + 1] + recip[i]

    best_sum = 0.0
    best_set = []
    
    # Initial greedy solution to establish a baseline
    # Greedily pick smallest numbers (highest weight) that don't form 3-AP
    def get_greedy_solution():
        chosen = set()
        current_sum = 0.0
        for i in range(1, N + 1):
            # Check if i conflicts with chosen
            conflict = False
            for a in chosen:
                # Check if i, a, b form AP
                # b = 2i - a or 2a - i or (i+a)/2
                b1 = 2 * i - a
                if b1 in chosen and b1 != i and b1 != a:
                    conflict = True
                    break
                b2 = 2 * a - i
                if b2 in chosen and b2 != i and b2 != a:
                    conflict = True
                    break
                if (i + a) % 2 == 0:
                    b3 = (i + a) // 2
                    if b3 in chosen and b3 != i and b3 != a:
                        conflict = True
                        break
            if not conflict:
                chosen.add(i)
                current_sum += recip[i]
        return current_sum, list(chosen)

    best_sum, best_set = get_greedy_solution()
    
    # Bitmask representation
    # Bit i is set if number i is available
    full_mask = (1 << (N + 1)) - 2  # Bits 1..N set (bit 0 unused)
    
    # Precompute for each pair (a, b) in 1..N, the set of c such that {a,b,c} is 3-AP
    # Actually, we need to know, given a new element v and existing chosen set C,
    # which elements u become forbidden.
    # Forbidden(u) if exists w in C such that {u, v, w} is 3-AP.
    # This depends on C.
    
    # To speed up, we can precompute for each v, a list of (w, u) pairs?
    # Or simply iterate through chosen elements.
    
    # Optimization: 
    # The "Available" mask is updated by removing forbidden elements.
    # When we add v, we remove u if (u, v, w) is AP for some w in Chosen.
    
    def dfs(available_mask, chosen_sum, chosen_list, avail_sum, depth):
        nonlocal best_sum, best_set
        
        if time.time() > deadline:
            return False # Signal timeout
            
        # Pruning: Upper bound
        # Sum of reciprocals of all available elements is avail_sum
        if chosen_sum + avail_sum <= best_sum:
            return True
            
        if available_mask == 0:
            if chosen_sum > best_sum:
                best_sum = chosen_sum
                best_set = list(chosen_list)
            return True
            
        # Find the smallest available index
        lowest_bit = available_mask & -available_mask
        idx = lowest_bit.bit_length() - 1
        
        # Branch 1: Include idx
        new_chosen_list = chosen_list + [idx]
        new_available = available_mask & ~lowest_bit
        
        # Calculate forbidden mask
        forbidden = 0
        forbidden_sum = 0.0
        for w in chosen_list:
            # u = 2*idx - w
            u1 = 2 * idx - w
            if 1 <= u1 <= N:
                bit = 1 << u1
                if new_available & bit:
                    forbidden |= bit
                    forbidden_sum += recip[u1]
            # u = 2*w - idx
            u2 = 2 * w - idx
            if 1 <= u2 <= N:
                bit = 1 << u2
                if new_available & bit:
                    forbidden |= bit
                    forbidden_sum += recip[u2]
            # u = (idx + w) / 2
            if (idx + w) % 2 == 0:
                u3 = (idx + w) // 2
                if 1 <= u3 <= N:
                    bit = 1 << u3
                    if new_available & bit:
                        forbidden |= bit
                        forbidden_sum += recip[u3]
                    
        new_available &= ~forbidden
        new_avail_sum = avail_sum - recip[idx] - forbidden_sum
        
        # Recurse
        if time.time() > deadline:
            return False
            
        dfs(new_available, chosen_sum + recip[idx], new_chosen_list, new_avail_sum, depth + 1)
        
        if time.time() > deadline:
            return False
            
        # Branch 2: Exclude idx
        # new_available is just available_mask without idx
        dfs(available_mask & ~lowest_bit, chosen_sum, chosen_list, avail_sum - recip[idx], depth + 1)
        
        return True

    # Start DFS
    # We want to explore including small numbers first.
    # The DFS above picks the smallest available number at each step.
    # This aligns with maximizing weight.
    
    # Initial call
    # available_mask has bits 1..N set
    initial_mask = 0
    initial_avail_sum = 0.0
    for i in range(1, N + 1):
        initial_mask |= (1 << i)
        initial_avail_sum += recip[i]
        
    dfs(initial_mask, 0.0, [], initial_avail_sum, 0)
    
    if not best_set:
        return [1]
        
    return best_set
