import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 128
    # Bit 0 is unused, bits 1..128 correspond to numbers 1..128
    # Actually, let's use bits 0..127 for numbers 1..128 to be 0-indexed internally?
    # No, easier to keep 1-indexed in logic, but use bit i for number i.
    # Python ints are arbitrary precision, so bit 128 is fine.
    
    # Precompute bad triples bitmask B[i][j]
    # B[i][j] is a mask of k such that {i, j, k} is a bad triple.
    # We only need i < j.
    B = [[0] * (N + 1) for _ in range(N + 1)]
    
    # Compute gcd table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g_ij = gcd_table[i][j]
            mask = 0
            for k in range(1, N + 1):
                if k == i or k == j:
                    continue
                g_ik = gcd_table[i][k]
                g_jk = gcd_table[j][k]
                if g_ij == g_ik == g_jk:
                    mask |= (1 << k)
            B[i][j] = mask
            B[j][i] = mask

    def is_valid_addition(x, current_set_mask):
        """
        Check if x can be added to the set represented by current_set_mask.
        x is an integer 1..N.
        current_set_mask is a bitmask of the current set.
        """
        # For every y in current_set, check if there exists z in current_set 
        # such that {x, y, z} is bad.
        # This is equivalent to: (B[x][y] & current_set_mask) == 0 for all y in current_set.
        # We can iterate through set bits of current_set_mask.
        
        # Optimization: If x is already in current_set_mask, return False (though we shouldn't call it then)
        if (current_set_mask >> x) & 1:
            return False
            
        # Iterate through y in current_set
        temp = current_set_mask
        while temp:
            # Get lowest set bit
            lsb = temp & -temp
            y = lsb.bit_length() - 1
            
            # Check if any z in current_set is in B[x][y]
            if (B[x][y] & current_set_mask) != 0:
                return False
                
            temp ^= lsb
            
        return True

    def greedy_build(order):
        s = 0
        for x in order:
            if is_valid_addition(x, s):
                s |= (1 << x)
        return s

    def local_search(s, time_limit):
        """
        Try to improve s by swapping elements.
        Returns improved s.
        """
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try to find a swap: remove u, add v, add w?
            # Or just remove u, add v (size stays same, might enable more adds later?)
            # Better: 1-for-2 swap. Remove u, add v and w.
            
            # Get list of elements in s
            elements_in_s = [i for i in range(1, N + 1) if (s >> i) & 1]
            elements_not_in_s = [i for i in range(1, N + 1) if not ((s >> i) & 1)]
            
            random.shuffle(elements_in_s)
            random.shuffle(elements_not_in_s)
            
            # Try 1-for-2 swaps
            for u in elements_in_s:
                if time.time() > time_limit:
                    break
                
                s_temp = s & ~(1 << u)
                
                # Try to add two elements v, w from not_in_s
                # To be fast, we can try to add one, then the other.
                # Or check if there exist v, w such that s_temp U {v, w} is valid.
                # This is expensive.
                
                # Simpler: Try to add one element v to s_temp.
                # If we can add v, we have size |s|. Then try to add w to s_temp U {v}.
                # If we can add w, we have size |s| + 1.
                
                added_v = -1
                for v in elements_not_in_s:
                    if is_valid_addition(v, s_temp):
                        s_temp_with_v = s_temp | (1 << v)
                        
                        # Now try to add w to s_temp_with_v
                        for w in elements_not_in_s:
                            if w == v:
                                continue
                            if is_valid_addition(w, s_temp_with_v):
                                s = s_temp_with_v | (1 << w)
                                improved = True
                                break
                        if improved:
                            break
                            
                if improved:
                    break
                    
            if not improved:
                # Try 1-for-1 swap to escape local minimum?
                # Or just restart?
                # Let's try a few 1-for-1 swaps that maintain size but change structure.
                # Actually, if we can't grow, we are likely at a local max for this structure.
                # Given time constraints, we might just break and restart with new random order.
                pass
                
        return s

    best_size = 0
    best_set = 0
    
    random.seed(42)
    
    # Multiple restarts with different orders
    # 1. Sorted by value
    # 2. Random shuffles
    # 3. Sorted by number of divisors? Or prime factors?
    
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1))
    ]
    
    for _ in range(10):
        shuffled = list(range(1, N + 1))
        random.shuffle(shuffled)
        orders.append(shuffled)
        
    # Also try orders based on "popularity" or "conflict density"
    # Precompute conflict density: number of bad triples involving i
    # This is a heuristic.
    
    for order in orders:
        if time.time() > deadline:
            break
            
        s = greedy_build(order)
        s = local_search(s, deadline)
        
        size = bin(s).count("1")
        if size > best_size:
            best_size = size
            best_set = s

    # Convert bitmask to list
    result = [i for i in range(1, N + 1) if (best_set >> i) & 1]
    
    # If result is empty (shouldn't happen), return [1]
    if not result:
        return [1]
        
    return result
