import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = 128
    all_elements = list(range(1, n + 1))
    
    # Precompute for fast validity checks
    # For each x, precompute which pairs (a,b) would create a violation if x is added
    # This is complex, so we'll use a faster bitmask approach
    
    def popcount(mask):
        return bin(mask).count('1')
    
    def is_valid_addition_fast(mask, x):
        """
        Fast check: can we add x to the set represented by mask?
        Returns True if valid.
        """
        bit_x = 1 << (x - 1)
        if mask & bit_x:
            return True
        
        # Case 1: x is a vertex c. Need a,b in S with a+x in S, b+x in S, and a+b in S (or a+b==x)
        # Find all a in S such that a+x in S
        # a in S means bit (a-1) set. a+x in S means bit (a+x-1) set.
        # So a-1 must be set in mask AND (a-1)+x must be set in mask.
        # Let idx = a-1. Then idx set in mask, and idx+x set in mask.
        # C_a = mask & (mask >> x) gives bits idx where both idx and idx+x are set.
        # These correspond to a = idx+1.
        
        C_a_mask = mask & (mask >> x)
        
        # Now we need two distinct a,b in C_a (excluding x) such that a+b in S or a+b==x
        # Get list of candidates from C_a_mask
        candidates = []
        temp = C_a_mask
        while temp:
            low_bit = temp & (-temp)
            idx = low_bit.bit_length() - 1
            a = idx + 1
            if a != x:
                candidates.append(a)
            temp ^= low_bit
        
        k = len(candidates)
        if k >= 2:
            for i in range(k):
                a = candidates[i]
                for j in range(i+1, k):
                    b = candidates[j]
                    s = a + b
                    if s == x:
                        return False
                    if s <= n and (mask >> (s-1)) & 1:
                        return False
        
        # Case 2: x is a sum. x = a+b for some a,b in S.
        # Need c in S\{a,b} such that a+c in S and b+c in S.
        for a in range(1, x//2 + 1):
            b = x - a
            if b <= a:
                continue
            if (mask >> (a-1)) & 1 and (mask >> (b-1)) & 1:
                # C_a = {c in S | a+c in S} = mask & (mask >> a)
                # C_b = {c in S | b+c in S} = mask & (mask >> b)
                C_a = mask & (mask >> a)
                C_b = mask & (mask >> b)
                potential_c = C_a & C_b
                # Exclude a and b
                potential_c &= ~(1 << (a-1))
                potential_c &= ~(1 << (b-1))
                if potential_c != 0:
                    return False
        
        return True
    
    def greedy_fill(mask):
        """Try to greedily add elements in order."""
        for x in range(1, n+1):
            if not (mask >> (x-1)) & 1:
                if is_valid_addition_fast(mask, x):
                    mask |= (1 << (x-1))
        return mask
    
    def simulated_annealing(initial_mask, time_limit):
        """Run simulated annealing starting from initial_mask."""
        current = initial_mask
        best = initial_mask
        best_size = popcount(best)
        current_size = popcount(current)
        
        T = 1.0
        cooling = 0.9995
        min_T = 0.01
        
        steps = 0
        end_time = time.time() + time_limit
        
        while time.time() < end_time and T > min_T:
            steps += 1
            # Random move: add or remove a random element
            x = random.randint(1, n)
            bit_x = 1 << (x-1)
            
            if mask_has_bit := (current >> (x-1)) & 1:
                # Remove x
                new_mask = current & ~bit_x
                new_size = current_size - 1
                delta = -1
            else:
                # Try to add x
                if is_valid_addition_fast(current, x):
                    new_mask = current | bit_x
                    new_size = current_size + 1
                    delta = 1
                else:
                    # Can't add, try removing a random element instead
                    idx = random.randint(0, n-1)
                    if (current >> idx) & 1:
                        x = idx + 1
                        new_mask = current & ~(1 << idx)
                        new_size = current_size - 1
                        delta = -1
                    else:
                        continue
            
            # Accept or reject based on simulated annealing
            if delta > 0 or random.random() < (T / max(0.1, abs(delta))):
                current = new_mask
                current_size = new_size
                if current_size > best_size:
                    best = current
                    best_size = current_size
            
            T *= cooling
        
        return best, best_size
    
    # Strategy: multiple restarts with different initial sets
    best_mask = 0
    best_size = 0
    
    # Initial set: upper half
    initial_mask = 0
    for i in range(65, 129):
        initial_mask |= (1 << (i-1))
    
    # First, greedy fill from upper half
    greedy_mask = greedy_fill(initial_mask)
    best_mask = greedy_mask
    best_size = popcount(best_mask)
    
    # Now run simulated annealing with time budget
    # Split remaining time among several runs
    elapsed = time.time() - start_time
    remaining = deadline - start_time - elapsed
    if remaining > 5:
        # Run SA for most of the time
        sa_mask, sa_size = simulated_annealing(greedy_mask, remaining - 5)
        if sa_size > best_size:
            best_mask = sa_mask
            best_size = sa_size
    
    # Try a few more restarts if time permits
    elapsed = time.time() - start_time
    if deadline - start_time - elapsed > 10:
        # Try starting from a random subset
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            rand_mask = 0
            for i in range(1, n+1):
                if random.random() < 0.4:
                    rand_mask |= (1 << (i-1))
            # Greedy fill
            g_mask = greedy_fill(rand_mask)
            g_size = popcount(g_mask)
            if g_size > best_size:
                best_mask = g_mask
                best_size = g_size
            # Quick SA
            if deadline - start_time - time.time() > 3:
                sa_mask, sa_size = simulated_annealing(g_mask, 3)
                if sa_size > best_size:
                    best_mask = sa_mask
                    best_size = sa_size
    
    # Convert to list
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(i + 1)
    
    return result
