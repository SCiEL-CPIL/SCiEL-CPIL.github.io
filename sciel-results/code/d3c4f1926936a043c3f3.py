import time
import random

def construct():
    n = 12
    m = 40
    deadline = time.time() + 85  # Safety margin
    
    # Precompute CompMask
    # CompMask[i] is a bitset of masks j such that i and j are strictly comparable
    CompMask = [0] * (1 << n)
    full_mask = (1 << n) - 1
    for i in range(1 << n):
        mask = 0
        # Strict subsets
        s = i
        while s:
            mask |= 1 << s
            s = (s - 1) & i
        if i != 0:
            mask |= 1  # 0 is a strict subset of i
            
        # Strict supersets
        rem = full_mask ^ i
        r = rem
        while r:
            sup = i | r
            mask |= 1 << sup
            r = (r - 1) & rem
            
        CompMask[i] = mask

    best_score = 0
    best_config = None
    
    # Multiple restarts
    while time.time() < deadline - 5:
        # Initialize
        if random.random() < 0.5:
            # Greedy Initialization
            selected = []
            curr_bits = 0
            for _ in range(m):
                best_b = -1
                best_gain = -1
                for b in range(1 << n):
                    if curr_bits & (1 << b):
                        continue
                    gain = (CompMask[b] & curr_bits).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_b = b
                selected.append(best_b)
                curr_bits |= 1 << best_b
        else:
            # Random Initialization
            selected = random.sample(range(1 << n), m)
            curr_bits = 0
            for b in selected:
                curr_bits |= 1 << b
        
        # Calculate initial score
        # Sum of degrees / 2
        current_score = 0
        for i in range(m):
            current_score += (CompMask[selected[i]] & curr_bits).bit_count()
        current_score //= 2
        
        # Local Search
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(m):
                A = selected[i]
                # Loss is number of neighbors of A in C
                loss = (CompMask[A] & curr_bits).bit_count()
                
                best_b = -1
                best_delta = 0
                # Try all B not in C
                for b in range(1 << n):
                    if curr_bits & (1 << b):
                        continue
                    gain = (CompMask[b] & curr_bits).bit_count()
                    delta = gain - loss
                    if delta > best_delta:
                        best_delta = delta
                        best_b = b
                        
                if best_b != -1:
                    # Swap A with best_b
                    selected[i] = best_b
                    curr_bits ^= (1 << A)  # Remove A
                    curr_bits |= (1 << best_b) # Add B
                    current_score += best_delta
                    improved = True
                    
        if current_score > best_score:
            best_score = current_score
            best_config = selected[:]

    # Convert to lists
    result = []
    for mask in best_config:
        s = [i for i in range(n) if (mask >> i) & 1]
        result.append(s)
    return result
