import time
import random
from math import lcm

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    N = 128
    universe = list(range(1, N + 1))
    
    # Precompute BadPartners[x][a]
    # BadPartners[x][a] is a bitmask of b such that {x, a, b} is a bad triple.
    # x, a, b are indices 1..128. We use 0..127 for bits.
    # bit i corresponds to element i+1.
    
    # First, identify all bad triples to fill the masks.
    # Or compute on the fly? 128^3 is ~2 million, which is fast enough for precomputation.
    
    bad_partners = [[0] * (N + 1) for _ in range(N + 1)]
    
    # Optimization: 
    # A triple {a,b,c} is bad if lcm(a,b)==lcm(a,c)==lcm(b,c).
    # Let's iterate over all pairs (a,b) and find c.
    # Actually, iterating a,b,c is 128*128*128 / 6 ~ 350k checks. Very fast.
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm(a, b)
            for c in range(b + 1, N + 1):
                if lcm(a, c) == lab and lcm(b, c) == lab:
                    # {a, b, c} is a bad triple
                    # Set bits in bad_partners
                    # For x=a, partner b, conflict c
                    bad_partners[a][b] |= (1 << (c - 1))
                    bad_partners[a][c] |= (1 << (b - 1))
                    bad_partners[b][a] |= (1 << (c - 1))
                    bad_partners[b][c] |= (1 << (a - 1))
                    bad_partners[c][a] |= (1 << (b - 1))
                    bad_partners[c][b] |= (1 << (a - 1))

    def is_valid_add(x, mask):
        # Check if x can be added to set represented by mask
        # x is 1-based index. mask has bits 0..127 set for elements in S.
        # For each a in S, check if mask & bad_partners[x][a] has any bit set.
        # Note: bad_partners[x][a] contains b's. If b is in S, conflict.
        # We need to iterate through set bits in mask.
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            a_idx = lsb.bit_length() - 1  # 0-based index
            a_val = a_idx + 1             # 1-based value
            # Check if any b in bad_partners[x][a_val] is in mask
            # bad_partners[x][a_val] is a bitmask of b's (0-based bits)
            if mask & bad_partners[x][a_val]:
                return False
            m ^= lsb
        return True

    # Greedy initialization
    S_mask = 0
    for x in range(1, N + 1):
        if is_valid_add(x, S_mask):
            S_mask |= (1 << (x - 1))
    
    best_mask = S_mask
    best_size = bin(S_mask).count('1')
    
    # Simulated Annealing
    # We will try random swaps or additions/removals.
    # State: S_mask
    # Move: 
    #   1. Pick random x in S to remove.
    #   2. Pick random y in U\S to add.
    #   3. Check if y can be added to S \ {x}.
    #   4. If valid, accept.
    #   Also try pure additions if possible.
    
    T = 1.0
    cooling = 0.999
    min_T = 1e-4
    
    while time.time() < deadline:
        if T < min_T:
            break
            
        # Try to improve
        # Randomly pick a candidate move
        # With 50% chance, try to add a random element not in S
        # With 50% chance, try a swap (remove one, add one)
        
        if random.random() < 0.5:
            # Try addition
            # Pick random y not in S
            # Efficient way to pick random not in S:
            # Generate random index, check if in S.
            # Or maintain a list of availables? List is faster for random pick.
            # But maintaining list is O(N) per change. N=128 is small.
            
            # Let's just pick random y from 1..128
            y = random.randint(1, N)
            if S_mask & (1 << (y - 1)):
                continue # y is already in S, skip
            
            if is_valid_add(y, S_mask):
                S_mask |= (1 << (y - 1))
                size = bin(S_mask).count('1')
                if size > best_size:
                    best_size = size
                    best_mask = S_mask
        else:
            # Try swap
            # Pick random x in S
            # Pick random y not in S
            # Check if y can be added to S without x
            
            # Pick x
            # Get list of indices in S? Or random until in S?
            # Expected tries N/|S|. If |S| is large, fast.
            x = random.randint(1, N)
            if not (S_mask & (1 << (x - 1))):
                continue # x not in S
            
            y = random.randint(1, N)
            if S_mask & (1 << (y - 1)):
                continue # y in S
            
            # Try adding y to S \ {x}
            new_mask = S_mask & ~(1 << (x - 1))
            if is_valid_add(y, new_mask):
                S_mask = new_mask | (1 << (y - 1))
                size = bin(S_mask).count('1')
                if size > best_size:
                    best_size = size
                    best_mask = S_mask
                # Accept swap even if size same? 
                # Usually yes, to escape local minima. 
                # But if we accept size-decreasing moves, we need SA logic.
                # Here we only accept if valid. 
                # To allow "downhill" moves, we should accept swaps that keep size same 
                # or decrease size with probability exp(-delta/T).
                # However, since we are maximizing size, and validity is a hard constraint,
                # moving to a smaller set is rarely useful unless it enables a larger set later.
                # Standard SA on valid states:
                # If new_size > old_size: Accept
                # If new_size == old_size: Accept with prob 1 (or random walk)
                # If new_size < old_size: Accept with prob exp(-(old_size - new_size)/T)
                
                # In my code above, I accepted the swap unconditionally if valid.
                # This allows random walk on the surface of constant size, and downhill.
                # This is good for exploration.
            
        T *= cooling

    # Convert best_mask to list
    result = []
    for i in range(N):
        if best_mask & (1 << i):
            result.append(i + 1)
    
    return result

# EVOLVE-BLOCK-END
