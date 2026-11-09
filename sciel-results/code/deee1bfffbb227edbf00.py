import random
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    N = 128
    FULL_MASK = (1 << N) - 1
    
    # Precompute nothing, just use bit ops on the fly
    # S is an integer bitmask
    
    def get_bits(mask):
        bits = []
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            bits.append(idx)
            mask ^= lsb
        return bits

    def is_valid_add(S, x):
        # S is bitmask. x is 0-indexed or 1-indexed?
        # Problem says 1..128.
        # Let's use 1-indexed for logic, but 0-indexed for bitmask?
        # Bit i corresponds to value i+1?
        # Or Bit i corresponds to value i?
        # Values are 1..128.
        # Let Bit i (0..127) correspond to value i+1.
        # So value v is at bit v-1.
        
        x_bit = 1 << (x - 1)
        
        # A_x = { a in S | a + x in S }
        # a is value. a_bit = a-1.
        # a+x is value. (a+x)_bit = a+x-1.
        # We need bit (a-1) in S AND bit (a+x-1) in S.
        # S >> x shifts bits down by x.
        # Bit k in (S >> x) is Bit k+x in S.
        # We want Bit a-1 in S and Bit a+x-1 in S.
        # Let idx_a = a-1.
        # Bit idx_a in S.
        # Bit idx_a + x in S.
        # So A_x_mask = S & (S >> x).
        
        A_x = S & (S >> x)
        
        if A_x == 0:
            return True
            
        # Iterate through bits in A_x
        # For each a in A_x, check if there is b in A_x, b != a, such that a+b in S.
        # a, b are values.
        # a_bit = a-1.
        # a+b in S means bit (a+b-1) in S is 1.
        
        # Get list of values in A_x
        # Instead of list, iterate bits
        ax_bits = get_bits(A_x)
        
        for a in ax_bits:
            a_val = a + 1
            # Find b in A_x such that b != a and a+b in S
            # a+b in S => bit (a+b-1) in S is 1
            # We need b in A_x.
            # Let B_a = A_x & (S >> a_val)
            # Bit b_idx in (S >> a_val) is Bit b_idx + a_val in S.
            # We want Bit b_idx in A_x (so b_val = b_idx+1 is in A_x)
            # AND Bit b_idx + a_val in S (so b_val + a_val in S).
            # So B_a_mask = A_x & (S >> a_val)
            
            B_a = A_x & (S >> a_val)
            
            # Remove a itself from B_a
            # a is in A_x. Is a in B_a?
            # Bit a-1 in B_a?
            # B_a has bit k if k in A_x and k+a_val in S.
            # If k = a-1, then a_val in A_x (yes) and a_val+a_val in S?
            # We need b != a. So we clear bit (a-1) from B_a.
            
            B_a &= ~(1 << (a - 1))
            
            if B_a != 0:
                return False
                
        return True

    def greedy_build():
        S = 0
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        for x in candidates:
            if is_valid_add(S, x):
                S |= (1 << (x - 1))
        return S

    def local_search(S):
        # Try to add any valid element
        improved = True
        while improved:
            if time.time() > DEADLINE:
                break
            improved = False
            for x in range(1, N + 1):
                if not (S & (1 << (x - 1))):
                    if is_valid_add(S, x):
                        S |= (1 << (x - 1))
                        improved = True
        return S

    best_S = 0
    best_size = 0
    
    # Run many restarts
    # Estimate: is_valid_add is ~ O(N^2/word_size) or similar.
    # With N=128, it's very fast.
    # Let's try 5000 restarts.
    
    for r in range(5000):
        if time.time() > DEADLINE:
            break
        
        S = greedy_build()
        S = local_search(S)
        
        size = bin(S).count('1')
        if size > best_size:
            best_size = size
            best_S = S
            
    # Convert best_S to list of ints
    result = []
    for i in range(N):
        if best_S & (1 << i):
            result.append(i + 1)
            
    return result
