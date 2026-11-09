import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = 128
    # Bitmask representation: bit i is 1 if (i+1) is in the set.
    # We use Python integers for arbitrary precision bitmasks.
    
    # Initial set: Upper half [65..128]
    # Bits 64..127 are set.
    initial_mask = (1 << n) - 1  # All bits 0..127 set
    # Clear lower half bits 0..63
    lower_mask = (1 << (n//2)) - 1
    current_mask = initial_mask & ~lower_mask
    
    best_mask = current_mask
    best_size = bin(current_mask).count('1')
    
    def get_size(mask):
        return bin(mask).count('1')
        
    def is_valid_addition(mask, x):
        """
        Check if adding element x (1-indexed) to the set represented by mask is valid.
        x is 1-indexed. Bit index is x-1.
        """
        bit_x = 1 << (x - 1)
        if mask & bit_x:
            return True # Already in set, no change
        
        # Case 1: x is a vertex (c=x).
        # We need a, b in S such that:
        # 1. a + x in S
        # 2. b + x in S
        # 3. a + b in S OR a + b == x
        # And a != b, a != x, b != x.
        
        # Find all a in S such that a+x in S.
        # If a is in S, bit (a-1) is set. a+x in S means bit (a+x-1) is set.
        # Let's iterate through possible a.
        # Optimization: Only consider a such that a < x? No, a can be anything.
        # But a+x <= 128 => a <= 128-x.
        
        candidates_a = []
        # Iterate through bits set in mask
        # To make this fast, we can iterate through possible a values 1..128
        # But 128 is small.
        for a in range(1, n + 1):
            if (mask >> (a - 1)) & 1:
                # a is in S
                if a + x <= n:
                    if (mask >> (a + x - 1)) & 1:
                        candidates_a.append(a)
                else:
                    # a+x > n, so a+x is not in S.
                    # But wait, condition is a+x in S. If a+x > n, it's not in S.
                    pass
        
        # Now check pairs in candidates_a
        # If len(candidates_a) is small, O(k^2) is fine.
        k = len(candidates_a)
        if k > 1:
            for i in range(k):
                a = candidates_a[i]
                if a == x: continue
                for j in range(i + 1, k):
                    b = candidates_a[j]
                    if b == x: continue
                    if a == b: continue
                    # Check a+b
                    s = a + b
                    if s == x:
                        return False
                    if s <= n and (mask >> (s - 1)) & 1:
                        return False
        elif k == 1:
            # Only one candidate a. No pair a,b distinct.
            pass
            
        # Case 2: x is a sum.
        # Subcase 2a: x = a + b.
        # We need c in S such that a+c in S and b+c in S.
        # And c distinct from a, b.
        
        # Iterate through pairs (a, b) in S such that a + b == x.
        # a < b to avoid duplicates.
        for a in range(1, x // 2 + 1):
            b = x - a
            if b <= a: continue
            # Check if a and b are in S
            if (mask >> (a - 1)) & 1 and (mask >> (b - 1)) & 1:
                # We found a pair (a,b) summing to x.
                # Now check if there exists c in S \ {a,b} such that a+c in S and b+c in S.
                # c must be such that a+c <= n and b+c <= n.
                # So c <= n - max(a,b).
                
                # Efficient check:
                # Let C_a = {c in S | a+c in S}
                # Let C_b = {c in S | b+c in S}
                # We need C_a ∩ C_b to contain an element c != a, c != b.
                
                # Construct bitmask for C_a:
                # c in S (bit c-1 in mask) AND a+c in S (bit a+c-1 in mask).
                # This is equivalent to: (mask & (mask >> a)) ?
                # If bit i is set in mask (i+1 in S), and bit i+a is set in mask (i+1+a in S).
                # So C_a_mask = mask & (mask >> a)
                # The bits set in C_a_mask correspond to c-1 where c in C_a.
                
                C_a_mask = mask & (mask >> a)
                C_b_mask = mask & (mask >> b)
                
                potential_c_mask = C_a_mask & C_b_mask
                
                # Clear bits for a and b
                # a is in C_a? a+a in S? If yes, bit a-1 is set in C_a_mask.
                # We need to exclude c=a and c=b.
                potential_c_mask &= ~(1 << (a - 1))
                potential_c_mask &= ~(1 << (b - 1))
                
                # If any bit remains, there is a valid c.
                if potential_c_mask != 0:
                    return False
        
        return True

    # Local Search
    # Use a greedy approach first to fill up, then local search.
    
    # Greedy initialization: Try to add elements in a specific order
    # Randomized restarts might be better.
    
    best_score = best_size
    
    # Try a few random restarts with different initial seeds
    # But upper half is a good start.
    
    # Fast Local Search
    # We can try to add any missing element.
    # If stuck, remove one.
    
    # To speed up, we can maintain a list of "valid" additions dynamically?
    # That's complex. Just loop.
    
    attempts = 0
    max_attempts = 5000000 # High limit, time is the guard
    
    # List of all elements 1..128
    all_elements = list(range(1, n + 1))
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # 1. Try to add a random missing element
        # Get missing elements
        # To avoid generating list every time, we can just pick a random x and check if it's in mask.
        # But if we pick x already in mask, we waste time.
        # Better: maintain a list of missing? 
        # Since n=128, generating list of missing is O(N). 
        # bin(mask).count('1') is O(N).
        
        # Let's just pick random x.
        x = random.randint(1, n)
        if not (current_mask >> (x - 1)) & 1:
            if is_valid_addition(current_mask, x):
                current_mask |= (1 << (x - 1))
                new_size = get_size(current_mask)
                if new_size > best_score:
                    best_score = new_size
                    best_mask = current_mask
        
        # 2. If we didn't add (or even if we did, occasionally), try to remove
        # To escape local optima.
        # Probability of removal increases if we are stuck?
        # Simple: 10% chance to remove a random element.
        if random.random() < 0.1:
            # Pick a random element from current set
            # Find a set bit
            # Randomly pick index
            idx = random.randint(0, n - 1)
            if (current_mask >> idx) & 1:
                current_mask &= ~(1 << idx)
        
        # Optional: If current size < best_size - 5, restart from best?
        # Or just continue.
        
    # Convert best_mask to list of ints
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    return result
