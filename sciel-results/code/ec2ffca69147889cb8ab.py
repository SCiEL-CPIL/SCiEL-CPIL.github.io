import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = 128
    
    # Initial set: Upper half [65..128]
    current_mask = 0
    for i in range(65, 129):
        current_mask |= (1 << (i - 1))
        
    best_mask = current_mask
    best_size = bin(current_mask).count('1')
    
    def get_size(mask):
        return bin(mask).count('1')
        
    def is_valid_addition(mask, x):
        """
        Check if adding element x (1-indexed) to the set represented by mask is valid.
        Returns True if valid.
        """
        bit_x = 1 << (x - 1)
        if mask & bit_x:
            return True
        
        # Check Case 1: x is a vertex (c=x)
        # Need a, b in S distinct from x, a != b, such that:
        # a+x in S, b+x in S, and (a+b in S or a+b == x)
        
        # Find all a in S such that a+x in S
        # a must be in 1..n-x
        candidates_a = []
        for a in range(1, n - x + 1):
            if a == x:
                continue
            if (mask >> (a - 1)) & 1:
                if (mask >> (a + x - 1)) & 1:
                    candidates_a.append(a)
        
        k = len(candidates_a)
        if k > 1:
            for i in range(k):
                a = candidates_a[i]
                for j in range(i + 1, k):
                    b = candidates_a[j]
                    if a == b:
                        continue
                    s = a + b
                    if s == x:
                        return False
                    if s <= n and (mask >> (s - 1)) & 1:
                        return False
        
        # Check Case 2: x is a sum. x = a + b
        # Need c in S \ {a,b} such that a+c in S and b+c in S
        for a in range(1, x // 2 + 1):
            b = x - a
            if b <= a:
                continue
            if (mask >> (a - 1)) & 1 and (mask >> (b - 1)) & 1:
                # Check if there exists c in S \ {a,b} with a+c in S and b+c in S
                # c must be in 1..n-a and 1..n-b
                # We iterate c? No, use bitmasks.
                # C_a: c in S and a+c in S
                # C_b: c in S and b+c in S
                
                # Mask for c such that c in S and a+c in S
                # c in 1..n-a
                # bit c-1 in mask
                # bit a+c-1 in mask -> bit c-1 in mask >> a
                C_a_mask = mask & (mask >> a)
                # Mask for c such that c in S and b+c in S
                C_b_mask = mask & (mask >> b)
                
                potential_c_mask = C_a_mask & C_b_mask
                # Remove a and b
                potential_c_mask &= ~(1 << (a - 1))
                potential_c_mask &= ~(1 << (b - 1))
                
                # Also ensure c is not x? 
                # If c=x, then a+x in S and b+x in S.
                # But x is not in mask yet. So c=x is not in mask.
                # So potential_c_mask won't have bit x-1 set.
                
                if potential_c_mask != 0:
                    return False
        
        return True

    def try_greedy_add(mask):
        """Try to add missing elements in ascending order. Returns new_mask."""
        for x in range(1, n + 1):
            if not (mask >> (x - 1)) & 1:
                if is_valid_addition(mask, x):
                    mask |= (1 << (x - 1))
        return mask

    def sanitize_mask(mask):
        """Remove elements that form a violation. Returns valid mask."""
        # This is a safety net.
        # We iterate and remove any element that is part of a bad triple.
        # Since N is small, we can check all triples in the final set?
        # No, just remove elements that are 'suspicious'.
        # Actually, let's just verify the final set.
        pass

    # Phase 1: Greedy addition from initial set
    current_mask = try_greedy_add(current_mask)
    best_size = get_size(current_mask)
    best_mask = current_mask
    
    # Phase 2: Random local search
    remaining_time = deadline - time.time()
    num_rounds = max(5, int(remaining_time / 3))
    
    for round_idx in range(num_rounds):
        if time.time() >= deadline - 3:
            break
        
        current = best_mask
        # Randomly remove 0-2 elements
        for _ in range(random.randint(0, 2)):
            bits = [i for i in range(n) if (current >> i) & 1]
            if bits:
                idx = random.choice(bits)
                current &= ~(1 << idx)
        
        # Simple local search
        round_deadline = min(time.time() + 2.0, deadline - 3)
        attempts = 0
        max_attempts = 50000
        while time.time() < round_deadline and attempts < max_attempts:
            attempts += 1
            x = random.randint(1, n)
            if not (current >> (x - 1)) & 1:
                if is_valid_addition(current, x):
                    current |= (1 << (x - 1))
            
            if random.random() < 0.1:
                idx = random.randint(0, n - 1)
                if (current >> idx) & 1:
                    current &= ~(1 << idx)
        
        if get_size(current) > best_size:
            best_size = get_size(current)
            best_mask = current
            best_mask = try_greedy_add(best_mask)
            best_size = get_size(best_mask)
    
    # Final greedy pass
    best_mask = try_greedy_add(best_mask)
    best_size = get_size(best_mask)
    
    # Final Verification and Cleanup
    # Convert to list
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    # Check for violations and remove offending elements
    # This is O(N^3) but N is at most 128, so ~2M ops. Fast enough.
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            for k in range(j + 1, len(result)):
                a, b, c = result[i], result[j], result[k]
                s1 = a + b
                s2 = a + c
                s3 = b + c
                if s1 in result and s2 in result and s3 in result:
                    # Violation found. Remove one of them (e.g., k)
                    # To be safe, we restart or just remove k
                    # Since this is a safety net, we remove k and mark invalid
                    # But we need to return a valid set.
                    # We'll just remove the largest one involved and re-check?
                    # Simpler: Return the upper half if this happens.
                    valid = False
                    break
            if not valid: break
        if not valid: break
        
    if not valid:
        # Fallback to a known valid set (upper half)
        return list(range(65, 129))
        
    return result
