"""Find the longest interval in [1,1008] with no modular inverse pair (including self)."""
import time

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s limit
    p = 1009
    
    # Precompute modular inverses
    inv = [0] * p
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
    
    # Identify self-inverse residues (x^2 ≡ 1 mod p): x = 1 and x = p-1 = 1008
    self_inv = {1, p - 1}
    
    best_len = 0
    best_start = 1
    
    # For each starting point s, find the maximum valid end point e
    # An interval [s, e] is valid iff:
    #   1. No element in [s,e] is self-inverse (1 or 1008)
    #   2. For every x in [s,e], inv[x] is NOT in [s,e]
    
    # Strategy: for each s, extend e as far as possible while maintaining validity
    # Key insight: if we fix s, the constraint is that for all x in [s,e], inv[x] notin [s,e]
    # This means inv[x] < s or inv[x] > e for all x in [s,e]
    
    # Efficient approach: 
    # For a given s, the maximum valid e is the minimum over all x >= s where x is in range
    # of (inv[x] - 1) if inv[x] >= s, but we also need to handle the case where inv[x] < s.
    
    # Actually, let's think differently. For a fixed s:
    # - We cannot include any self-inverse element
    # - For each x in [s, e], we need inv[x] < s or inv[x] > e
    # - Equivalently, e < min{inv[x] : x in [s,e], inv[x] >= s} (if such x exists)
    
    # Better approach: iterate s from 1 to p-1, and for each s, find the longest valid interval.
    # For a fixed s, the interval [s, e] is valid iff:
    #   - s, s+1, ..., e are not in self_inv
    #   - For all x in [s, e], inv[x] < s or inv[x] > e
    
    # The binding constraint is: e < min{inv[x] : x in [s, e], inv[x] >= s}
    # But this is circular. Let's use a different approach.
    
    # For each s, we can greedily extend e. But that's O(p^2) worst case.
    # Since p=1009, O(p^2) is ~1 million operations, which is fast.
    
    for s in range(1, p):
        # Early termination: max possible length from s is p - s
        if p - s <= best_len:
            break
        
        # If s itself is self-inverse, skip
        if s in self_inv:
            continue
        
        # Try to extend from s
        e = s
        valid = True
        while e < p - 1:  # e goes up to p-1 = 1008, but 1008 is self-inverse
            # Check if e+1 can be added
            next_e = e + 1
            if next_e in self_inv:
                break
            
            # Check if adding next_e violates the inverse constraint
            # For all x in [s, next_e], inv[x] must not be in [s, next_e]
            # It suffices to check:
            #   - inv[next_e] not in [s, next_e]  (new element's inverse)
            #   - For all x in [s, e], inv[x] not in [s, next_e]
            #     But we already know inv[x] notin [s,e] for x in [s,e]
            #     So we only need to check if inv[x] == next_e for some x in [s,e]
            #     Which is equivalent to: x = inv[next_e] and x in [s,e]
            #     But inv[next_e] is a single value, so check if s <= inv[next_e] <= e
            
            if next_e in self_inv:
                break
            
            # Check inv[next_e]
            inv_next = inv[next_e]
            if s <= inv_next <= next_e:
                break
            
            # Also need to check: for x in [s,e], is inv[x] == next_e?
            # inv[x] == next_e means x == inv[next_e] == inv_next
            # So check if s <= inv_next <= e
            if s <= inv_next <= e:
                break
            
            e = next_e
        
        current_len = e - s + 1
        if current_len > best_len:
            best_len = current_len
            best_start = s
    
    return [best_start, best_len]
