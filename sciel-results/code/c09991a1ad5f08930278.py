import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0
    
    n = 128
    base = list(range(n//2 + 1, n + 1))
    smalls = list(range(1, n//2 + 1))
    
    best_set = set(base)
    best_size = len(base)
    
    # Pre-compute base pairs? No, base has no pairs with sum in base.
    
    for _ in range(200):
        if time.time() > deadline:
            break
            
        random.shuffle(smalls)
        current_set = set(base)
        
        # List of pairs (a, b) with a < b, a, b in current_set, and a+b in current_set
        valid_pairs = []
        
        for x in smalls:
            if time.time() > deadline:
                break
                
            # Check if adding x is valid
            valid = True
            
            # Check against existing valid pairs
            for a, b in valid_pairs:
                # We need x+a in current_set and x+b in current_set
                # Note: a+b is already in current_set by definition of valid_pairs
                if (x + a) in current_set and (x + b) in current_set:
                    valid = False
                    break
            
            # Also, x might form new valid pairs with existing elements
            # But we only care if adding x *creates* a violation.
            # A violation is a triple (a, b, c) such that a+b, a+c, b+c in S.
            # If we add x, the new triples involving x are (x, a, b) where a, b in old S.
            # For such a triple to be a violation, we need:
            # x+a in S_new, x+b in S_new, a+b in S_new.
            # Since S_new = S_old U {x}, and x+a != x (a>0), x+a must be in S_old.
            # Similarly x+b in S_old. And a+b in S_old.
            # So the condition is exactly: a+b in S_old, x+a in S_old, x+b in S_old.
            # This is what we checked above using valid_pairs (which are pairs in S_old with sum in S_old).
            
            if valid:
                current_set.add(x)
                
                # Update valid_pairs
                # New pairs involving x: (x, a) where a in current_set, a != x
                # Check if x+a in current_set
                for a in current_set:
                    if a == x:
                        continue
                    if (x + a) in current_set:
                        # Add pair (min(x,a), max(x,a))
                        if x < a:
                            valid_pairs.append((x, a))
                        else:
                            valid_pairs.append((a, x))
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
        if best_size == n:
            break

    return list(best_set)
