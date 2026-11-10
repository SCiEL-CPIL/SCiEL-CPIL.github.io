import time
import random

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin before 60s
    
    # 1. Greedy Initialization
    def greedy_construct():
        values = []
        products = set()
        for x in range(2, 129):
            tails = []
            prod = x
            # Check if x can be appended
            # New products are x, x*last, x*last*prev, etc.
            # We need to check if any of these are in `products`
            # Also, the new products themselves must be unique (which they are if x is unique and seq is increasing? 
            # No, x*a1 could equal x*b1? No, a1 != b1. 
            # Could x*a1*b1 equal x*a2? No, a1*b1 != a2 if seq is increasing? 
            # Actually, if seq is [2, 3], x=6. Tails: 6, 18, 36. 
            # Existing products: 2, 3, 6. 
            # Collision: 6 is in existing. So 6 is rejected.
            
            # Efficient check:
            # Generate tails
            # If any tail in products, reject.
            # Also, tails must be unique among themselves? 
            # Since seq is strictly increasing and positive, tails are strictly increasing. 
            # So tails are unique.
            
            valid = True
            p = x
            tails = [p]
            for a in reversed(values):
                p *= a
                tails.append(p)
                if p in products:
                    valid = False
                    break
            
            if valid:
                # Double check: are any tails colliding with each other? No, they are increasing.
                # Are any tails colliding with existing products? We checked `if p in products` in the loop?
                # Wait, the loop breaks on first collision. But we need to check ALL tails against `products`.
                # The loop above only checks the current `p` against `products`.
                # If `x` is in `products`, we catch it.
                # If `x*last` is in `products`, we catch it.
                # But what if `x*last*prev` is in `products`? We catch it when p becomes that.
                # So the loop is correct.
                
                values.append(x)
                products.update(tails)
        return values

    seq = greedy_construct()
    
    # Helper to check validity of a sequence
    def is_valid(sequence):
        products = set()
        n = len(sequence)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= sequence[j]
                if p in products:
                    return False
                products.add(p)
        return True

    # Local Search
    # Moves: 1-out, 2-in
    # We try to remove one element and insert two others to increase length.
    
    def try_improve(seq):
        n = len(seq)
        used = set(seq)
        unused = [x for x in range(2, 129) if x not in used]
        
        # Sort unused to try smaller numbers first? Or random?
        # Smaller numbers might be safer?
        # Let's iterate through all positions and candidate pairs.
        
        # To avoid O(N^3) or worse, we limit the search.
        # We'll try all 1-out/2-in moves, but break if time runs out.
        
        for i in range(n):
            left_bound = seq[i-1] if i > 0 else 1
            right_bound = seq[i+1] if i < n-1 else 129
            
            # Candidates x, y such that left_bound < x < y < right_bound
            # And x, y not in used
            
            # We can filter unused
            candidates = [x for x in unused if left_bound < x < right_bound]
            
            if len(candidates) < 2:
                continue
                
            # Try all pairs
            # To speed up, we can limit the number of pairs tried per i
            # Or just iterate.
            
            # Optimization: Precompute products of seq without i?
            # That's expensive.
            
            # Let's just build the candidate sequence and check.
            # But checking is O(N^2).
            
            # We'll sample if the space is too large.
            # For now, iterate through all pairs in candidates.
            # If candidates is large, this might be slow.
            
            # Let's limit to first 50 candidates per i to keep it fast?
            # No, let's try to be smart.
            
            # If we find a valid move, return the new seq.
            
            # We'll iterate i from 0 to n-1.
            # For each i, iterate pairs.
            
            # To make it deterministic and fast, we'll just try a subset if it's too big.
            # But 128 is small.
            
            # Let's try all pairs.
            for idx_x in range(len(candidates)):
                x = candidates[idx_x]
                for idx_y in range(idx_x + 1, len(candidates)):
                    y = candidates[idx_y]
                    
                    # Construct new seq
                    # Remove i, insert x, y at i
                    new_seq = seq[:i] + [x, y] + seq[i+1:]
                    
                    # Check if strictly increasing (guaranteed by bounds)
                    # Check if valid
                    if is_valid(new_seq):
                        return new_seq
                        
        return None

    # Main loop
    while time.time() - start_time < timeout:
        new_seq = try_improve(seq)
        if new_seq is not None:
            seq = new_seq
        else:
            # No improvement found with 1-out/2-in
            # Try 1-out/1-in to rearrange?
            # Or break?
            # Let's try 1-out/1-in to see if it unlocks a 2-in later.
            # But 1-out/1-in keeps length same.
            # If we can't increase length, we stop.
            break
            
    return seq
