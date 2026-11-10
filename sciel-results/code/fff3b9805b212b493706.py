import sys
sys.setrecursionlimit(200)

def construct():
    candidates = list(range(2, 129))
    n = len(candidates)
    best_sequence = []
    best_length = 0
    
    # To speed up, we can pre-filter or just rely on B&B
    # We use a global variable for best found
    def backtrack(idx, current_seq, products):
        nonlocal best_sequence, best_length
        
        # Pruning: Upper bound on length
        # Even if we pick all remaining, can we beat best?
        if len(current_seq) + (n - idx) <= best_length:
            return
            
        if idx == n:
            if len(current_seq) > best_length:
                best_length = len(current_seq)
                best_sequence = current_seq[:]
            return
        
        x = candidates[idx]
        
        # Option 1: Include x
        # Check if x can be added
        # We need to generate all new tail products
        # tails = [x]
        # prod = x
        # for a in reversed(current_seq):
        #     prod *= a
        #     tails.append(prod)
        
        # Optimization: If x is in products, we cannot add it (singleton collision)
        if x not in products:
            tails = [x]
            prod = x
            valid = True
            # Generate tails
            # We need to check if any tail is in products
            # Also, we must ensure that the new tails don't collide with each other?
            # Since the sequence is strictly increasing and integers >= 2,
            # tails are strictly increasing: x < x*last < x*last*prev...
            # So tails are distinct from each other.
            # We only need to check against existing products.
            
            for a in reversed(current_seq):
                prod *= a
                tails.append(prod)
                if prod in products:
                    valid = False
                    break
            
            if valid:
                # Check if any other tail is in products
                # The loop above breaks on first collision.
                # We need to check ALL tails if the first few didn't collide.
                # Re-doing the check properly:
                if valid:
                    for t in tails:
                        if t in products:
                            valid = False
                            break
                
                if valid:
                    new_products = products.union(tails)
                    backtrack(idx + 1, current_seq + [x], new_products)
                    
        # Option 2: Exclude x
        backtrack(idx + 1, current_seq, products)

    backtrack(0, [], set())
    return best_sequence
