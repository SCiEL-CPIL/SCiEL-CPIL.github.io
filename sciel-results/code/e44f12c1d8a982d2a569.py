import sys

def construct():
    n = 256
    # 1. Generate initial greedy solution
    S = set()
    # We store as a list for fast iteration, but set for lookup
    # Greedy order: 1 to n
    for x in range(1, n + 1):
        # Check if x forms a 3-AP with any pair in S
        # Since x is the largest, we only check if x is the 3rd term: x = 2b - a
        # a, b in S, a < b
        valid = True
        # Optimization: iterate over S
        for a in S:
            for b in S:
                if a < b:
                    if 2 * b - a == x:
                        valid = False
                        break
            if not valid:
                break
        if valid:
            S.add(x)
    
    S_list = sorted(S)
    
    # 2. Local Search: Try to swap larger elements with smaller rejected elements
    # We want to maximize sum(1/x).
    # Rejected elements are those in [1, n] not in S.
    rejected = [x for x in range(1, n + 1) if x not in S]
    
    improved = True
    while improved:
        improved = False
        # Sort rejected by x (ascending) to prioritize highest weight (smallest x)
        # But we need to find a valid swap.
        # For each x in rejected (sorted ascending), try to find a y in S (sorted descending)
        # such that y > x and S \ {y} + {x} is valid.
        # We pick the best swap (largest gain) or first valid swap?
        # First valid swap for smallest x is good because 1/x is large.
        
        # To be efficient, we can just iterate x in rejected (ascending)
        # and y in S (descending, y > x)
        # If swap is valid, do it, mark improved, and restart (or continue with updated sets)
        
        # Since sets change, it's safer to restart the loop or use a flag.
        # Given N=256, this should be fast.
        
        for x in rejected:
            if x in S:
                continue # x might have been added in a previous swap
            # Try to swap x with some y in S where y > x
            # We want to remove the "least useful" y that allows x to be added?
            # Or just any y?
            # Let's try to find ANY valid y.
            # To maximize gain, we should probably remove the largest y?
            # Because 1/y is small, so losing 1/y is cheap.
            # And 1/x is large.
            # So iterate y in S sorted descending.
            
            for y in sorted(S, reverse=True):
                if y <= x:
                    break
                # Check if S' = S - {y} + {x} is 3-AP free
                # New set S_new = (S - {y}) U {x}
                # We need to check if any 3-AP exists in S_new.
                # Most elements are unchanged. We only need to check APs involving x.
                # APs involving x:
                # 1. x, a, b where a,b in S_new, a<b. x=2a-b? No, x is smallest.
                #    So x is 1st term: x, a, b => 2a = x+b => b = 2a-x.
                #    Check if b in S_new.
                # 2. a, x, b where a,b in S_new. 2x = a+b.
                #    Check if a,b in S_new.
                # 3. a, b, x where a,b in S_new. 2b = a+x => a = 2b-x.
                #    Check if a in S_new.
                
                # Since x < y, and y is removed, x is added.
                # S_new contains x.
                
                valid_swap = True
                
                # Check APs where x is involved
                # We iterate over pairs in S_new? That's O(N^2).
                # Better: iterate over elements in S_new (excluding x) and check.
                
                # Let's just do a full check for simplicity, N=256 is small.
                # But we do this many times.
                # Optimization: Only check APs containing x.
                
                S_new = S - {y}
                S_new.add(x)
                
                # Check if any 3-AP in S_new involves x
                # Case 1: x is middle. a, x, b => a+b=2x.
                # a,b must be in S_new.
                for a in S_new:
                    if a == x: continue
                    b = 2 * x - a
                    if b in S_new and b != a:
                        valid_swap = False
                        break
                
                if valid_swap:
                    # Case 2: x is first. x, a, b => b = 2a - x.
                    for a in S_new:
                        if a == x: continue
                        b = 2 * a - x
                        if b in S_new and b > a:
                            valid_swap = False
                            break
                
                if valid_swap:
                    # Case 3: x is last. a, b, x => a = 2b - x.
                    for b in S_new:
                        if b == x: continue
                        a = 2 * b - x
                        if a in S_new and a < b:
                            valid_swap = False
                            break
                
                if valid_swap:
                    # Perform swap
                    S = S_new
                    # Update rejected list? 
                    # x is now in S. y is now rejected.
                    # We can just restart the outer loop to be safe and simple.
                    # Or update lists.
                    improved = True
                    break # Break y loop, restart x loop?
            if improved:
                break # Restart rejected loop with updated S and rejected
        
    return sorted(S)
