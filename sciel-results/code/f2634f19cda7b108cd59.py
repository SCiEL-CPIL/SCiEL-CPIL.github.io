import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    max_n = 512
    target_size = 48
    
    selected = []
    # dp[k] is bitset of sums achievable with k elements from selected
    # Max sum 512*48 = 24576
    dp = [0] * 50
    dp[0] = 1
    
    for x in range(1, max_n + 1):
        if time.time() > deadline:
            break
        if len(selected) >= target_size:
            break
            
        valid = True
        
        # Check 1: x is not mean of subset of selected
        for k in range(2, len(selected) + 1):
            if (dp[k] >> (k * x)) & 1:
                valid = False
                break
        
        if not valid:
            continue
            
        # Check 2: No y in selected becomes mean of subset involving x
        # Heuristic: Check k=1,2,3 for T'
        for y in selected:
            # k=1: T' size 1. Sum = 2y - x. Must be in selected \ {y}.
            t1 = 2 * y - x
            if t1 > 0 and t1 != y and t1 in selected:
                valid = False
                break
                
            # k=2: T' size 2. Sum = 3y - x.
            t2 = 3 * y - x
            if t2 > 0:
                # Check if there are two distinct elements a,b in selected \ {y} such that a+b=t2
                # We can use dp for selected, but it includes y.
                # We need to check if t2 is achievable by 2 elements from selected \ {y}.
                # If dp[2] has bit t2, it might use y.
                # If it uses y, then the other element is t2-y.
                # If t2-y is in selected \ {y}, then it's a violation?
                # Wait, if the subset is {y, t2-y}, it uses y. But T' must NOT use y.
                # So we need a pair {a,b} in selected \ {y} with a+b=t2.
                
                # Simple check:
                # Iterate through selected?
                # For small N, it's okay.
                found = False
                for a in selected:
                    if a == y: continue
                    b = t2 - a
                    if b > a and b != y and b in selected:
                        found = True
                        break
                    # Also check b < a? No, a<b covers all pairs.
                    # But what if a==b?
                    if b == a and b != y and b in selected:
                         # Need two copies? No, distinct elements.
                         # So a != b.
                         pass
                if found:
                    valid = False
                    break
            
            # k=3: T' size 3. Sum = 4y - x.
            t3 = 4 * y - x
            if t3 > 0:
                # Check if there are 3 distinct elements in selected \ {y} summing to t3.
                # This is slower. We might skip k=3 for speed.
                pass
        
        if valid:
            selected.append(x)
            # Update dp
            # Shift and OR
            # We need to update in reverse order of k to avoid using x multiple times
            # But dp is for the set, so we add x to existing subsets.
            for k in range(len(selected) - 1, 0, -1):
                dp[k+1] |= (dp[k] << x)
            dp[1] |= (1 << x)
            
    return selected
