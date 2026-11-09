import time

def construct():
    t_start = time.time()
    DEADLINE = t_start + 75.0  # Safety margin for 90s limit
    
    N = 4096
    selected = []
    triple_sums = set()
    
    # To speed up, we can pre-compute or just iterate.
    # Iterating 1..4096 is fine.
    
    for x in range(1, N + 1):
        if time.time() > DEADLINE:
            break
            
        # Check if x can be added
        # New sums involving x:
        # 1. x + x + x
        # 2. x + x + s (for s in selected)
        # 3. x + s + s (for s in selected)
        # 4. x + s1 + s2 (for s1, s2 in selected, s1 <= s2)
        
        new_sums = []
        
        # Case 1: 3x
        s1 = 3 * x
        if s1 in triple_sums:
            continue
        new_sums.append(s1)
        
        valid = True
        
        # Case 2: 2x + s
        for s in selected:
            s2 = 2 * x + s
            if s2 in triple_sums or s2 in new_sums:
                valid = False
                break
            new_sums.append(s2)
        
        if not valid:
            continue
            
        # Case 3: x + 2s
        for s in selected:
            s3 = x + 2 * s
            if s3 in triple_sums or s3 in new_sums:
                valid = False
                break
            new_sums.append(s3)
        
        if not valid:
            continue
            
        # Case 4: x + s1 + s2
        # s1, s2 from selected
        # We need to check all pairs (s1, s2) with s1 <= s2
        # Note: s1, s2 can be the same element in selected, or different.
        # The triple is {x, s1, s2}.
        # Since x is the new largest (we iterate ascending), x >= s1, x >= s2 is NOT necessarily true?
        # Wait, we iterate x from 1 upwards. So x is the largest element in the set {selected + [x]}.
        # So the triple is (s1, s2, x) with s1 <= s2 <= x.
        # The sums are x + s1 + s2.
        
        m = len(selected)
        for i in range(m):
            s1 = selected[i]
            for j in range(i, m):
                s2 = selected[j]
                s4 = x + s1 + s2
                if s4 in triple_sums or s4 in new_sums:
                    valid = False
                    break
                new_sums.append(s4)
            if not valid:
                break
        
        if valid:
            selected.append(x)
            triple_sums.update(new_sums)
            
    return selected
