def construct():
    n = 512
    # Base 3 construction
    selected = set()
    for x in range(1, n + 1):
        temp = x
        valid = True
        while temp > 0:
            if temp % 3 == 2:
                valid = False
                break
            temp //= 3
        if valid:
            selected.add(x)
    
    # Greedy completion
    # Try to add remaining numbers
    # To ensure no 4-AP, we check if adding x creates a 4-AP with existing elements
    # A 4-AP is a, a+d, a+2d, a+3d.
    # If we add x, it can be part of a 4-AP in 4 positions.
    # 1. x is a: a, a+d, a+2d, a+3d. Check if a+d, a+2d, a+3d in selected.
    # 2. x is a+d: a, x, x+d, x+2d. Check if a, x+d, x+2d in selected.
    # 3. x is a+2d: a, a+d, x, x+d. Check if a, a+d, x+d in selected.
    # 4. x is a+3d: a, a+d, a+2d, x. Check if a, a+d, a+2d in selected.
    
    # To make it efficient, we can just check all possible 4-APs involving x.
    # Or simpler: for each d, check if the other 3 terms are in selected.
    
    candidates = [x for x in range(1, n + 1) if x not in selected]
    # Sort candidates? Maybe smaller d first?
    # Just iterate
    
    added = True
    while added:
        added = False
        for x in candidates[:]: # iterate over copy
            if x in selected:
                continue
            ok = True
            # Check all d
            # Max d such that 4 terms fit in [1, n]
            # x can be 1st term: x, x+d, x+2d, x+3d <= n => d <= (n-x)//3
            # x can be 2nd term: x-d, x, x+d, x+2d >= 1 => d <= x-1
            # x can be 3rd term: x-2d, x-d, x, x+d >= 1 => 2d <= x-1 => d <= (x-1)//2
            # x can be 4th term: x-3d, x-2d, x-d, x >= 1 => 3d <= x-1 => d <= (x-1)//3
            
            # Combine ranges
            max_d = n - 1 # safe upper bound
            # Actually we can just check d up to n//3
            
            for d in range(1, n // 3 + 1):
                # Check if x is 1st term
                if x + 3 * d <= n:
                    if (x + d) in selected and (x + 2 * d) in selected and (x + 3 * d) in selected:
                        ok = False
                        break
                # Check if x is 2nd term
                if x - d >= 1:
                    if (x - d) in selected and (x + d) in selected and (x + 2 * d) in selected:
                        ok = False
                        break
                # Check if x is 3rd term
                if x - 2 * d >= 1:
                    if (x - 2 * d) in selected and (x - d) in selected and (x + d) in selected:
                        ok = False
                        break
                # Check if x is 4th term
                if x - 3 * d >= 1:
                    if (x - 3 * d) in selected and (x - 2 * d) in selected and (x - d) in selected:
                        ok = False
                        break
            
            if ok:
                selected.add(x)
                candidates.remove(x)
                added = True
                
    return sorted(selected)
