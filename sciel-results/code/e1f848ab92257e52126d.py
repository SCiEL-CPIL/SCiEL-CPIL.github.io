def construct():
    from fractions import Fraction
    import time
    
    start_time = time.time()
    deadline = start_time + 55  # 55 second deadline for safety
    
    denominators = list(range(2, 49))
    n = len(denominators)
    reciprocals = [Fraction(1, d) for d in denominators]
    
    # Initial greedy: include in order if sum stays < 1
    selected = [False] * n
    total = Fraction(0)
    for i in range(n):
        if total + reciprocals[i] < 1:
            selected[i] = True
            total += reciprocals[i]
    
    best_deficit = 1 - total
    best_selected = selected[:]
    
    # Local search: try swaps to minimize deficit (keep sum < 1)
    # A swap: remove i (included), add j (excluded)
    # New total = total - recip[i] + recip[j]
    # Must have new_total < 1, and new_deficit < best_deficit
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try all single swaps: remove one included, add one excluded
        included = [i for i in range(n) if selected[i]]
        excluded = [i for i in range(n) if not selected[i]]
        
        for i in included:
            for j in excluded:
                new_total = total - reciprocals[i] + reciprocals[j]
                if new_total < 1:
                    new_deficit = 1 - new_total
                    if new_deficit < best_deficit:
                        # Apply swap
                        selected[i] = False
                        selected[j] = True
                        total = new_total
                        best_deficit = new_deficit
                        best_selected = selected[:]
                        improved = True
                        # Rebuild included/excluded lists
                        included = [k for k in range(n) if selected[k]]
                        excluded = [k for k in range(n) if not selected[k]]
                        break
            if improved:
                break
        
        # Also try single additions (if possible) - but sum must stay < 1
        # And single removals won't help (increases deficit)
        
        # Try double swaps: remove i1, add j1, remove i2, add j2
        if time.time() < deadline:
            included = [i for i in range(n) if selected[i]]
            excluded = [i for i in range(n) if not selected[i]]
            for i1 in included:
                for j1 in excluded:
                    t1 = total - reciprocals[i1] + reciprocals[j1]
                    if t1 >= 1:
                        continue
                    for i2 in included:
                        if i2 == i1:
                            continue
                        for j2 in excluded:
                            if j2 == j1:
                                continue
                            t2 = t1 - reciprocals[i2] + reciprocals[j2]
                            if t2 < 1:
                                new_deficit = 1 - t2
                                if new_deficit < best_deficit:
                                    selected[i1] = False
                                    selected[j1] = True
                                    selected[i2] = False
                                    selected[j2] = True
                                    total = t2
                                    best_deficit = new_deficit
                                    best_selected = selected[:]
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
    
    # Convert to zero-based indices
    result = [i for i in range(n) if best_selected[i]]
    
    # Verify
    if not result:
        # Fallback: just pick a single small denominator
        result = [0]  # denominator 2, sum = 1/2, deficit = 1/2
    
    return result
