def construct():
    n = 256
    S = set()
    candidates = set(range(1, n + 1))
    
    def creates_4ap(S, x):
        # Check if adding x to S creates a 4-AP
        # x is in [1, n]
        for d in range(1, n + 1):
            # x is 1st: x, x+d, x+2d, x+3d
            if x + d <= n and x + 2 * d <= n and x + 3 * d <= n:
                if (x + d in S) and (x + 2 * d in S) and (x + 3 * d in S):
                    return True
            # x is 2nd: x-d, x, x+d, x+2d
            if x - d >= 1:
                if (x - d in S) and (x + d in S) and (x + 2 * d <= n and x + 2 * d in S):
                    return True
            # x is 3rd: x-2d, x-d, x, x+d
            if x - 2 * d >= 1:
                if (x - 2 * d in S) and (x - d in S) and (x + d in S):
                    return True
            # x is 4th: x-3d, x-2d, x-d, x
            if x - 3 * d >= 1:
                if (x - 3 * d in S) and (x - 2 * d in S) and (x - d in S):
                    return True
        return False

    def calc_gain(S, x):
        gain = 0
        for d in range(1, n + 1):
            # x is middle: x-d, x, x+d
            if x - d >= 1 and x + d <= n:
                if (x - d in S) and (x + d in S):
                    gain += 1
            # x is first: x, x+d, x+2d
            if x + d <= n and x + 2 * d <= n:
                if (x + d in S) and (x + 2 * d in S):
                    gain += 1
            # x is last: x-2d, x-d, x
            if x - 2 * d >= 1 and x - d >= 1:
                if (x - 2 * d in S) and (x - d in S):
                    gain += 1
        return gain

    best_gain = 0
    best_x = None
    
    while candidates:
        best_gain = -1
        best_x = None
        
        for x in candidates:
            if creates_4ap(S, x):
                continue
            
            g = calc_gain(S, x)
            
            if g > best_gain:
                best_gain = g
                best_x = x
                
        if best_x is None:
            # No valid element? Or all gains are <= -1?
            # If no element is valid, we break.
            # If elements are valid but gain is 0, should we add?
            # Adding 0 gain elements doesn't increase score. 
            # It might block future gains. 
            # So we should stop if max gain <= 0?
            # Wait, if best_gain is 0, it means we found a valid element with 0 gain.
            # If we add it, score doesn't change. 
            # But maybe it enables a 3-AP later? No, 3-APs are local.
            # Adding an element only creates NEW 3-APs involving that element.
            # It doesn't create new 3-APs between existing elements.
            # So if gain is 0, it adds nothing to the count.
            # Therefore, we should only add if gain > 0.
            # However, if we stop, we might have left valid elements.
            # But they don't contribute to the score.
            # So stopping is correct for maximizing the COUNT of 3-APs.
            
            # Check if best_gain was -1 (no valid x) or 0 (valid but no gain)
            if best_gain <= 0:
                break
                
        S.add(best_x)
        candidates.remove(best_x)

    return sorted(S)
