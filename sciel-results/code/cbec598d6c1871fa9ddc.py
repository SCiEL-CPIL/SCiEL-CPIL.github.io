# EVOLVE-BLOCK-START
def construct():
    n = 256
    # Start with all numbers
    selected = set(range(1, n + 1))
    
    # We need to efficiently find 3-APs and remove elements.
    # A 3-AP is (a, b, c) with a < b < c and 2b = a + c.
    # We want to remove the largest element c if possible, to preserve 1/a and 1/b.
    
    # To speed up, we can maintain a list of active numbers.
    # Finding APs:
    # Iterate b from 1 to n.
    # For each a < b in selected, c = 2b - a.
    # If c in selected and c > b, then (a, b, c) is an AP.
    
    # Optimization:
    # We can just loop until no AP is found.
    # In each pass, find the first AP, remove its max element, and restart the search?
    # Or find all APs, pick the one with the smallest max element?
    # Let's try: Find the AP with the smallest maximum element. Removing the smallest possible "large" number is safest?
    # Actually, if we have AP (1, 2, 3) and (2, 3, 4), removing 3 breaks both.
    # If we just pick the first AP found, it might be (1, 100, 199), removing 199.
    # Then later we find (1, 2, 3) and remove 3.
    # The order might matter.
    
    # Let's try a strategy:
    # 1. Identify all "bad" triples.
    # 2. Pick the triple (a,b,c) with a<b<c.
    # 3. Remove c.
    # 4. Repeat.
    
    # To make it better, we can prioritize removing the element with the smallest weight (largest value) 
    # from the set of all elements involved in any AP?
    # No, we must break every AP.
    
    # Let's stick to: Find any AP, remove its largest element.
    # To be smarter, find the AP where the largest element is smallest.
    # This ensures we remove "small" large numbers first, preserving "big" large numbers?
    # No, we want to preserve small numbers (high weight).
    # Removing 3 is worse than removing 256.
    # So we should avoid removing small numbers if possible.
    # If we have AP (1,2,3) and (1,255,256), we should remove 3 from the first? No, 3 is small (high weight). 
    # Wait, 1/3 is large. We want to keep 3.
    # So if we have AP (1,2,3), we should remove 3? Or 1? Or 2?
    # We must remove one. 3 has the smallest weight (1/3 < 1/2 < 1/1).
    # So removing 3 is the best choice for that AP.
    
    # So the rule "remove the largest element of the AP" is correct for a single AP.
    # The question is the order of processing APs.
    # If we process APs with smaller max-elements first, we remove smaller numbers earlier.
    # Example: AP (1,2,3) max=3. AP (100, 101, 102) max=102.
    # If we remove 3 first, we lose 1/3.
    # If we remove 102 first, we lose 1/102.
    # Then we still have to handle (1,2,3). We remove 3.
    # Total loss: 1/3 + 1/102.
    # If we had a choice to remove 2 instead of 3 in (1,2,3)? No, 2 is smaller (higher weight).
    # So removing 3 is always better than removing 1 or 2 for that AP.
    
    # So the local decision is always: remove max(a,b,c).
    # The global strategy is the order.
    # If we remove elements in increasing order of their value (1, 2, 3...), we might remove small numbers that are part of many APs.
    # If we remove elements in decreasing order of their value (256, 255...), we remove large numbers (low weight) first.
    # This seems much better!
    
    # Strategy:
    # Check if 256 is in any AP. If yes, remove it.
    # Check if 255 is in any AP. If yes, remove it.
    # ...
    # Check if 1 is in any AP. If yes, remove it.
    
    # This ensures we only remove small numbers if they are "forced" to be removed (i.e., they form an AP with two larger numbers that we kept? No, we removed larger numbers first).
    # If we remove 256, it might break APs involving 256.
    # If 1 is part of an AP (1, x, y), and x,y < 256, and we didn't remove x or y, then 1 is still in an AP.
    # But if we removed x or y earlier (because they were larger than 1? No, x,y > 1), then 1 might be safe.
    # Since we iterate from N down to 1, when we check $k$, all numbers $> k$ have already been considered and potentially removed.
    # So if $k$ is in an AP with $a, b$ where $a,b > k$, those $a,b$ might have been removed.
    # If $k$ is in an AP with $a < k < b$, then $b > k$ was checked earlier. If $b$ was removed, the AP is broken.
    # If $b$ was kept, then the AP is still valid (assuming $a$ is kept, which it will be if we haven't reached it yet).
    # So this greedy "sweep from top to bottom" is a very strong heuristic.
    
    # Algorithm:
    # S = {1..N}
    # for k in range(N, 0, -1):
    #     if k in S:
    #         check if k is part of any 3-AP in S.
    #         if yes:
    #             remove k from S
    # return S
    
    # How to check if k is part of any 3-AP in S efficiently?
    # k can be the middle: a, k, c -> a = 2k - c, c = 2k - a.
    #   Check if exists a in S, a < k, such that c = 2k - a in S and c > k.
    #   Or exists c in S, c > k, such that a = 2k - c in S and a < k.
    # k can be the first: k, b, c -> b = k + d, c = k + 2d.
    #   Check if exists d > 0 such that k+d in S and k+2d in S.
    # k can be the last: a, b, k -> b = k - d, a = k - 2d.
    #   Check if exists d > 0 such that k-d in S and k-2d in S.
    
    # Since we iterate k downwards, S changes.
    # This is O(N^2) or O(N^3) worst case, but N=256 is small.
    # O(N^2) check for each k -> O(N^3) total ~ 16 million ops. Very fast.
    
    selected = set(range(1, n + 1))
    
    for k in range(n, 0, -1):
        if k not in selected:
            continue
            
        # Check if k is in any 3-AP
        in_ap = False
        
        # Case 1: k is the middle element (a, k, c)
        # a = 2k - c, c = 2k - a
        # We need a, c in selected, a < k < c
        # a can range from 1 to k-1
        # c = 2k - a
        for a in range(1, k):
            if a in selected:
                c = 2 * k - a
                if c > k and c in selected:
                    in_ap = True
                    break
        if in_ap:
            selected.remove(k)
            continue
            
        # Case 2: k is the first element (k, b, c)
        # b = k + d, c = k + 2d
        # d > 0
        # Max d such that k + 2d <= n
        max_d = (n - k) // 2
        for d in range(1, max_d + 1):
            b = k + d
            c = k + 2 * d
            if b in selected and c in selected:
                in_ap = True
                break
        if in_ap:
            selected.remove(k)
            continue
            
        # Case 3: k is the last element (a, b, k)
        # b = k - d, a = k - 2d
        # d > 0
        # Max d such that k - 2d >= 1
        max_d = (k - 1) // 2
        for d in range(1, max_d + 1):
            b = k - d
            a = k - 2 * d
            if a in selected and b in selected:
                in_ap = True
                break
        if in_ap:
            selected.remove(k)
            continue
            
    return list(selected)
# EVOLVE-BLOCK-END
