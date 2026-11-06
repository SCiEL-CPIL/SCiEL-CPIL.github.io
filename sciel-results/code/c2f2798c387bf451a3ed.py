def construct():
    n = 128
    
    # Precompute 4-APs ending at each position
    ap_ending = [[] for _ in range(n + 1)]
    for x in range(1, n + 1):
        for d in range(1, (x - 1) // 3 + 1):
            a = x - 3 * d
            if a >= 1:
                ap_ending[x].append((a, a + d, a + 2 * d))
                
    # Precompute 3-APs ending at each position
    tp_ending = [[] for _ in range(n + 1)]
    for x in range(1, n + 1):
        for d in range(1, (x - 1) // 2 + 1):
            a = x - 2 * d
            if a >= 1:
                tp_ending[x].append((a, a + d))

    def solve(k):
        colors = [-1] * (n + 1)
        
        def backtrack(x):
            if x > n:
                return True
            
            for c in range(k):
                valid = True
                # Check 4-APs ending at x
                for a, b, c3 in ap_ending[x]:
                    if len({colors[a], colors[b], colors[c3], c}) < 3:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check 3-APs ending at x (to avoid monochromatic 3-APs which are dead ends)
                for a, b in tp_ending[x]:
                    if colors[a] == colors[b] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                colors[x] = c
                if backtrack(x + 1):
                    return True
                colors[x] = -1
            
            return False
        
        if backtrack(1):
            return colors[1:]
        
        return None

    # Try to find a solution with the minimum number of colors
    for k in range(3, 8):
        res = solve(k)
        if res is not None:
            return res
            
    # Fallback: If no solution found with up to 7 colors, 
    # return a random valid coloring or raise exception.
    # For n=128, a solution with 4 or 5 colors should exist.
    raise Exception("No solution found")
