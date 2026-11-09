import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Precompute AP constraints
    # For each x (1-based), list of (i1, i2, i3) 0-based indices
    # such that x-1 is the last element of a 4-AP.
    constraints = [[] for _ in range(128)]
    for x in range(1, 129):
        idx = x - 1
        for d in range(1, (x - 1) // 3 + 1):
            a = x - 3 * d
            if a < 1:
                break
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            constraints[idx].append((i1, i2, i3))

    def search_backtracking(max_c, time_limit):
        colors = [-1] * 128
        bt_start = time.time()
        
        # Optimization: Use a list of valid colors to try
        # Heuristic: try colors that are "popular" or just 0,1,2
        # For 3 colors, just iterate 0, 1, 2
        
        def backtrack(idx):
            if time.time() > bt_start + time_limit:
                return False
            
            if idx == 128:
                return True
            
            # Try all max_c colors
            # We can order them. For now, 0..max_c-1
            for c in range(max_c):
                valid = True
                for i1, i2, i3 in constraints[idx]:
                    c1 = colors[i1]
                    c2 = colors[i2]
                    c3 = colors[i3]
                    # Check if adding c results in < 3 distinct colors
                    # Equivalent to: c is in {c1, c2, c3} AND len({c1, c2, c3}) <= 2
                    # Or simpler: len({c1, c2, c3, c}) < 3
                    if len({c1, c2, c3, c}) < 3:
                        valid = False
                        break
                
                if valid:
                    colors[idx] = c
                    if backtrack(idx + 1):
                        return True
                    colors[idx] = -1
            
            return False

        if backtrack(0):
            return colors
        return None

    # Try to find a 3-coloring
    # 5 seconds is a reasonable budget. If it fails, we assume 3 is hard or impossible.
    res = search_backtracking(3, 5.0)
    if res:
        return res

    # Fallback: Greedy with 4 colors (seed logic)
    def try_construct_greedy(max_c):
        colors = [None] * 128
        used = set()
        for x in range(128):
            if time.time() > deadline:
                return None
            valid_colors = []
            for c in range(max_c):
                ok = True
                for i1, i2, i3 in constraints[x]:
                    c1, c2, c3 = colors[i1], colors[i2], colors[i3]
                    if len({c1, c2, c3, c}) < 3:
                        ok = False
                        break
                if ok:
                    valid_colors.append(c)
            
            if not valid_colors:
                return None
            
            # Heuristic: prefer used colors to minimize distinct count if max_c > needed
            # But here we fix max_c=4. If we use 3, great. If we use 4, ok.
            # Sorting by (0 if used else 1, c) encourages reusing colors.
            valid_colors.sort(key=lambda c: (0 if c in used else 1, c))
            chosen = valid_colors[0]
            colors[x] = chosen
            used.add(chosen)
            
        return colors

    res = try_construct_greedy(4)
    if res:
        return res

    # Fallback: Seed's original loop (slower, but robust)
    colors=[]
    for x in range(1,129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color);break
    return colors
