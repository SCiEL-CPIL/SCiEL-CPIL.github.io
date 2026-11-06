def construct():
    n = 128
    colors = [-1] * (n + 1)
    used_colors = set()
    
    for x in range(1, n + 1):
        candidates = sorted(used_colors)
        for c in range(128):
            if c not in used_colors:
                candidates.append(c)
        
        valid = False
        for c in candidates:
            ok = True
            # Check 4-term APs ending at x
            for d in range(1, (x - 1) // 3 + 1):
                a = x - 3 * d
                if a < 1:
                    break
                c1 = colors[a]
                c2 = colors[a + d]
                c3 = colors[a + 2 * d]
                if len({c1, c2, c3, c}) < 3:
                    ok = False
                    break
            if not ok:
                continue
            # Check that we don't create a monochromatic prefix for future 4-term APs
            for d in range(1, n - x + 1):
                if x - 2 * d < 1:
                    break
                if x + d > n:
                    continue
                c1 = colors[x - 2 * d]
                c2 = colors[x - d]
                if c1 == c2 == c:
                    ok = False
                    break
            if ok:
                colors[x] = c
                used_colors.add(c)
                valid = True
                break
        
        if not valid:
            raise Exception("No valid color found at position %d" % x)
    
    return colors[1:]
