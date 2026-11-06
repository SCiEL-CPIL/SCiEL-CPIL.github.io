import random

def construct():
    n = 128
    k = 7
    
    aps = []
    pos_aps = [[] for _ in range(n)]
    for a in range(1, n+1):
        for d in range(1, (n - a)//3 + 1):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            if e <= n:
                idx = len(aps)
                aps.append((a-1, b-1, c-1, e-1))
                for pos in (a-1, b-1, c-1, e-1):
                    pos_aps[pos].append(idx)
    
    best_colors = None
    best_count = k + 1
    
    for trial in range(40):
        random.seed(trial * 7919 + 104729)
        colors = [random.randint(0, k-1) for _ in range(n)]
        
        for iteration in range(300):
            fixed = False
            for idx, (i, j, l, m) in enumerate(aps):
                if len({colors[i], colors[j], colors[l], colors[m]}) >= 3:
                    continue
                for pos in (i, j, l, m):
                    for new_c in range(k):
                        if new_c == colors[pos]:
                            continue
                        old_c = colors[pos]
                        colors[pos] = new_c
                        ok = True
                        for ap_idx in pos_aps[pos]:
                            ii, jj, ll, mm = aps[ap_idx]
                            if len({colors[ii], colors[jj], colors[ll], colors[mm]}) < 3:
                                ok = False
                                break
                        if ok:
                            fixed = True
                            break
                        else:
                            colors[pos] = old_c
                    if fixed:
                        break
                if fixed:
                    break
            if not fixed:
                break
        
        used = len(set(colors))
        if used < best_count:
            best_count = used
            best_colors = colors[:]
    
    if best_colors is None:
        colors = []
        for x in range(1, n+1):
            found = False
            for c in range(k):
                valid = True
                for d in range(1, (x-1)//3 + 1):
                    a = x - 3*d
                    if a >= 1:
                        if len({c, colors[a-1], colors[a-1+d], colors[a-1+2*d]}) < 3:
                            valid = False
                            break
                if valid:
                    colors.append(c)
                    found = True
                    break
            if not found:
                colors.append(0)
        best_colors = colors
    
    return best_colors
