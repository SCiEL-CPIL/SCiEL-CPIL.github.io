for _ in range(200):
    if time.time() > deadline - 4: break
    changed = False
    maxc = int(c.max())
    for i in range(N):
        cur = c[i]
        best_new = cur
        for new in range(maxc + 1):
            if new == cur: continue
            c[i] = new
            if pos_all_valid(c, i):
                best_new = new
                break  # smallest valid
            c[i] = cur
        c[i] = best_new
        if best_new != cur:
            changed = True
    if not changed: break
