def construct():
    import time
    import random

    t0 = time.time()
    DEADLINE = t0 + 55.0
    n = 128

    aps = []
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    m = len(aps)

    pos_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for p in ap:
            pos_aps[p].append(idx)

    def is_bad_ap(colors, ap):
        a, b, c, d = ap
        c0, c1, c2, c3 = colors[a], colors[b], colors[c], colors[d]
        return len({c0, c1, c2, c3}) < 3

    def try_find_coloring(k, budget):
        end_time = t0 + budget
        best_colors = None
        best_du = k + 1

        while time.time() < end_time:
            if random.random() < 0.5:
                colors = [i % k for i in range(n)]
            else:
                colors = [random.randrange(k) for _ in range(n)]

            vset = set(i for i in range(m) if is_bad_ap(colors, aps[i]))
            stall = 0

            while time.time() < end_time and vset:
                ei = random.choice(list(vset))
                ap = aps[ei]
                cur = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}

                best_delta = 10**9
                best_mv = None

                # Only consider colors not in current AP's color set to maximize chance of fixing
                # But we must evaluate delta for all to avoid breaking others
                for p in ap:
                    for c in range(k):
                        if c == colors[p]:
                            continue
                        dlt = 0
                        old_c = colors[p]
                        colors[p] = c
                        for idx in pos_aps[p]:
                            a2 = aps[idx]
                            was = idx in vset
                            now = is_bad_ap(colors, a2)
                            if was and not now:
                                dlt -= 1
                            elif (not was) and now:
                                dlt += 1
                        colors[p] = old_c

                        if dlt < best_delta:
                            best_delta = dlt
                            best_mv = (p, c)

                if best_mv is None:
                    break

                p, c = best_mv
                colors[p] = c

                for idx in pos_aps[p]:
                    a2 = aps[idx]
                    was = idx in vset
                    now = is_bad_ap(colors, a2)
                    if was and not now:
                        vset.discard(idx)
                    elif (not was) and now:
                        vset.add(idx)

                if best_delta < 0:
                    stall = 0
                else:
                    stall += 1
                    if stall > 1000:
                        for _ in range(10):
                            p2 = random.randrange(n)
                            c2 = random.randrange(k)
                            if c2 != colors[p2]:
                                colors[p2] = c2
                                for idx in pos_aps[p2]:
                                    a2 = aps[idx]
                                    was = idx in vset
                                    now = is_bad_ap(colors, a2)
                                    if was and not now:
                                        vset.discard(idx)
                                    elif (not was) and now:
                                        vset.add(idx)
                        stall = 0

            if not vset:
                du = len(set(colors))
                if du < best_du:
                    best_du = du
                    best_colors = colors[:]
                if k == 3:
                    return best_colors

        return best_colors

    res = try_find_coloring(3, 45.0)
    if res is not None:
        return res

    res = try_find_coloring(4, 5.0)
    if res is not None:
        return res

    # Fallback
    return [i % 3 for i in range(128)]
