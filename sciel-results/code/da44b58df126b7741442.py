import random
import time

def construct():
    n = 128
    start_time = time.time()
    time_budget = 52

    aps = []
    for a in range(1, n + 1):
        for d in range(1, n + 1):
            if a + 3 * d <= n:
                aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    m = len(aps)

    pos_aps = [None] * n
    for i in range(n):
        pos_aps[i] = []
    for i, ap in enumerate(aps):
        for p in ap:
            pos_aps[p].append(i)

    def ap_ok(colors, ap):
        ca = colors[ap[0]]; cb = colors[ap[1]]
        cc = colors[ap[2]]; cd = colors[ap[3]]
        distinct = 1
        if cb != ca:
            distinct += 1
        if cc != ca and cc != cb:
            distinct += 1
        if cd != ca and cd != cb and cd != cc:
            distinct += 1
        return distinct >= 3

    def full_valid(colors):
        for i in range(m):
            if not ap_ok(colors, aps[i]):
                return False
        return True

    def try_reduce(colors):
        used_colors = sorted(set(colors))
        if len(used_colors) <= 3:
            return colors
        for target in used_colors:
            positions = [i for i in range(n) if colors[i] == target]
            if not positions:
                continue
            new_colors = colors[:]
            success = True
            for pos in positions:
                placed = False
                for c in used_colors:
                    if c == target:
                        continue
                    new_colors[pos] = c
                    ok = True
                    for i in pos_aps[pos]:
                        if not ap_ok(new_colors, aps[i]):
                            ok = False
                            break
                    if ok:
                        placed = True
                        break
                if not placed:
                    success = False
                    break
            if success and full_valid(new_colors):
                return new_colors
        return colors

    best = None
    best_count = n + 1

    for k in range(3, 8):
        if time.time() - start_time > time_budget:
            break
        trials = 35 if k == 3 else 25
        for trial in range(trials):
            if time.time() - start_time > time_budget:
                break
            random.seed(trial * 7919 + k * 131)
            colors = [random.randint(0, k - 1) for _ in range(n)]

            viol_set = set()
            for i, ap in enumerate(aps):
                if not ap_ok(colors, ap):
                    viol_set.add(i)

            for it in range(300000):
                if time.time() - start_time > time_budget:
                    break
                if not viol_set:
                    break
                viol_ap = None
                for _ in range(8):
                    r = random.randrange(m)
                    if r in viol_set:
                        viol_ap = r
                        break
                if viol_ap is None:
                    viol_ap = next(iter(viol_set))
                ap = aps[viol_ap]
                fixed = False
                positions = list(ap)
                random.shuffle(positions)
                for pos in positions:
                    old = colors[pos]
                    for c in range(k):
                        if c == old:
                            continue
                        colors[pos] = c
                        for i in pos_aps[pos]:
                            if ap_ok(colors, aps[i]):
                                viol_set.discard(i)
                            else:
                                viol_set.add(i)
                        if viol_ap not in viol_set:
                            fixed = True
                            break
                        colors[pos] = old
                        for i in pos_aps[pos]:
                            if ap_ok(colors, aps[i]):
                                viol_set.discard(i)
                            else:
                                viol_set.add(i)
                    if fixed:
                        break
                if not fixed:
                    pos = random.randint(0, n - 1)
                    old = colors[pos]
                    colors[pos] = random.randint(0, k - 1)
                    if colors[pos] != old:
                        for i in pos_aps[pos]:
                            if ap_ok(colors, aps[i]):
                                viol_set.discard(i)
                            else:
                                viol_set.add(i)
            if not viol_set:
                colors = try_reduce(colors)
                used = len(set(colors))
                if used < best_count:
                    best_count = used
                    best = colors[:]
                if best_count <= k:
                    break
        if best is not None and best_count <= k:
            break

    if best is None:
        colors = []
        for x in range(1, n + 1):
            for color in range(n):
                if all(len({color} | {colors[x - 1 - j * d] for j in (1, 2, 3)}) >= 3
                       for d in range(1, (x - 1) // 3 + 1)):
                    colors.append(color)
                    break
        return colors
    return best
