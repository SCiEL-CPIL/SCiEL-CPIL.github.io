import time
import random
from math import gcd

def construct():
    deadline = time.time() + 85

    def _lcm(a, b):
        return a // gcd(a, b) * b

    n = 128
    all_mask = (1 << n) - 1

    # Precompute bad triples and 2-section conflict graph
    bad_triples = []
    conflict_pairs = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            ab = _lcm(a, b)
            for c in range(b + 1, n + 1):
                if _lcm(a, c) == ab and _lcm(b, c) == ab:
                    bad_triples.append((a, b, c))
                    conflict_pairs[a].add((b, c))
                    conflict_pairs[b].add((a, c))
                    conflict_pairs[c].add((a, b))

    # Build conflict_dict: for each x, dict u -> bitmask of v's (v != u) such that (x,u,v) is bad
    conflict_dict = [dict() for _ in range(n + 1)]
    for x in range(1, n + 1):
        for u, v in conflict_pairs[x]:
            if u not in conflict_dict[x]:
                conflict_dict[x][u] = 0
            conflict_dict[x][u] |= (1 << (v - 1))
            if v not in conflict_dict[x]:
                conflict_dict[x][v] = 0
            conflict_dict[x][v] |= (1 << (u - 1))

    # 2-section adjacency: adj[x] = bitmask of y != x such that x and y appear in some bad triple
    adj = [0] * (n + 1)
    for x in range(1, n + 1):
        m = 0
        for u in conflict_dict[x]:
            m |= (1 << (u - 1))
        adj[x] = m
    deg = [bin(adj[i]).count('1') for i in range(n + 1)]

    def can_add(x, mask):
        for u, v_mask in conflict_dict[x].items():
            if mask & (1 << (u - 1)):
                if mask & v_mask:
                    return False
        return True

    def popcount(x):
        return bin(x).count('1')

    def greedy_fill(mask, order):
        for x in order:
            if not (mask & (1 << (x - 1))):
                if can_add(x, mask):
                    mask |= (1 << (x - 1))
        return mask

    def local_search(mask, max_iters=80):
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 3:
            improved = False
            iters += 1
            # 1) Greedy fill
            for x in range(1, n + 1):
                if not (mask & (1 << (x - 1))):
                    if can_add(x, mask):
                        mask |= (1 << (x - 1))
                        improved = True
            if improved:
                continue

            # 2) Try 1-for-2 swap: remove x, add y, z
            s_list = [i for i in range(1, n + 1) if (mask & (1 << (i - 1)))]
            random.shuffle(s_list)
            for x in s_list:
                if time.time() > deadline - 3:
                    break
                mask_no_x = mask & ~(1 << (x - 1))
                new_mask = mask_no_x
                added = 0
                for y in range(1, n + 1):
                    if y == x:
                        continue
                    if not (new_mask & (1 << (y - 1))):
                        if can_add(y, new_mask):
                            new_mask |= (1 << (y - 1))
                            added += 1
                            if added == 2:
                                break
                if added >= 2 and popcount(new_mask) > popcount(mask):
                    mask = new_mask
                    improved = True
                    break
            if improved:
                continue

            # 3) Try 2-for-2 swap: remove x, y, add z, w
            if time.time() < deadline - 4:
                s_list2 = [i for i in range(1, n + 1) if (mask & (1 << (i - 1)))]
                random.shuffle(s_list2)
                for i in range(min(len(s_list2), 25)):
                    if time.time() > deadline - 4:
                        break
                    x = s_list2[i]
                    for j in range(i + 1, min(len(s_list2), i + 6)):
                        y = s_list2[j]
                        mask_no = mask & ~(1 << (x - 1)) & ~(1 << (y - 1))
                        new_mask = mask_no
                        added = 0
                        for z in range(1, n + 1):
                            if z == x or z == y:
                                continue
                            if not (new_mask & (1 << (z - 1))):
                                if can_add(z, new_mask):
                                    new_mask |= (1 << (z - 1))
                                    added += 1
                                    if added == 2:
                                        break
                        if added >= 2 and popcount(new_mask) > popcount(mask):
                            mask = new_mask
                            improved = True
                            break
                    if improved:
                        break
            if improved:
                continue

            # 4) Random perturbation: remove k random, refill
            k = random.choice([2, 3, 4])
            if len(s_list2) > k + 1:
                idxs = random.sample(range(len(s_list2)), k)
                pm = mask
                for idx in idxs:
                    pm &= ~(1 << (s_list2[idx] - 1))
                pm = greedy_fill(pm, list(range(1, n + 1)))
                if popcount(pm) > popcount(mask):
                    mask = pm
                    improved = True

        return mask

    # Greedy coloring upper bound on the 2-section conflict graph restricted to candidates
    def coloring_bound(cands):
        if cands == 0:
            return 0
        colors = [0] * (n + 1)
        remaining = cands
        upper = 0
        while remaining:
            upper += 1
            # Pick a vertex with highest degree in remaining
            best_v = -1
            best_d = -1
            t = remaining
            while t:
                lsb = t & (-t)
                v = lsb.bit_length()
                d = popcount(adj[v] & remaining)
                if d > best_d:
                    best_d = d
                    best_v = v
                t &= t - 1
            if best_v == -1:
                break
            colors[best_v] = upper
            remaining &= ~(1 << (best_v - 1))
            # Remove all neighbors of best_v from remaining (they can't share color)
            remaining &= ~adj[best_v]
        return upper

    # Branch and bound using 2-section conflict graph with coloring bound
    def bb(selected, cands, best_size, depth=0):
        if time.time() > deadline - 6:
            return selected, best_size
        if cands == 0:
            if popcount(selected) > best_size:
                best_size = popcount(selected)
                return selected, best_size
            return selected, best_size

        bound = popcount(selected) + coloring_bound(cands)
        if bound <= best_size:
            return selected, best_size

        if depth >= 20:
            # Greedy completion
            gm = selected
            rem = cands
            while rem:
                best_v = -1
                best_conf = float('inf')
                t = rem
                while t:
                    lsb = t & (-t)
                    v = lsb.bit_length()
                    cd = conflict_dict[v]
                    conf = 0
                    for u, vm in cd.items():
                        if rem & (1 << (u - 1)):
                            conf += popcount(rem & vm)
                    if conf < best_conf:
                        best_conf = conf
                        best_v = v
                    t &= t - 1
                if best_v == -1:
                    break
                if can_add(best_v, gm):
                    gm |= (1 << (best_v - 1))
                rem &= ~(1 << (best_v - 1))
            if popcount(gm) > best_size:
                best_size = popcount(gm)
                return gm, best_size
            return selected, best_size

        # Pick vertex with max degree in cands
        best_v = -1
        best_d = -1
        t = cands
        while t:
            lsb = t & (-t)
            v = lsb.bit_length()
            d = popcount(adj[v] & cands)
            if d > best_d:
                best_d = d
                best_v = v
            t &= t - 1

        if best_v == -1:
            return selected, best_size

        # Branch: include best_v (if possible)
        if can_add(best_v, selected):
            new_sel = selected | (1 << (best_v - 1))
            # New candidates: remove best_v and its 2-section neighbors (they conflict)
            new_cands = (cands & ~(1 << (best_v - 1))) & ~adj[best_v]
            # Also: removing best_v might free up some elements? No - 2-section is static.
            # But the 3-uniform constraint: elements that were only blocked via triples
            # with best_v are now potentially addable. The can_add check handles this.
            mask1, size1 = bb(new_sel, new_cands, best_size, depth + 1)
            if size1 > best_size:
                best_size = size1
                selected = mask1

        # Branch: exclude best_v
        new_cands2 = cands & ~(1 << (best_v - 1))
        mask2, size2 = bb(selected, new_cands2, best_size, depth + 1)
        if size2 > best_size:
            best_size = size2
            selected = mask2

        return selected, best_size

    best_mask = 0
    best_size = 0

    # Phase 1: Diverse greedy + local search
    num_restarts = 400
    for i in range(num_restarts):
        if time.time() > deadline - 30:
            break
        strat = i % 5
        if strat == 0:
            order = sorted(range(1, n + 1), key=lambda x: deg[x] + random.random() * 0.1)
        elif strat == 1:
            order = sorted(range(1, n + 1), key=lambda x: -deg[x] + random.random() * 0.1)
        elif strat == 2:
            order = list(range(1, n + 1))
            random.shuffle(order)
        elif strat == 3:
            order = sorted(range(1, n + 1), key=lambda x: (deg[x] % 7, random.random()))
        else:
            # Weighted by inverse degree
            order = list(range(1, n + 1))
            random.shuffle(order)

        mask = greedy_fill(0, order)
        mask = local_search(mask, max_iters=50)
        sz = popcount(mask)
        if sz > best_size:
            best_size = sz
            best_mask = mask

    # Phase 2: B&B from scratch
    if time.time() < deadline - 20:
        bm, bs = bb(0, all_mask, best_size, max_depth=18)
        if bs > best_size:
            best_size = bs
            best_mask = bm

    # Phase 3: B&B from current best (candidates = not in best_mask)
    if time.time() < deadline - 12:
        cands = all_mask & ~best_mask
        bm2, bs2 = bb(best_mask, cands, best_size, max_depth=14)
        if bs2 > best_size:
            best_size = bs2
            best_mask = bm2

    # Phase 4: Final polish
    if time.time() < deadline - 3:
        best_mask = local_search(best_mask, max_iters=120)

    result = [i for i in range(1, n + 1) if (best_mask & (1 << (i - 1)))]
    if not result:
        result = [1]
    return result
