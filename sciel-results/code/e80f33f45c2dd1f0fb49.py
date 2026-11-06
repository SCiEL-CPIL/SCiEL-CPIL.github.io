# EVOLVE-BLOCK-START
import time
import random

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    if W <= 0 or H <= 0 or D <= 0:
        return [(0, 0, 0, 0, w, h, d) for (w, h, d) in boxes]

    def vol(w, h, d):
        return w * h * d

    def find_spot(bin_items, w, h, d):
        """Return lowest-feasible (x,y,z) for a box, or None.
        Only test positions adjacent to a blocker face (existing box or wall)."""
        if w > W or h > H or d > D:
            return None
        blocked = list(bin_items)
        blocked.append((-1, 0, 0, 1, H + 1, D + 1))   # x=0 face
        blocked.append((0, -1, 0, W + 1, 1, D + 1))   # y=0 face
        blocked.append((0, 0, -1, W + 1, H + 1, 1))   # z=0 face

        best = None
        best_key = None
        for (bx, by, bz, bw, bh, bd) in blocked:
            cands = []
            if bx + bw <= W:
                cands.append((bx + bw, by, bz))
            if by + bh <= H:
                cands.append((bx, by + bh, bz))
            if bz + bd <= D:
                cands.append((bx, by, bz + bd))
            for (x, y, z) in cands:
                if x < 0 or y < 0 or z < 0:
                    continue
                if x + w > W or y + h > H or z + d > D:
                    continue
                ok = True
                for (ex, ey, ez, ew, eh, ed) in bin_items:
                    if (x < ex + ew and x + w > ex) and \
                       (y < ey + eh and y + h > ey) and \
                       (z < ez + ed and z + d > ez):
                        ok = False
                        break
                if not ok:
                    continue
                key = (x, y, z)
                if best_key is None or key < best_key:
                    best_key = key
                    best = (x, y, z)
        return best

    def try_order(order):
        """First-fit-decreasing over the given (idx,w,h,d) order.
        Returns (placements dict, bins list) or None on failure."""
        bins = []
        placements = {}
        for idx, w, h, d in order:
            placed = False
            for bi in range(len(bins)):
                pos = find_spot(bins[bi], w, h, d)
                if pos is not None:
                    x, y, z = pos
                    bins[bi].append((x, y, z, w, h, d))
                    placements[idx] = (bi, x, y, z, w, h, d)
                    placed = True
                    break
            if not placed:
                bi = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                placements[idx] = (bi, 0, 0, 0, w, h, d)
        return placements, bins

    def score_of(placements):
        if not placements:
            return 0.0
        bin_ids = set(p[0] for p in placements.values())
        B = len(bin_ids)
        last_bin = max(bin_ids)
        last_vol = 0
        for p in placements.values():
            if p[0] == last_bin:
                last_vol += p[4] * p[5] * p[6]
        cv = W * H * D
        fill = 1.0 if cv == 0 else last_vol / cv
        return B + 0.1 * (1 - fill)

    def to_output(placements):
        return [placements[i] for i in range(n)]

    def local_search(placements, bins, order, max_iters=250):
        """Relocate boxes into a less-full bin (reduces B) or into the last bin (raises fill)."""
        cur = score_of(placements)
        improved = True
        it = 0
        while improved and it < max_iters and time.time() < deadline - 3:
            improved = False
            it += 1
            bin_members = {}
            for idx in range(n):
                bin_members.setdefault(placements[idx][0], []).append(idx)
            sizes = {b: len(m) for b, m in bin_members.items()}
            last_bin = max(bin_members)
            for bi in sorted(bin_members, key=lambda b: -sizes[b]):
                if bi == last_bin or sizes[bi] <= 1:
                    continue
                members = bin_members[bi][:]
                for idx in members:
                    if time.time() > deadline - 3:
                        return placements, bins, cur
                    p = placements[idx]
                    x, y, z, w, h, d = p[1], p[2], p[3], p[4], p[5], p[6]
                    bins[bi].remove((x, y, z, w, h, d))
                    sizes[bi] -= 1
                    if sizes[bi] == 0:
                        del bins[bi]
                        del sizes[bi]
                    newbin = None
                    for tb in sorted(sizes, key=lambda b: sizes[b]):
                        if sizes[tb] >= sizes.get(bi, 0):
                            continue
                        pos = find_spot(bins[tb], w, h, d)
                        if pos is not None:
                            newbin = tb
                            break
                    if newbin is None:
                        if bi in bins:
                            bins[bi].append((x, y, z, w, h, d))
                            sizes[bi] = sizes.get(bi, 0) + 1
                        else:
                            bins[bi] = [(x, y, z, w, h, d)]
                            sizes[bi] = 1
                        placements[idx] = (bi, x, y, z, w, h, d)
                        continue
                    x2, y2, z2 = find_spot(bins[newbin], w, h, d)
                    bins[newbin].append((x2, y2, z2, w, h, d))
                    sizes[newbin] = sizes.get(newbin, 0) + 1
                    placements[idx] = (newbin, x2, y2, z2, w, h, d)
                    ns = score_of(placements)
                    if ns < cur:
                        cur = ns
                        improved = True
                    else:
                        bins[newbin].remove((x2, y2, z2, w, h, d))
                        sizes[newbin] -= 1
                        if sizes[newbin] == 0:
                            del bins[newbin]
                            del sizes[newbin]
                        if bi in bins:
                            bins[bi].append((x, y, z, w, h, d))
                            sizes[bi] = sizes.get(bi, 0) + 1
                        else:
                            bins[bi] = [(x, y, z, w, h, d)]
                            sizes[bi] = 1
                        placements[idx] = (bi, x, y, z, w, h, d)
        return placements, bins, cur

    indexed = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]

    best_placements = None
    best_bins = None
    best_score = float('inf')
    best_order = None

    def consider(order):
        nonlocal best_placements, best_bins, best_score, best_order
        try:
            placements, bins = try_order(order)
        except Exception:
            return
        if placements is None or len(placements) != n:
            return
        s = score_of(placements)
        if s < best_score - 1e-9:
            best_score = s
            best_placements = placements
            best_bins = bins
            best_order = order

    strategies = [
        lambda a: sorted(a, key=lambda x: -vol(x[1], x[2], x[3])),
        lambda a: sorted(a, key=lambda x: (-(max(x[1], x[2], x[3])), -vol(x[1], x[2], x[3]))),
        lambda a: sorted(a, key=lambda x: (-(x[1] + x[2] + x[3]), -vol(x[1], x[2], x[3]))),
        lambda a: sorted(a, key=lambda x: (-(x[1] * x[2]), -x[3])),
        lambda a: sorted(a, key=lambda x: (-(x[2] * x[3]), -x[1])),
        lambda a: sorted(a, key=lambda x: (-(x[1] * x[3]), -x[2])),
        lambda a: sorted(a, key=lambda x: (-(x[1] * x[2] * x[3]), -max(x[1], x[2], x[3]))),
        lambda a: sorted(a, key=lambda x: (-(min(x[1], x[2], x[3])), -vol(x[1], x[2], x[3]))),
    ]

    for strat in strategies:
        if time.time() > deadline - 12:
            break
        consider(strat(indexed))

    if best_placements is not None and time.time() < deadline - 15:
        bins_by_id = {}
        for p in best_placements.values():
            bins_by_id.setdefault(p[0], []).append((p[1], p[2], p[3], p[4], p[5], p[6]))
        bins_list = [bins_by_id[k] for k in sorted(bins_by_id)]
        remap = {k: i for i, k in enumerate(sorted(bins_by_id))}
        for idx in range(n):
            p = best_placements[idx]
            best_placements[idx] = (remap[p[0]], p[1], p[2], p[3], p[4], p[5], p[6])
        best_placements, best_bins, best_score = local_search(best_placements, bins_list, best_order, max_iters=250)

    if best_placements is not None and time.time() < deadline - 20:
        base = strategies[0](indexed)
        rng = random.Random(12345)
        restarts = 0
        while time.time() < deadline - 6 and restarts < 60:
            restarts += 1
            order = base[:]
            rng.shuffle(order)
            order = sorted(order, key=lambda x: (rng.random(), -vol(x[1], x[2], x[3])))
            consider(order)

    if best_placements is None:
        best_placements = {}
        b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
        for i, (w, h, d) in enumerate(boxes):
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            best_placements[i] = (b, x, y, z, w, h, d)
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)

    return to_output(best_placements)

# EVOLVE-BLOCK-END
