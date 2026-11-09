import time

# EVOLVE-BLOCK-START
def load(boxes, container):
    start = time.time()
    deadline = start + 110.0  # safety margin before 150s kill
    W, H, D = container
    CW, CH, CD = float(W), float(H), float(D)
    total_cap = W * H * D

    n = len(boxes)
    # orientation permutations of (w,h,d)
    def rots(w, h, d):
        s = [(w, h, d), (w, d, h), (h, w, d), (h, d, w), (d, w, h), (d, h, w)]
        out, seen = [], set()
        for r in s:
            if r not in seen:
                seen.add(r)
                out.append(r)
        return out

    # order: largest volume first, tie-break by max dimension descending
    order = sorted(range(n), key=lambda i: (-(boxes[i][0]*boxes[i][1]*boxes[i][2]),
                                            -max(boxes[i])))

    def try_shelf(order_seq):
        """Shelf packing with rotation; returns list of placements (len n)."""
        placed = [None]*n
        # shelves: each has (y_base, z_base, x_cursor, shelf_h, shelf_d)
        shelves = []
        cur_y = 0.0
        cur_z = 0.0
        for idx in order_seq:
            w, h, d = boxes[idx]
            best = None  # (score, x, y, z, rw, rh, rd)
            for (rw, rh, rd) in rots(w, h, d):
                # try existing shelves
                for si, (sy, sz, sx, sh, sd) in enumerate(shelves):
                    if sx + rw <= W and rh <= sh and rd <= sd:
                        # feasible in this shelf
                        # score: prefer tighter fit (smaller leftover x, smaller shelf)
                        score = (sx, sy, sz)
                        if best is None or (sx + rw) < (best[1] + best[3]):
                            best = (score, sx, sy, sz, rw, rh, rd, si)
                # try new shelf in current layer
                if cur_y + rh <= H and rd <= D:
                    # new shelf at x=0
                    if best is None or (0 + rw) < (best[1] + best[3]):
                        best = ((0, cur_y, cur_z), 0, cur_y, cur_z, rw, rh, rd, -1)
            if best is None:
                continue
            _, x, y, z, rw, rh, rd, si = best
            if si >= 0:
                shelves[si] = (y, z, x + rw, max(shelves[si][3], rh), max(shelves[si][4], rd))
            else:
                shelves.append((y, z, x + rw, rh, rd))
                if y + rh > cur_y + (shelves[-1][3]):
                    # track top of layer for layer advance
                    pass
            placed[idx] = (0, x, y, z, rw, rh, rd)
        # finalize: unplaced -> left out
        res = []
        for i in range(n):
            if placed[i] is None:
                res.append((-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]))
            else:
                res.append(placed[i])
        return res

    def filled_volume(placements):
        v = 0
        for p in placements:
            if p[0] == 0:
                v += p[4]*p[5]*p[6]
        return v

    # baseline shelf solution
    best = try_shelf(order)
    best_vol = filled_volume(best)

    # try a few alternative orderings (by different dimension) to diversify
    alt_orders = [
        sorted(range(n), key=lambda i: (-max(boxes[i]), -(boxes[i][0]*boxes[i][1]*boxes[i][2]))),
        sorted(range(n), key=lambda i: (-(boxes[i][0]+boxes[i][1]+boxes[i][2]), -(boxes[i][0]*boxes[i][1]*boxes[i][2]))),
        sorted(range(n), key=lambda i: (-(min(boxes[i])), -(boxes[i][0]*boxes[i][1]*boxes[i][2]))),
    ]
    for ao in alt_orders:
        if time.time() > deadline:
            break
        cand = try_shelf(ao)
        cv = filled_volume(cand)
        if cv > best_vol:
            best, best_vol = cand, cv

    # local improvement: try to re-insert left-out boxes by finding a free spot
    # Build an occupancy grid at coarse resolution to find gaps.
    def try_fill_gaps(placements):
        """Attempt to place leftover boxes into gaps using a coarse 3D grid."""
        # resolution: choose cell size ~ min dim / 10, capped
        cell = max(1, min(W, H, D) // 12)
        nx, ny, nz = (W + cell - 1)//cell, (H + cell - 1)//cell, (D + cell - 1)//cell
        occ = [[False]*nz for _ in range(ny)]
        # simpler: use a dict of occupied cells
        occset = set()
        for p in placements:
            if p[0] != 0:
                x, y, z, rw, rh, rd = p[1], p[2], p[3], p[4], p[5], p[6]
                for cx in range(x//cell, (x+rw+cell-1)//cell):
                    for cy in range(y//cell, (y+rh+cell-1)//cell):
                        for cz in range(z//cell, (z+rd+cell-1)//cell):
                            occset.add((cx, cy, cz))
        newpl = list(placements)
        # attempt in volume-desc order
        for idx in order:
            if newpl[idx][0] != -1:
                continue
            w, h, d = boxes[idx]
            placed = False
            for (rw, rh, rd) in rots(w, h, d):
                # scan candidate positions on grid
                for gx in range(0, W - rw + 1, cell):
                    if placed: break
                    for gy in range(0, H - rh + 1, cell):
                        if placed: break
                        for gz in range(0, D - rd + 1, cell):
                            # check overlap
                            ok = True
                            for cx in range(gx//cell, (gx+rw+cell-1)//cell):
                                if not ok: break
                                for cy in range(gy//cell, (gy+rh+cell-1)//cell):
                                    if not ok: break
                                    for cz in range(gz//cell, (gz+rd+cell-1)//cell):
                                        if (cx, cy, cz) in occset:
                                            ok = False
                                            break
                            if ok:
                                # place
                                for cx in range(gx//cell, (gx+rw+cell-1)//cell):
                                    for cy in range(gy//cell, (gy+rh+cell-1)//cell):
                                        for cz in range(gz//cell, (gz+rd+cell-1)//cell):
                                            occset.add((cx, cy, cz))
                                newpl[idx] = (0, gx, gy, gz, rw, rh, rd)
                                placed = True
                                break
            if time.time() > deadline:
                break
        return newpl

    # run gap fill once (time bounded)
    if time.time() < deadline:
        cand = try_fill_gaps(best)
        cv = filled_volume(cand)
        if cv > best_vol:
            best, best_vol = cand, cv

    return best
# EVOLVE-BLOCK-END
