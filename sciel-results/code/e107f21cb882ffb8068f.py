"""3D Bin Packing: best-fit decreasing with consolidation pass."""
import time

# EVOLVE-BLOCK-START

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    start_time = time.time()
    deadline = start_time + 110.0

    indexed_boxes = [(i, w, h, d, w * h * d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[4], reverse=True)

    result = [None] * n
    bins = []

    def find_pos(w, h, d, bin_pl):
        """Find best (x,y,z) for box in bin, or None."""
        if not bin_pl:
            return (0, 0, 0) if (w <= W and h <= H and d <= D) else None

        cands = {(0, 0, 0)}
        for (ex, ey, ez, ew, eh, ed) in bin_pl:
            cands.add((ex + ew, ey, ez))
            cands.add((ex, ey + eh, ez))
            cands.add((ex, ey, ez + ed))
            cands.add((ex + ew, ey + eh, ez))
            cands.add((ex + ew, ey, ez + ed))
            cands.add((ex, ey + eh, ez + ed))
            cands.add((ex + ew, ey + eh, ez + ed))

        best = None
        best_s = None
        for (cx, cy, cz) in cands:
            if cx + w > W or cy + h > H or cz + d > D:
                continue
            ok = True
            for (ex, ey, ez, ew, eh, ed) in bin_pl:
                if cx < ex + ew and ex < cx + w and cy < ey + eh and ey < cy + h and cz < ez + ed and ez < cz + d:
                    ok = False
                    break
            if not ok:
                continue
            s = (cz, cy, cx)
            if best_s is None or s < best_s:
                best_s = s
                best = (cx, cy, cz)
        return best

    def bvol(bin_pl):
        return sum(ew * eh * ed for (_, _, _, ew, eh, ed) in bin_pl)

    for idx, (oi, w, h, d, vol) in enumerate(indexed_boxes):
        if time.time() > deadline:
            _shelf(indexed_boxes, idx, len(bins), result, W, H, D)
            break

        bb = -1
        bp = None
        bs = None
        for bi, bpl in enumerate(bins):
            p = find_pos(w, h, d, bpl)
            if p is not None:
                s = (-bvol(bpl), p[2], p[1], p[0])
                if bs is None or s < bs:
                    bs = s
                    bb = bi
                    bp = p

        if bb == -1:
            bb = len(bins)
            bins.append([])
            bp = (0, 0, 0)

        bins[bb].append((bp[0], bp[1], bp[2], w, h, d))
        result[oi] = (bb, bp[0], bp[1], bp[2], w, h, d)

    # Consolidation: try to reduce bin count
    if time.time() < deadline - 5.0 and len(bins) > 1:
        _consolidate(bins, result, W, H, D, deadline)

    return result


def _shelf(indexed_boxes, start, b, result, W, H, D):
    x = y = z = 0
    rh = ld = 0
    for (oi, rw, rh2, rd, rv) in indexed_boxes[start:]:
        if x + rw > W:
            x, y, rh = 0, y + rh, 0
        if y + rh2 > H:
            y, z, ld = 0, z + ld, 0
            x, rh = 0, 0
        if z + rd > D:
            b += 1
            x, y, z, rh, ld = 0, 0, 0, 0, 0
        result[oi] = (b, x, y, z, rw, rh2, rd)
        x, rh, ld = x + rw, max(rh, rh2), max(ld, rd)


def _consolidate(bins, result, W, H, D, deadline):
    def find_pos(w, h, d, bin_pl):
        if not bin_pl:
            return (0, 0, 0) if (w <= W and h <= H and d <= D) else None
        cands = {(0, 0, 0)}
        for (ex, ey, ez, ew, eh, ed) in bin_pl:
            cands.add((ex + ew, ey, ez))
            cands.add((ex, ey + eh, ez))
            cands.add((ex, ey, ez + ed))
            cands.add((ex + ew, ey + eh, ez))
            cands.add((ex + ew, ey, ez + ed))
            cands.add((ex, ey + eh, ez + ed))
            cands.add((ex + ew, ey + eh, ez + ed))
        best = None
        best_s = None
        for (cx, cy, cz) in cands:
            if cx + w > W or cy + h > H or cz + d > D:
                continue
            ok = True
            for (ex, ey, ez, ew, eh, ed) in bin_pl:
                if cx < ex + ew and ex < cx + w and cy < ey + eh and ey < cy + h and cz < ez + ed and ez < cz + d:
                    ok = False
                    break
            if not ok:
                continue
            s = (cz, cy, cx)
            if best_s is None or s < best_s:
                best_s = s
                best = (cx, cy, cz)
        return best

    # Try to empty smaller bins by moving boxes to larger ones
    for _attempt in range(3):
        if time.time() > deadline:
            return
        # Sort bins by volume desc
        order = sorted(range(len(bins)), key=lambda i: -sum(ew*eh*ed for (_,_,_,ew,eh,ed) in bins[i]))
        merged = False
        for src in order[1:]:
            if time.time() > deadline:
                return
            if not bins[src]:
                continue
            for (bx, by, bz, bw, bh, bd) in list(bins[src]):
                if time.time() > deadline:
                    return
                for dst in order:
                    if dst == src or not bins[dst]:
                        continue
                    p = find_pos(bw, bh, bd, bins[dst])
                    if p is not None:
                        bins[src].remove((bx, by, bz, bw, bh, bd))
                        bins[dst].append((p[0], p[1], p[2], bw, bh, bd))
                        for i in range(len(result)):
                            r = result[i]
                            if r is not None and r[0] == src and r[4] == bw and r[5] == bh and r[6] == bd and r[2] == by and r[3] == bz:
                                result[i] = (dst, p[0], p[1], p[2], bw, bh, bd)
                                break
                        break
            if not bins[src]:
                # Remove empty bin
                bins.pop(src)
                # Renumber all bins above src
                for i in range(len(result)):
                    r = result[i]
                    if r is not None and r[0] > src:
                        result[i] = (r[0] - 1,) + r[1:]
                # Rebuild order
                order = sorted(range(len(bins)), key=lambda i: -sum(ew*eh*ed for (_,_,_,ew,eh,ed) in bins[i]))
                merged = True
                break
        if not merged:
            break
# EVOLVE-BLOCK-END
