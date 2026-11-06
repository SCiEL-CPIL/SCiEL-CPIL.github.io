# EVOLVE-BLOCK-START
import time

def pack(boxes, container):
    """Place boxes (unrotated, no overlap) into numbered bins.

    Returns one (bin_id, x, y, z, w, h, d) per box, in the original order.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    start = time.time()
    deadline = start + 115.0  # safety margin before the 120s kill

    # ---- Sort: volume descending, tie-break by max side (biggest boxes first) ----
    indexed = sorted(
        range(n),
        key=lambda i: (-(boxes[i][0] * boxes[i][1] * boxes[i][2]),
                       -max(boxes[i])),
    )
    dims = [boxes[i] for i in indexed]

    # ---- Fast overlap test (short-circuit on x) ----
    def can_place(placed, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in placed:
            if x < px + pw and x + w > px and \
               y < py + ph and y + h > py and \
               z < pz + pd and z + d > pz:
                return False
        return True

    # ---- Candidate position generation from existing box corners ----
    def candidates(placed, w, h, d):
        xs = {0}; ys = {0}; zs = {0}
        for (px, py, pz, pw, ph, pd) in placed:
            xs.add(px); xs.add(px + pw)
            ys.add(py); ys.add(py + ph)
            zs.add(pz); zs.add(pz + pd)
        for x in xs:
            if x + w > W:
                continue
            for y in ys:
                if y + h > H:
                    continue
                for z in zs:
                    if z + d <= D:
                        yield (x, y, z)

    # ---- Best-fit placement: minimise the resulting bin bounding box ----
    def best_fit(placed, w, h, d):
        best_pos = None
        best_key = None
        if not placed:
            if w <= W and h <= H and d <= D:
                return (0, 0, 0)
            return None
        mx = max(px + pw for (px, py, pz, pw, ph, pd) in placed)
        my = max(py + ph for (px, py, pz, pw, ph, pd) in placed)
        mz = max(pz + pd for (px, py, pz, pw, ph, pd) in placed)
        for (x, y, z) in candidates(placed, w, h, d):
            if can_place(placed, x, y, z, w, h, d):
                key = (max(mx, x + w), max(my, y + h), max(mz, z + d))
                if best_key is None or key < best_key:
                    best_key = key
                    best_pos = (x, y, z)
                    if (0, 0, 0) == (x, y, z):
                        break
        return best_pos

    # ---- Greedy Best-Fit-Decreasing baseline ----
    bins = []
    for (w, h, d) in dims:
        best_bin = -1
        best_key = None
        best_pos = None
        for bi, placed in enumerate(bins):
            pos = best_fit(placed, w, h, d)
            if pos is not None:
                x, y, z = pos
                if placed:
                    mx = max(px + pw for (px, py, pz, pw, ph, pd) in placed)
                    my = max(py + ph for (px, py, pz, pw, ph, pd) in placed)
                    mz = max(pz + pd for (px, py, pz, pw, ph, pd) in placed)
                else:
                    mx = my = mz = 0
                key = (max(mx, x + w), max(my, y + h), max(mz, z + d))
                if best_key is None or key < best_key:
                    best_key = key
                    best_bin = bi
                    best_pos = pos
        if best_bin >= 0:
            x, y, z = best_pos
            bins[best_bin].append((x, y, z, w, h, d))
        else:
            bins.append([(0, 0, 0, w, h, d)])

    # ---- Time-bounded DFS: try to fit all boxes into a smaller bin count ----
    def try_pack(k):
        """Return a list of k bin-placement lists, or None if infeasible/budget."""
        state = {"nodes": 0, "fail": False, "budget": 250000}
        placed = [[] for _ in range(k)]
        memo = set()

        def place(i):
            if i == n:
                return True
            if state["fail"]:
                return False
            w, h, d = dims[i]
            state["nodes"] += 1
            if state["nodes"] > state["budget"]:
                state["fail"] = True
                return False
            order = sorted(range(k), key=lambda j: len(placed[j]), reverse=True)
            for bi in order:
                b = placed[bi]
                best_pos = None
                best_key = None
                if not b:
                    if w <= W and h <= H and d <= D:
                        best_pos = (0, 0, 0)
                else:
                    mx = max(px + pw for (px, py, pz, pw, ph, pd) in b)
                    my = max(py + ph for (px, py, pz, pw, ph, pd) in b)
                    mz = max(pz + pd for (px, py, pz, pw, ph, pd) in b)
                    for (x, y, z) in candidates(b, w, h, d):
                        if can_place(b, x, y, z, w, h, d):
                            key = (max(mx, x + w), max(my, y + h), max(mz, z + d))
                            if best_key is None or key < best_key:
                                best_key = key
                                best_pos = (x, y, z)
                                if (0, 0, 0) == (x, y, z):
                                    break
                if best_pos is None:
                    continue
                x, y, z = best_pos
                b.append((x, y, z, w, h, d))
                if place(i + 1):
                    return True
                b.pop()
                if state["fail"]:
                    return False
            # dead end: memoise the signature of the current configuration
            sig = tuple(sorted(
                (px, py, pz, pw, ph, pd) for b in placed for (px, py, pz, pw, ph, pd) in b
            ))
            memo.add(sig)
            return False

        ok = place(0)
        if ok and not state["fail"]:
            return placed
        return None

    best_bins = bins
    best_count = len(bins)
    for k in range(best_count - 1, 0, -1):
        if time.time() > deadline - 30:
            break
        res = try_pack(k)
        if res is not None:
            best_bins = res
            best_count = k
            break  # smallest k that works is optimal for B

    # ---- Final-bin fill compaction: move boxes out of the last bin ----
    if time.time() < deadline - 5 and len(best_bins) > 1:
        last = len(best_bins) - 1
        moved = True
        while moved and time.time() < deadline - 5:
            moved = False
            for box in list(best_bins[last]):
                x, y, z, w, h, d = box
                for bi in range(last):
                    pos = best_fit(best_bins[bi], w, h, d)
                    if pos is not None:
                        best_bins[last].remove(box)
                        nx, ny, nz = pos
                        best_bins[bi].append((nx, ny, nz, w, h, d))
                        moved = True
                        break
                if moved:
                    break

    # ---- Rebuild result in original order ----
    result = [None] * n
    for bi, placed in enumerate(best_bins):
        for (x, y, z, w, h, d) in placed:
            orig = indexed.index(w, h, d) if False else None
        # (dims are unique by original index; map via a lookup built below)
    # Build a stable mapping: each placed box references its dims; recover original
    # index by matching (w, h, d) to the indexed list, consuming one occurrence.
    remaining = list(indexed)
    for bi, placed in enumerate(best_bins):
        for (x, y, z, w, h, d) in placed:
            # find an unused original index with these dims
            target = None
            for j in range(len(remaining)):
                if boxes[remaining[j]] == (w, h, d):
                    target = remaining.pop(j)
                    break
            if target is None:
                # fallback: first unused index (should not happen)
                target = remaining.pop(0) if remaining else 0
            result[target] = (bi, x, y, z, w, h, d)
    # safety: any None (shouldn't happen) -> place in a fresh bin
    max_bin = max((r[0] for r in result if r is not None), default=-1)
    for i in range(n):
        if result[i] is None:
            w, h, d = boxes[i]
            max_bin += 1
            result[i] = (max_bin, 0, 0, 0, w, h, d)
    return result
# EVOLVE-BLOCK-END
