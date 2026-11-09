"""3D bin packing: skyline placement + bounded local-search polish."""
import time
import random

_DEADLINE = 150.0
_MARGIN = 10.0


def _now():
    return time.perf_counter()


def _skyline_place(sky, x, y, w, h, W, H):
    """Place a w x h rectangle at (x, y) on a 2D skyline (list of (x, height)).
    Returns True and updates sky, or False if it doesn't fit."""
    # Find the max height over [x, x+w)
    mx = 0
    for sx, sh in sky:
        if sx < x + w and x < sx + w:
            if sh > mx:
                mx = sh
    if y + h > H or y < 0:
        return False
    # Check that the box's bottom sits at or above the skyline in its span
    # i.e., the space [y, y+h) must be free. Since skyline height = top of used,
    # we need y >= mx? No: skyline stores the top of the highest used block per x.
    # A new block at (x,y) is valid if y >= mx (it sits on top of existing) OR
    # it fits in a gap. For simplicity (bottom-left skyline), we require y >= mx
    # is NOT right; we need the region [y,y+h) empty, which means the top of used
    # within span must be <= y. So require mx <= y.
    if mx > y:
        return False
    # Update skyline: for x in [x, x+w), new top = max(old, y+h)
    new_top = y + h
    new_sky = []
    i = 0
    n = len(sky)
    # Insert/merge
    for sx, sh in sky:
        if sx >= x + w:
            break
        if sx + w > x and sx < x + w:
            # overlap: take max
            merged_h = sh if sh > new_top else new_top
            if new_sky and new_sky[-1][1] == merged_h:
                new_sky[-1] = (new_sky[-1][0], merged_h)
            else:
                new_sky.append((sx, merged_h))
    # Add the segment of the box not covered by existing nodes
    # Determine the full interval [x, x+w) coverage
    # Easier: rebuild by sampling.
    # Rebuild properly:
    new_sky = []
    # The skyline is a list of (x_start, height) for intervals [x_start, next_x_start)
    # We'll create a list of (x, h) for the span and merge.
    # Represent existing as intervals.
    intervals = []
    for idx in range(n):
        sx, sh = sky[idx]
        ex = sky[idx + 1][0] if idx + 1 < n else W
        intervals.append((sx, ex, sh))
    # Build a map of x->height for the affected region, then add box
    # For each x in [x, x+w), height = max(existing, new_top) but only where box is.
    # Actually box sets height to max(existing, y+h) only if box is placed at y and
    # y >= existing top. Since we already checked mx <= y, y+h >= existing for all,
    # so new height = y+h over the whole span where box covers, but existing might
    # be taller in sub-parts? No, mx <= y means all existing <= y < y+h, so new = y+h.
    # So over [x, x+w), height becomes y+h.
    # Merge with existing skyline.
    result = []
    for sx, ex, sh in intervals:
        # intersection with [x, x+w)
        lo = max(sx, x)
        hi = min(ex, x + w)
        if lo < hi:
            if result and result[-1][0] == lo:
                pass
            result.append((lo, hi, y + h))
        else:
            if result and result[-1][0] == sx:
                pass
            result.append((sx, ex, sh))
    # Now merge adjacent same-height
    merged = []
    for sx, ex, sh in result:
        if merged and merged[-1][2] == sh and merged[-1][1] == sx:
            merged[-1] = (merged[-1][0], ex, sh)
        else:
            merged.append((sx, ex, sh))
    # Convert back to (x, height) list
    new_sky = [(sx, sh) for sx, ex, sh in merged]
    sky[:] = new_sky
    return True


def _skyline_best_pos(sky, w, h, W, H):
    """Find best (x, y) to place w x h on skyline, minimizing (y, x). Returns (x,y) or None."""
    # Candidate x positions: 0 and right edges of skyline nodes
    candidates = set([0])
    for sx, sh in sky:
        candidates.add(sx)
        candidates.add(sx + w if sx + w <= W else None)
    best = None
    for x in candidates:
        if x is None or x + w > W or x < 0:
            continue
        # max height in span
        mx = 0
        for sx, sh in sky:
            if sx < x + w and x < sx + w:
                if sh > mx:
                    mx = sh
        y = mx
        if y + h <= H:
            if best is None or (y, x) < (best[1], best[0]):
                best = (x, y)
    return best


def _empty_sky(W, H):
    return [(0, 0)]


def _place_all(boxes, W, H, D):
    """Greedy skyline packing. Returns placements list or None if fails."""
    placements = []
    bins = []  # each bin: list of (x,y,z,w,h,d)
    for idx, (w, h, d) in enumerate(boxes):
        placed = False
        for b in range(len(bins)):
            # Try to place in bin b: for each z-layer (skyline per z)
            # Represent bin as list of (z, sky) where sky is 2D skyline for that layer
            bin_data = bins[b]
            # Find best (z, x, y)
            best = None
            for (z, sky) in bin_data:
                if z + d > D:
                    continue
                pos = _skyline_best_pos(sky, w, h, W, H)
                if pos is not None:
                    x, y = pos
                    if best is None or (z, y, x) < (best[2], best[1], best[0]):
                        best = (x, y, z)
            if best is None and not bin_data:
                # empty bin
                best = (0, 0, 0)
            if best is None:
                # try new z in this bin
                # find max z used
                max_z = 0
                for (z, sky) in bin_data:
                    if z + d > max_z:
                        max_z = z + d
                if max_z + d <= D:
                    best = (0, 0, max_z)
            if best is not None:
                x, y, z = best
                # Place
                # Find or create the sky for this z
                found = False
                for i, (zz, sky) in enumerate(bin_data):
                    if zz == z:
                        _skyline_place(sky, x, y, w, h, W, H)
                        found = True
                        break
                if not found:
                    sky = _empty_sky(W, H)
                    _skyline_place(sky, x, y, w, h, W, H)
                    bin_data.append((z, sky))
                placements.append((b, x, y, z, w, h, d))
                placed = True
                break
        if not placed:
            # New bin
            b = len(bins)
            bins.append([(0, _empty_sky(W, H))])
            x, y, z = 0, 0, 0
            sky = bins[b][0][1]
            _skyline_place(sky, x, y, w, h, W, H)
            placements.append((b, x, y, z, w, h, d))
    return placements


def _score(boxes, placements, W, H, D):
    """Compute score: B + 0.1*(1 - last_bin_frac)."""
    if not placements:
        return 0.0
    bins = {}
    for (b, x, y, z, w, h, d) in placements:
        if b not in bins:
            bins[b] = 0
        bins[b] += w * h * d
    B = len(bins)
    last_bin = max(bins.keys())
    vol = W * H * D
    frac = bins[last_bin] / vol if vol > 0 else 0
    return B + 0.1 * (1 - frac)


def pack(boxes, container):
    W, H, D = container
    t0 = _now()
    deadline = t0 + _DEADLINE - _MARGIN

    # Sort boxes by volume descending for better initial packing
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    sorted_boxes = [boxes[i] for i in order]

    placements = _place_all(sorted_boxes, W, H, D)
    if placements is None:
        # Fallback: naive shelf
        placements = []
        b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            placements.append((b, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        # Map back to original order
        # This fallback is in original order, so just return
        return placements

    # Map placements back to original box order
    # placements[i] corresponds to sorted_boxes[i] = boxes[order[i]]
    # We need to produce one placement per original box index
    final = [None] * len(boxes)
    for i, (b, x, y, z, w, h, d) in enumerate(placements):
        orig_idx = order[i]
        final[orig_idx] = (b, x, y, z, w, h, d)

    best_score = _score(boxes, final, W, H, D)
    best_placements = final[:]

    # Local search: try swapping box orderings or re-placing
    # Simple improvement: try to re-pack with different sort orders
    if _now() < deadline:
        # Try a few alternative orderings
        rng = random.Random(42)
        tried = 0
        while _now() < deadline and tried < 20:
            tried += 1
            # Random perturbation: sort by different key
            key_choice = rng.randint(0, 3)
            if key_choice == 0:
                order2 = sorted(range(len(boxes)), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
            elif key_choice == 1:
                order2 = sorted(range(len(boxes)), key=lambda i: max(boxes[i][0], boxes[i][1], boxes[i][2]), reverse=True)
            elif key_choice == 2:
                order2 = sorted(range(len(boxes)), key=lambda i: boxes[i][0]*boxes[i][1], reverse=True)
            else:
                order2 = list(range(len(boxes)))
                rng.shuffle(order2)
            sb = [boxes[i] for i in order2]
            pl = _place_all(sb, W, H, D)
            if pl is None:
                continue
            fin = [None] * len(boxes)
            for i, (b, x, y, z, w, h, d) in enumerate(pl):
                fin[order2[i]] = (b, x, y, z, w, h, d)
            sc = _score(boxes, fin, W, H, D)
            if sc < best_score:
                best_score = sc
                best_placements = fin[:]

    return best_placements
