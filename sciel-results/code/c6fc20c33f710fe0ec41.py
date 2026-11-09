"""3D Bin Packer using columnar/skyline representation with multi-restart stochastic search.

Strategy: Each bin maintains a 2D grid of column heights. For a given (x,y) column,
the box can be placed at z = height[x][y]. This gives O(grid_size) placement checks
per box per bin instead of O(n_items) overlap checks, enabling many more restarts
and local-search moves within the time budget.
"""
import time
import random
import math

_DEADLINE = 140.0  # safety margin before 150s kill


def _skyline_place(box, heights, W, H, D):
    """Try to place box (w,h,d) on a 2D height grid (W x H).
    Returns (x, y, z) if placed, else None.
    heights[x][y] = current top z at column (x,y).
    """
    w, h, d = box
    if w > W or h > H or d > D:
        return None

    # Collect candidate (x, y) positions: x in [0, W-w], y in [0, H-h]
    # Use a priority: lowest z first, then lowest y, then lowest x
    best = None
    best_key = None

    # For efficiency, we check all possible (x,y) but early-exit if z+d > D
    for x in range(W - w + 1):
        for y in range(H - h + 1):
            # Find max height in the footprint
            max_z = 0
            for dx in range(w):
                for dy in range(h):
                    if heights[x + dx][y + dy] > max_z:
                        max_z = heights[x + dx][y + dy]
            z = max_z
            if z + d > D:
                continue
            key = (z, y, x)
            if best_key is None or key < best_key:
                best_key = key
                best = (x, y, z)

    if best is None:
        return None

    x, y, z = best
    # Commit: update heights
    for dx in range(w):
        for dy in range(h):
            heights[x + dx][y + dy] = z + d
    return (x, y, z)


def _skyline_remove(box, heights, x, y, z, W, H):
    """Remove a box from the skyline. This is approximate: set heights in footprint
    to min of neighbors. For exact removal we'd need a full 3D structure, but for
    local search moves we can rebuild the affected region."""
    w, h, d = box
    # Simple approach: for each column in footprint, recompute height from scratch
    # is expensive. Instead, we just set to z (the bottom of the removed box)
    # and let subsequent placements handle it. This is only used in local search
    # where we verify validity after the move.
    for dx in range(w):
        for dy in range(h):
            if heights[x + dx][y + dy] == z + d:
                heights[x + dx][y + dy] = z


def _pack_one_restart(boxes, container, order, randomize_ties=False):
    """Pack boxes in given order using skyline method. Returns (placements, bins_data)."""
    W, H, D = container
    n = len(boxes)
    bins_data = []  # list of (heights_2d, items_list)
    placements = [None] * n

    for idx in order:
        box = boxes[idx]
        w, h, d = box
        if w > W or h > H or d > D:
            # Can't fit anywhere, put in a new bin at origin (will be invalid but we handle)
            bi = len(bins_data)
            heights = [[0] * H for _ in range(W)]
            heights[x if (x:=0) else 0][0] = d  # placeholder
            # Actually just store it
            bins_data.append((heights, [(0, 0, 0, w, h, d)]))
            placements[idx] = (bi, 0, 0, 0, w, h, d)
            continue

        best_bin = None
        best_pos = None
        best_key = None

        for bi in range(len(bins_data)):
            heights, items = bins_data[bi]
            # Try to place
            # Save state for rollback
            saved = [row[:] for row in heights]
            pos = _skyline_place(box, heights, W, H, D)
            if pos is None:
                # Rollback
                for xi in range(W):
                    for yi in range(H):
                        heights[xi][yi] = saved[xi][yi]
                continue

            x, y, z = pos
            # Compute fill metric for tie-breaking
            vol = sum(it[3] * it[4] * it[5] for it in items)
            key = (-vol, z, y, x)
            if randomize_ties and random.random() < 0.3:
                # Perturb key slightly for diversity
                key = (-vol + random.uniform(-0.01, 0.01), z + random.uniform(-0.1, 0.1), y, x)

            if best_key is None or key < best_key:
                best_key = key
                best_bin = bi
                best_pos = (x, y, z)
                # Save the modified heights state
                best_heights = [row[:] for row in heights]

        if best_bin is not None:
            bi = best_bin
            x, y, z = best_pos
            # Commit
            bins_data[bi] = (best_heights, bins_data[bi][1] + [(x, y, z, w, h, d)])
            placements[idx] = (bi, x, y, z, w, h, d)
        else:
            # New bin
            bi = len(bins_data)
            heights = [[0] * H for _ in range(W)]
            heights[0][0] = d
            # Properly set all columns in footprint
            for dx in range(w):
                for dy in range(h):
                    heights[dx][dy] = d
            bins_data.append((heights, [(0, 0, 0, w, h, d)]))
            placements[idx] = (bi, 0, 0, 0, w, h, d)

    return placements, bins_data


def _compute_score(placements, container):
    W, H, D = container
    cvol = W * H * D
    if not placements:
        return 0.0
    bins = {}
    last = -1
    for bi, x, y, z, w, h, d in placements:
        bins.setdefault(bi, 0)
        bins[bi] += w * h * d
        if bi > last:
            last = bi
    B = len(bins)
    last_vol = bins.get(last, 0)
    return B + 0.1 * (1 - last_vol / cvol)


def _validate(placements, boxes, container):
    """Check no overlaps and all within bounds."""
    W, H, D = container
    bins = {}
    for i, (bi, x, y, z, w, h, d) in enumerate(placements):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        bins.setdefault(bi, []).append((x, y, z, w, h, d, i))

    for bi, items in bins.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a = items[i][:6]
                b = items[j][:6]
                if (a[0] + a[3] <= b[0] or b[0] + b[3] <= a[0] or
                    a[1] + a[4] <= b[1] or b[1] + b[4] <= a[1] or
                    a[2] + a[5] <= b[2] or b[2] + b[5] <= a[2]):
                    continue
                return False
    return True


def _local_search(placements, boxes, container, t0, max_iters=200):
    """Try to move boxes from last bin to earlier bins."""
    W, H, D = container
    n = len(boxes)
    best = [list(p) for p in placements]
    best_s = _compute_score(best, container)

    for it in range(max_iters):
        if time.time() - t0 > _DEADLINE - 5:
            break

        used_bins = sorted(set(b for b, *_ in best))
        if len(used_bins) <= 1:
            break
        last_bin = used_bins[-1]
        last_items = [i for i in range(n) if best[i][0] == last_bin]
        if not last_items:
            break

        random.shuffle(last_items)
        moved = False
        for i in last_items:
            if time.time() - t0 > _DEADLINE - 5:
                break
            bi, x, y, z, w, h, d = best[i]
            # Try moving to each earlier bin
            for tb in used_bins[:-1]:
                # Build skyline for target bin
                target_items = [(bx, by, bz, bw, bh, bd) for j in range(n)
                                if best[j][0] == tb
                                for bx, by, bz, bw, bh, bd in [tuple(best[j][1:])]]
                heights = [[0] * H for _ in range(W)]
                for (bx, by, bz, bw, bh, bd) in target_items:
                    for dx in range(bw):
                        for dy in range(bh):
                            if bx + dx < W and by + dy < H:
                                if bz + bd > heights[bx + dx][by + dy]:
                                    heights[bx + dx][by + dy] = bz + bd

                # Try to place box i in target bin
                saved = [row[:] for row in heights]
                pos = _skyline_place((w, h, d), heights, W, H, D)
                if pos is not None:
                    nx, ny, nz = pos
                    # Tentatively update
                    new_best = [row[:] for row in best]
                    new_best[i] = [tb, nx, ny, nz, w, h, d]
                    # Validate
                    if _validate(new_best, boxes, container):
                        s = _compute_score(new_best, container)
                        if s < best_s:
                            best = new_best
                            best_s = s
                            moved = True
                            break
                # Rollback heights
                for xi in range(W):
                    for yi in range(H):
                        heights[xi][yi] = saved[xi][yi]
            if moved:
                break
        if not moved:
            break

    return best, best_s


def pack(boxes, container):
    W, H, D = container
    t0 = time.time()
    n = len(boxes)

    # Generate diverse orderings
    def gen_orders():
        orders = []
        # Volume descending
        orders.append(sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True))
        # Max dim descending
        orders.append(sorted(range(n), key=lambda i: max(boxes[i]), reverse=True))
        # Surface area descending
        orders.append(sorted(range(n), key=lambda i: 2 * (boxes[i][0] * boxes[i][1] + boxes[i][1] * boxes[i][2] + boxes[i][0] * boxes[i][2]), reverse=True))
        # Random perturbations of volume-desc
        for _ in range(10):
            base = sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)
            # Swap a few adjacent pairs
            for _ in range(3):
                if len(base) > 1:
                    p = random.randint(0, len(base) - 2)
                    base[p], base[p + 1] = base[p + 1], base[p]
            orders.append(base)
        # Fully random
        for _ in range(5):
            r = list(range(n))
            random.shuffle(r)
            orders.append(r)
        return orders

    all_orders = gen_orders()
    random.shuffle(all_orders)

    best_placements = None
    best_score = float('inf')

    for order in all_orders:
        if time.time() - t0 > _DEADLINE - 10:
            break

        placements, bins_data = _pack_one_restart(boxes, container, order, randomize_ties=True)

        if not _validate(placements, boxes, container):
            continue

        s = _compute_score(placements, container)
        if s < best_score:
            best_score = s
            best_placements = [list(p) for p in placements]

    # Local search on best
    if best_placements is not None and time.time() - t0 < _DEADLINE - 5:
        improved, improved_s = _local_search(best_placements, boxes, container, t0, max_iters=100)
        if improved_s < best_score:
            best_score = improved_s
            best_placements = improved

    return [tuple(p) for p in best_placements]
