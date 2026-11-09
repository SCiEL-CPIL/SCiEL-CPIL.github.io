"""Combined 3D container loading: spatial-grid acceleration + diverse restarts."""
import time
import itertools
import random
import math

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    start_time = time.time()
    DEADLINE = 125.0  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def check_deadline():
        return elapsed() > DEADLINE

    # Pre-compute rotations for each unique box dimension set (cached)
    _rot_cache = {}
    def rotations(w, h, d):
        key = (w, h, d)
        if key not in _rot_cache:
            seen = set()
            rots = []
            for perm in itertools.permutations((w, h, d)):
                if perm not in seen:
                    seen.add(perm)
                    rots.append(perm)
            _rot_cache[key] = rots
        return _rot_cache[key]

    # --- Spatial grid occupancy for fast overlap checks ---
    # We maintain a dict mapping (x, y, z) -> list of box indices occupying that cell.
    # For a box at (bx,by,bz) with dims (bw,bh,bd), we mark all integer cells
    # (bx..bx+bw-1, by..by+bh-1, bz..bz+bd-1) as occupied by that box index.
    # Overlap check: for candidate (x,y,z,w,h,d), check if any cell in its volume
    # is occupied by a box that is NOT the same box (but since we place one at a time,
    # we just check if any cell is occupied at all).
    #
    # To keep memory reasonable, we only mark cells if the total container volume is
    # not too large. If W*H*D > 5_000_000, fall back to O(n) list scan.

    use_grid = (W * H * D) <= 5_000_000

    def make_grid():
        return {}

    def grid_mark(grid, x, y, z, w, h, d):
        for ix in range(x, x + w):
            for iy in range(y, y + h):
                for iz in range(z, z + d):
                    grid[(ix, iy, iz)] = True

    def grid_unmark(grid, x, y, z, w, h, d):
        for ix in range(x, x + w):
            for iy in range(y, y + h):
                for iz in range(z, z + d):
                    grid.pop((ix, iy, iz), None)

    def grid_overlaps(grid, x, y, z, w, h, d):
        for ix in range(x, min(x + w, W)):
            for iy in range(y, min(y + h, H)):
                for iz in range(z, min(z + d, D)):
                    if (ix, iy, iz) in grid:
                        return True
        return False

    # Fallback O(n) overlap
    def overlaps(x, y, z, w, h, d, placed):
        for px, py, pz, pw, ph, pd in placed:
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return True
        return False

    # Candidate position generator: corner-based + face-aligned
    def candidate_positions(placed, W, H, D):
        candidates = set()
        candidates.add((0, 0, 0))
        for px, py, pz, pw, ph, pd in placed:
            # 6 face-adjacent
            if px + pw < W:
                candidates.add((px + pw, py, pz))
            if py + ph < H:
                candidates.add((px, py + ph, pz))
            if pz + pd < D:
                candidates.add((px, py, pz + pd))
            # 12 edge-adjacent (pairs of faces)
            if px + pw < W and py + ph < H:
                candidates.add((px + pw, py + ph, pz))
            if px + pw < W and pz + pd < D:
                candidates.add((px + pw, py, pz + pd))
            if py + ph < H and pz + pd < D:
                candidates.add((px, py + ph, pz + pd))
            # 8 corner-adjacent (triples)
            if px + pw < W and py + ph < H and pz + pd < D:
                candidates.add((px + pw, py + ph, pz + pd))
            # Align to container edges
            candidates.add((0, py, pz))
            candidates.add((px, 0, pz))
            candidates.add((px, py, 0))
            if px + pw < W:
                candidates.add((px + pw, 0, pz))
                candidates.add((px + pw, py, 0))
            if py + ph < H:
                candidates.add((0, py + ph, pz))
                candidates.add((px, py + ph, 0))
            if pz + pd < D:
                candidates.add((0, py, pz + pd))
                candidates.add((px, 0, pz + pd))
            if px + pw < W and py + ph < H:
                candidates.add((px + pw, py + ph, 0))
            if px + pw < W and pz + pd < D:
                candidates.add((px + pw, 0, pz + pd))
            if py + ph < H and pz + pd < D:
                candidates.add((0, py + ph, pz + pd))

        valid = []
        for (x, y, z) in candidates:
            if 0 <= x < W and 0 <= y < H and 0 <= z < D:
                valid.append((x, y, z))
        return valid

    # Score: prefer lower y, then x, then z; slight preference for filling gaps
    def score_position(x, y, z, w, h, d, W, H, D):
        return y * 1.0 + x * 0.3 + z * 0.2

    def try_place_box(w, h, d, placed, grid, W, H, D, deadline_check):
        rots = rotations(w, h, d)
        best = None
        best_score = float('inf')
        positions = candidate_positions(placed, W, H, D)

        for (rw, rh, rd) in rots:
            for (x, y, z) in positions:
                if x + rw > W or y + rh > H or z + rd > D:
                    continue
                if use_grid:
                    if grid_overlaps(grid, x, y, z, rw, rh, rd):
                        continue
                else:
                    if overlaps(x, y, z, rw, rh, rd, placed):
                        continue
                sc = score_position(x, y, z, rw, rh, rd, W, H, D)
                if sc < best_score:
                    best_score = sc
                    best = (x, y, z, rw, rh, rd)
                if deadline_check():
                    return best
            if deadline_check():
                return best
        return best

    # Sorting strategies
    def sort_by_volume(bl):
        return sorted(bl, key=lambda t: (-t[1]*t[2]*t[3], -max(t[1], t[2], t[3])))
    def sort_by_max_dim(bl):
        return sorted(bl, key=lambda t: (-max(t[1], t[2], t[3]), -t[1]*t[2]*t[3]))
    def sort_by_surface(bl):
        return sorted(bl, key=lambda t: (-2*(t[1]*t[2]+t[2]*t[3]+t[1]*t[3]), -t[1]*t[2]*t[3]))
    def sort_by_min_dim(bl):
        return sorted(bl, key=lambda t: (-min(t[1], t[2], t[3]), -t[1]*t[2]*t[3]))
    def sort_by_base_area(bl):
        return sorted(bl, key=lambda t: (-t[1]*t[2], -t[1]*t[2]*t[3]))
    def sort_by_height(bl):
        return sorted(bl, key=lambda t: (-t[2], -t[1]*t[2]*t[3]))

    def pack_with_order(order, boxes, W, H, D, deadline_check):
        placed = []
        grid = make_grid() if use_grid else None
        result = [None] * len(boxes)
        for idx in order:
            if deadline_check():
                break
            w, h, d = boxes[idx]
            pos = try_place_box(w, h, d, placed, grid, W, H, D, deadline_check)
            if pos is not None:
                x, y, z, rw, rh, rd = pos
                placed.append((x, y, z, rw, rh, rd))
                if use_grid:
                    grid_mark(grid, x, y, z, rw, rh, rd)
                result[idx] = (0, x, y, z, rw, rh, rd)
        for i in range(len(boxes)):
            if result[i] is None:
                w, h, d = boxes[i]
                result[i] = (-1, 0, 0, 0, w, h, d)
        return result

    def volume_filled(placements):
        return sum(w*h*d for (b, x, y, z, w, h, d) in placements if b == 0)

    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]

    sort_strategies = [sort_by_volume, sort_by_max_dim, sort_by_surface,
                       sort_by_min_dim, sort_by_base_area, sort_by_height]

    best_result = None
    best_vol = 0

    for sort_fn in sort_strategies:
        if check_deadline():
            break
        order = [t[0] for t in sort_fn(indexed_boxes)]
        result = pack_with_order(order, boxes, W, H, D, check_deadline)
        vol = volume_filled(result)
        if vol > best_vol:
            best_vol = vol
            best_result = result

    if best_result is None:
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    if check_deadline():
        return best_result

    # Random restarts: diverse orderings
    rng = random.Random(42)
    # Budget: use remaining time for restarts
    remaining = DEADLINE - elapsed()
    # Estimate time per restart ~ 0.5s for 60 boxes with grid; allow up to ~200 restarts
    # but cap by remaining time
    max_restarts = min(200, int(remaining / 0.3))
    max_restarts = max(max_restarts, 20)

    for restart in range(max_restarts):
        if check_deadline():
            break
        # Diverse random ordering strategies
        strategy = restart % 5
        shuffled = list(indexed_boxes)
        if strategy == 0:
            # Pure random
            rng.shuffle(shuffled)
        elif strategy == 1:
            # Volume-sorted with adjacent swaps
            shuffled.sort(key=lambda t: -t[1]*t[2]*t[3])
            for _ in range(len(shuffled) // 3):
                i = rng.randint(0, len(shuffled) - 2)
                shuffled[i], shuffled[i+1] = shuffled[i+1], shuffled[i]
        elif strategy == 2:
            # Max-dim sorted with random tail
            shuffled.sort(key=lambda t: -max(t[1], t[2], t[3]))
            k = rng.randint(0, len(shuffled)//2)
            tail = shuffled[k:]
            rng.shuffle(tail)
            shuffled = shuffled[:k] + tail
        elif strategy == 3:
            # Interleave two sorted orders
            s1 = sorted(indexed_boxes, key=lambda t: -t[1]*t[2]*t[3])
            s2 = sorted(indexed_boxes, key=lambda t: -max(t[1], t[2], t[3]))
            shuffled = []
            i, j = 0, 0
            while i < len(s1) and j < len(s2):
                if rng.random() < 0.5:
                    shuffled.append(s1[i]); i += 1
                else:
                    shuffled.append(s2[j]); j += 1
            while i < len(s1):
                shuffled.append(s1[i]); i += 1
            while j < len(s2):
                shuffled.append(s2[j]); j += 1
        else:
            # Random with large boxes biased to front
            shuffled.sort(key=lambda t: -t[1]*t[2]*t[3])
            rng.shuffle(shuffled)
            # Move top-volume boxes to front with probability
            top = [t for t in indexed_boxes if t[1]*t[2]*t[3] > best_vol * 0.1]
            rng.shuffle(top)
            shuffled = top[:rng.randint(0, len(top))] + shuffled

        order_r = [t[0] for t in shuffled]
        result_r = pack_with_order(order_r, boxes, W, H, D, check_deadline)
        vol_r = volume_filled(result_r)
        if vol_r > best_vol:
            best_vol = vol_r
            best_result = result_r

    return best_result
# EVOLVE-BLOCK-END
