"""Combined 3D container loading: grid-accelerated collision + shelf positions + reinsertion."""
import time
import itertools
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    start_time = time.time()
    DEADLINE = 125.0

    def elapsed():
        return time.time() - start_time

    def check_deadline():
        return elapsed() > DEADLINE

    def rotations(w, h, d):
        seen = set()
        rots = []
        for perm in itertools.permutations((w, h, d)):
            if perm not in seen:
                seen.add(perm)
                rots.append(perm)
        return rots

    # Grid-based collision: use a boolean numpy array of size (W, H, D)
    # Only feasible if W*H*D is manageable. For safety, fall back to list-based if too large.
    # Typical containers in these benchmarks are small enough (e.g., 10-30 per dim).
    grid_size = W * H * D
    USE_GRID = grid_size <= 500000  # ~500k voxels max

    if USE_GRID:
        grid = np.zeros((W, H, D), dtype=np.uint8)

        def mark_grid(x, y, z, w, h, d):
            grid[x:x+w, y:y+h, z:z+d] = 1

        def unmark_grid(x, y, z, w, h, d):
            grid[x:x+w, y:y+h, z:z+d] = 0

        def grid_overlaps(x, y, z, w, h, d):
            if x + w > W or y + h > H or z + d > D:
                return True
            return np.any(grid[x:x+w, y:y+h, z:z+d])
    else:
        # Fallback: list of placed boxes
        def mark_grid(x, y, z, w, h, d):
            pass

        def unmark_grid(x, y, z, w, h, d):
            pass

        def grid_overlaps(x, y, z, w, h, d):
            if x + w > W or y + h > H or z + d > D:
                return True
            for px, py, pz, pw, ph, pd in placed_fallback:
                if not (x + w <= px or px + pw <= x or
                        y + h <= py or py + ph <= y or
                        z + d <= pz or pz + pd <= z):
                    return True
            return False

    # We'll maintain a list of placed boxes for candidate generation regardless
    # The grid is used only for fast collision checks

    def candidate_positions(placed, W, H, D):
        candidates = set()
        candidates.add((0, 0, 0))

        for px, py, pz, pw, ph, pd in placed:
            # Face-adjacent
            if px + pw < W:
                candidates.add((px + pw, py, pz))
            if py + ph < H:
                candidates.add((px, py + ph, pz))
            if pz + pd < D:
                candidates.add((px, py, pz + pd))

            # Edge-adjacent
            if px + pw < W and py + ph < H:
                candidates.add((px + pw, py + ph, pz))
            if px + pw < W and pz + pd < D:
                candidates.add((px + pw, py, pz + pd))
            if py + ph < H and pz + pd < D:
                candidates.add((px, py + ph, pz + pd))

            # Corner
            if px + pw < W and py + ph < H and pz + pd < D:
                candidates.add((px + pw, py + ph, pz + pd))

            # Shelf positions: align to edges
            candidates.add((0, py, pz))
            if px + pw < W:
                candidates.add((px + pw, py, pz))
            candidates.add((px, 0, pz))
            if py + ph < H:
                candidates.add((px, py + ph, pz))
            candidates.add((px, py, 0))
            if pz + pd < D:
                candidates.add((px, py, pz + pd))

        valid = []
        for (x, y, z) in candidates:
            if 0 <= x < W and 0 <= y < H and 0 <= z < D:
                valid.append((x, y, z))
        return valid

    def score_position(x, y, z, w, h, d, W, H, D):
        # Prefer low y, then compact x and z
        return y * 1.0 + x * 0.3 + z * 0.2

    def try_place_box(w, h, d, placed, W, H, D, deadline_check):
        rots = rotations(w, h, d)
        best = None
        best_score = float('inf')
        positions = candidate_positions(placed, W, H, D)

        for (rw, rh, rd) in rots:
            for (x, y, z) in positions:
                if x + rw > W or y + rh > H or z + rd > D:
                    continue
                if grid_overlaps(x, y, z, rw, rh, rd):
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

    def pack_with_order(order, boxes, W, H, D, deadline_check):
        # Reset grid if using it
        if USE_GRID:
            grid[:] = 0

        placed = []  # (x, y, z, w, h, d)
        result = [None] * len(boxes)

        for idx in order:
            if deadline_check():
                break
            w, h, d = boxes[idx]
            pos = try_place_box(w, h, d, placed, W, H, D, deadline_check)
            if pos is not None:
                x, y, z, rw, rh, rd = pos
                placed.append((x, y, z, rw, rh, rd))
                if USE_GRID:
                    mark_grid(x, y, z, rw, rh, rd)
                result[idx] = (0, x, y, z, rw, rh, rd)

        # Reinsertion pass: try to place remaining boxes into gaps
        if not check_deadline():
            unplaced = [i for i in range(len(boxes)) if result[i] is None]
            # Sort unplaced by volume descending for better gap filling
            unplaced.sort(key=lambda i: -(boxes[i][0] * boxes[i][1] * boxes[i][2]))
            for idx in unplaced:
                if check_deadline():
                    break
                w, h, d = boxes[idx]
                pos = try_place_box(w, h, d, placed, W, H, D, deadline_check)
                if pos is not None:
                    x, y, z, rw, rh, rd = pos
                    placed.append((x, y, z, rw, rh, rd))
                    if USE_GRID:
                        mark_grid(x, y, z, rw, rh, rd)
                    result[idx] = (0, x, y, z, rw, rh, rd)

        for i in range(len(boxes)):
            if result[i] is None:
                w, h, d = boxes[i]
                result[i] = (-1, 0, 0, 0, w, h, d)

        return result

    def volume_filled(placements):
        total = 0
        for (b, x, y, z, w, h, d) in placements:
            if b == 0:
                total += w * h * d
        return total

    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]

    # Multiple sort strategies
    def sort_by_volume(bl):
        return sorted(bl, key=lambda t: (-t[1]*t[2]*t[3], -max(t[1], t[2], t[3])))

    def sort_by_max_dim(bl):
        return sorted(bl, key=lambda t: (-max(t[1], t[2], t[3]), -t[1]*t[2]*t[3]))

    def sort_by_surface(bl):
        return sorted(bl, key=lambda t: (-2*(t[1]*t[2] + t[2]*t[3] + t[1]*t[3]), -t[1]*t[2]*t[3]))

    def sort_by_min_dim(bl):
        return sorted(bl, key=lambda t: (-min(t[1], t[2], t[3]), -t[1]*t[2]*t[3]))

    def sort_by_base_area(bl):
        return sorted(bl, key=lambda t: (-t[1]*t[2], -t[1]*t[2]*t[3]))

    sort_strategies = [
        sort_by_volume,
        sort_by_max_dim,
        sort_by_surface,
        sort_by_min_dim,
        sort_by_base_area,
    ]

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

    # Random restarts
    rng = random.Random(42)
    for restart in range(300):
        if check_deadline():
            break
        shuffled = list(indexed_boxes)
        rng.shuffle(shuffled)
        shuffled.sort(key=lambda t: -t[1]*t[2]*t[3])
        # Perturb
        for _ in range(len(shuffled) // 4):
            i = rng.randint(0, len(shuffled) - 2)
            shuffled[i], shuffled[i+1] = shuffled[i+1], shuffled[i]
        order_r = [t[0] for t in shuffled]
        result_r = pack_with_order(order_r, boxes, W, H, D, check_deadline)
        vol_r = volume_filled(result_r)
        if vol_r > best_vol:
            best_vol = vol_r
            best_result = result_r

    return best_result
# EVOLVE-BLOCK-END
