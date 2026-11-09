import time
import random
import itertools
from collections import defaultdict

def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0 or not boxes:
        return [(-1, 0, 0, 0, w, h, d) for (w, h, d) in boxes]

    start_time = time.time()
    DEADLINE = start_time + 145.0  # safety margin before the 150s kill

    def check_deadline():
        return time.time() > DEADLINE

    n = len(boxes)
    container_volume = W * H * D

    # Pre-compute unique rotations for each box (cached)
    rot_cache = {}
    def get_rotations(w, h, d):
        key = (w, h, d)
        if key in rot_cache:
            return rot_cache[key]
        seen = set()
        rots = []
        for perm in itertools.permutations((w, h, d)):
            if perm not in seen:
                seen.add(perm)
                rots.append(perm)
        rot_cache[key] = rots
        return rots

    box_rotations = [get_rotations(*b) for b in boxes]

    # Spatial hash parameters
    avg_w = sum(b[0] for b in boxes) / n
    avg_h = sum(b[1] for b in boxes) / n
    avg_d = sum(b[2] for b in boxes) / n
    cell_size = max(1, int((avg_w + avg_h + avg_d) / 3))

    def get_cells(x, y, z, w, h, d):
        cells = set()
        x_min = int(x // cell_size)
        x_max = int((x + w - 1e-9) // cell_size)
        y_min = int(y // cell_size)
        y_max = int((y + h - 1e-9) // cell_size)
        z_min = int(z // cell_size)
        z_max = int((z + d - 1e-9) // cell_size)
        for cx in range(x_min, x_max + 1):
            for cy in range(y_min, y_max + 1):
                for cz in range(z_min, z_max + 1):
                    cells.add((cx, cy, cz))
        return cells

    def can_place(placed, spatial_hash, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        new_cells = get_cells(x, y, z, w, h, d)
        for cell in new_cells:
            if cell in spatial_hash:
                for idx in spatial_hash[cell]:
                    px, py, pz, pw, ph, pd = placed[idx]
                    if not (x + w <= px + 1e-9 or px + pw <= x + 1e-9 or
                            y + h <= py + 1e-9 or py + ph <= y + 1e-9 or
                            z + d <= pz + 1e-9 or pz + pd <= z + 1e-9):
                        return False
        return True

    def score_position(x, y, z, w, h, d):
        # Prefer lower (y), then more compact (x, z)
        return y * 1.0 + x * 0.3 + z * 0.2

    def solve_with_order(order):
        placed = []  # list of (x, y, z, w, h, d)
        spatial_hash = defaultdict(list)
        placements = [None] * n
        candidates = {(0, 0, 0)}

        for box_id in order:
            if check_deadline():
                break
            best_pos = None
            best_rot = None
            best_score = float('inf')

            for rot in box_rotations[box_id]:
                rw, rh, rd = rot
                # iterate over a snapshot to avoid mutation issues
                for (cx, cy, cz) in list(candidates):
                    if can_place(placed, spatial_hash, cx, cy, cz, rw, rh, rd):
                        sc = score_position(cx, cy, cz, rw, rh, rd)
                        if sc < best_score:
                            best_score = sc
                            best_pos = (cx, cy, cz)
                            best_rot = rot
                if check_deadline():
                    break

            if best_pos is not None:
                bx, by, bz = best_pos
                bw, bh, bd = best_rot
                placed_idx = len(placed)
                placed.append((bx, by, bz, bw, bh, bd))
                placements[box_id] = (0, bx, by, bz, bw, bh, bd)

                cells = get_cells(bx, by, bz, bw, bh, bd)
                for cell in cells:
                    spatial_hash[cell].append(placed_idx)

                # Richer candidate generation (inspiration): face, edge, corner
                new_corners = [
                    (bx + bw, by, bz),
                    (bx + bw, by + bh, bz),
                    (bx + bw, by, bz + bd),
                    (bx + bw, by + bh, bz + bd),
                    (bx, by + bh, bz),
                    (bx, by + bh, bz + bd),
                    (bx, by, bz + bd),
                ]
                for nc in new_corners:
                    if 0 <= nc[0] <= W and 0 <= nc[1] <= H and 0 <= nc[2] <= D:
                        candidates.add(nc)

        vol = 0
        for i in range(n):
            if placements[i] is None:
                w, h, d = boxes[i]
                placements[i] = (-1, 0, 0, 0, w, h, d)
            else:
                vol += placements[i][4] * placements[i][5] * placements[i][6]
        return placements, vol

    indices = list(range(n))

    # Sorting strategies
    def order_vol():
        return sorted(indices, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    def order_max():
        return sorted(indices, key=lambda i: max(boxes[i]), reverse=True)
    def order_surface():
        return sorted(indices, key=lambda i: 2*(boxes[i][0]*boxes[i][1]+boxes[i][1]*boxes[i][2]+boxes[i][0]*boxes[i][2]), reverse=True)
    def order_min():
        return sorted(indices, key=lambda i: min(boxes[i]), reverse=True)
    def order_base():
        return sorted(indices, key=lambda i: boxes[i][0]*boxes[i][1], reverse=True)

    strategies = [order_vol, order_max, order_surface, order_min, order_base]

    best_placements = None
    best_vol = 0

    for fn in strategies:
        if check_deadline():
            break
        p, v = solve_with_order(fn())
        if v > best_vol:
            best_vol = v
            best_placements = p

    # Random restarts
    rng = random.Random(12345)
    while not check_deadline():
        order = indices[:]
        rng.shuffle(order)
        p, v = solve_with_order(order)
        if v > best_vol:
            best_vol = v
            best_placements = p

    if best_placements is None:
        best_placements = [(-1, 0, 0, 0, w, h, d) for (w, h, d) in boxes]

    return best_placements
