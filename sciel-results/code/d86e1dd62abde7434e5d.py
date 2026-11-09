import time
import itertools
import random
import sys

def load(boxes, container):
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    W, H, D = container
    n = len(boxes)

    # Precompute unique orientations for each box
    box_orients = []
    for w, h, d in boxes:
        perms = set(itertools.permutations((w, h, d)))
        box_orients.append(list(perms))

    def overlaps(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        if x1 + w1 <= x2 or x2 + w2 <= x1:
            return False
        if y1 + h1 <= y2 or y2 + h2 <= y1:
            return False
        if z1 + d1 <= z2 or z2 + d2 <= z1:
            return False
        return True

    def build_grid(placed, cell_size=1):
        """Build a spatial grid of placed boxes for fast overlap queries."""
        if not placed:
            return {}
        grid = {}
        for (idx, x, y, z, w, h, d) in placed:
            # Store in cells that the box occupies
            x0, x1 = int(x // cell_size), int((x + w - 1) // cell_size)
            y0, y1 = int(y // cell_size), int((y + h - 1) // cell_size)
            z0, z1 = int(z // cell_size), int((z + d - 1) // cell_size)
            for cx in range(x0, x1 + 1):
                for cy in range(y0, y1 + 1):
                    for cz in range(z0, z1 + 1):
                        key = (cx, cy, cz)
                        if key not in grid:
                            grid[key] = set()
                        grid[key].add(idx)
        return grid

    def can_place_grid(x, y, z, w, h, d, placed, grid, cell_size=1):
        """Check if a box can be placed at (x,y,z) with dims (w,h,d) using grid."""
        if x + w > W or y + h > H or z + d > D:
            return False
        if not placed:
            return True
        # Get candidate indices from grid cells
        x0, x1 = int(x // cell_size), int((x + w - 1) // cell_size)
        y0, y1 = int(y // cell_size), int((y + h - 1) // cell_size)
        z0, z1 = int(z // cell_size), int((z + d - 1) // cell_size)
        candidates = set()
        for cx in range(x0, x1 + 1):
            for cy in range(y0, y1 + 1):
                for cz in range(z0, z1 + 1):
                    key = (cx, cy, cz)
                    if key in grid:
                        candidates.update(grid[key])
        # Check overlap with candidates
        placed_dict = {p[0]: p for p in placed}
        for idx in candidates:
            _, px, py, pz, pw, ph, pd = placed_dict[idx]
            if overlaps(x, y, z, w, h, d, px, py, pz, pw, ph, pd):
                return False
        return True

    def can_place_brute(x, y, z, w, h, d, placed):
        """Brute force overlap check."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for _, px, py, pz, pw, ph, pd in placed:
            if overlaps(x, y, z, w, h, d, px, py, pz, pw, ph, pd):
                return False
        return True

    def greedy_pack(order, use_grid=True, cell_size=1):
        """Pack boxes in given order. Returns (result_list, total_volume)."""
        placed = []
        corners = [(0, 0, 0)]
        total_vol = 0
        grid = {}

        for idx in order:
            if time.time() - start_time > DEADLINE:
                break
            w, h, d = boxes[idx]
            orientations = box_orients[idx]

            best_placement = None
            best_score = (float('inf'), float('inf'), float('inf'))  # (z, y, x)

            for ow, oh, od in orientations:
                for cx, cy, cz in corners:
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue

                    if use_grid:
                        valid = can_place_grid(cx, cy, cz, ow, oh, od, placed, grid, cell_size)
                    else:
                        valid = can_place_brute(cx, cy, cz, ow, oh, od, placed)

                    if not valid:
                        continue

                    score = (cz, cy, cx)
                    if score < best_score:
                        best_score = score
                        best_placement = (idx, cx, cy, cz, ow, oh, od)

            if best_placement:
                placed.append(best_placement)
                total_vol += best_placement[4] * best_placement[5] * best_placement[6]

                # Update grid
                _, bx, by, bz, bw, bh, bd = best_placement
                x0, x1 = int(bx // cell_size), int((bx + bw - 1) // cell_size)
                y0, y1 = int(by // cell_size), int((by + bh - 1) // cell_size)
                z0, z1 = int(bz // cell_size), int((bz + bd - 1) // cell_size)
                for gx in range(x0, x1 + 1):
                    for gy in range(y0, y1 + 1):
                        for gz in range(z0, z1 + 1):
                            key = (gx, gy, gz)
                            if key not in grid:
                                grid[key] = set()
                            grid[key].add(idx)

                # Generate new corners
                _, bx, by, bz, bw, bh, bd = best_placement
                new_corners = [
                    (bx + bw, by, bz),
                    (bx, by + bh, bz),
                    (bx, by, bz + bd),
                    (bx + bw, by + bh, bz),
                    (bx + bw, by, bz + bd),
                    (bx, by + bh, bz + bd),
                ]
                for nc in new_corners:
                    if nc not in corners:
                        corners.append(nc)

                # Prune dominated corners (keep only Pareto-optimal)
                if len(corners) > 200:
                    corners = prune_corners(corners, W, H, D, placed)

        result = [(-1, 0, 0, 0, 0, 0, 0)] * n
        for idx, x, y, z, w, h, d in placed:
            result[idx] = (0, x, y, z, w, h, d)
        return result, total_vol, placed

    def prune_corners(corners, W, H, D, placed):
        """Keep only corners that are not strictly dominated by another corner."""
        # A corner (x1,y1,z1) is dominated by (x2,y2,z2) if x2<=x1, y2<=y1, z2<=z1 and at least one strict
        # For efficiency, just keep a subset
        if len(corners) <= 200:
            return corners
        # Sort by z, then y, then x
        sorted_corners = sorted(corners, key=lambda c: (c[2], c[1], c[0]))
        kept = []
        for c in sorted_corners:
            # Check if dominated by any kept corner
            dominated = False
            for kc in kept:
                if kc[0] <= c[0] and kc[1] <= c[1] and kc[2] <= c[2] and (kc[0] < c[0] or kc[1] < c[1] or kc[2] < c[2]):
                    dominated = True
                    break
            if not dominated:
                kept.append(c)
            if len(kept) >= 200:
                break
        return kept

    def reinsertion_pass(result, placed, total_vol):
        """Try to reinsert left-out boxes into the current placement."""
        placed_dict = {p[0]: p for p in placed}
        placed_set = set(placed_dict.keys())
        new_placed = list(placed)
        new_vol = total_vol
        corners = [(0, 0, 0)]
        for p in new_placed:
            _, bx, by, bz, bw, bh, bd = p
            for nc in [(bx+bw, by, bz), (bx, by+bh, bz), (bx, by, bz+bd),
                       (bx+bw, by+bh, bz), (bx+bw, by, bz+bd), (bx, by+bh, bz+bd)]:
                if nc not in corners:
                    corners.append(corners)  # placeholder

        # Rebuild corners properly
        corners = [(0, 0, 0)]
        for p in new_placed:
            _, bx, by, bz, bw, bh, bd = p
            for nc in [(bx+bw, by, bz), (bx, by+bh, bz), (bx, by, bz+bd),
                       (bx+bw, by+bh, bz), (bx+bw, by, bz+bd), (bx, by+bh, bz+bd)]:
                if nc not in corners:
                    corners.append(nc)

        # Try to place each left-out box
        left_out = [i for i in range(n) if i not in placed_set]
        # Sort left-out by volume descending
        left_out.sort(key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)

        for idx in left_out:
            if time.time() - start_time > DEADLINE:
                break
            w, h, d = boxes[idx]
            orientations = box_orients[idx]
            best_placement = None
            best_score = (float('inf'), float('inf'), float('inf'))

            for ow, oh, od in orientations:
                for cx, cy, cz in corners:
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue
                    valid = can_place_brute(cx, cy, cz, ow, oh, od, new_placed)
                    if not valid:
                        continue
                    score = (cz, cy, cx)
                    if score < best_score:
                        best_score = score
                        best_placement = (idx, cx, cy, cz, ow, oh, od)

            if best_placement:
                new_placed.append(best_placement)
                new_vol += best_placement[4] * best_placement[5] * best_placement[6]
                _, bx, by, bz, bw, bh, bd = best_placement
                for nc in [(bx+bw, by, bz), (bx, by+bh, bz), (bx, by, bz+bd),
                           (bx+bw, by+bh, bz), (bx+bw, by, bz+bd), (bx, by+bh, bz+bd)]:
                    if nc not in corners:
                        corners.append(nc)

        # Build result
        new_result = [(-1, 0, 0, 0, 0, 0, 0)] * n
        for idx, x, y, z, w, h, d in new_placed:
            new_result[idx] = (0, x, y, z, w, h, d)
        return new_result, new_vol

    # --- Main search ---
    best_result = None
    best_vol = 0

    # Strategy 1: Sorted by various keys
    strategies = [
        ("volume", lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2]),
        ("max_dim", lambda i: max(boxes[i])),
        ("sum_dim", lambda i: sum(boxes[i])),
        ("surface", lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2])),
        ("vol_per_max", lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2] / max(boxes[i]) if max(boxes[i]) > 0 else 0),
        ("ratio", lambda i: boxes[i][0]*boxes[i][1]/boxes[i][2] if boxes[i][2] > 0 else boxes[i][0]*boxes[i][1]),
    ]

    for name, key in strategies:
        if time.time() - start_time > DEADLINE:
            break
        order = sorted(range(n), key=key, reverse=True)
        result, vol, placed = greedy_pack(order, use_grid=True, cell_size=1)
        if vol > best_vol:
            best_vol = vol
            best_result = result
            # Try reinsertion
            if time.time() - start_time < DEADLINE - 5:
                r2, v2 = reinsertion_pass(result, placed, vol)
                if v2 > best_vol:
                    best_vol = v2
                    best_result = r2

    # Strategy 2: Random restarts with shuffled orders
    rng = random.Random(42)
    num_random_trials = 0
    while time.time() - start_time < DEADLINE - 10:
        num_random_trials += 1
        # Mix of sorted and random
        if num_random_trials % 3 == 0:
            order = list(range(n))
            rng.shuffle(order)
        elif num_random_trials % 3 == 1:
            order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
        else:
            # Partially random: large boxes first, then random
            vol_sorted = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
            order = vol_sorted[:]
            rng.shuffle(order[len(order)//3:])  # Shuffle the smaller ones

        result, vol, placed = greedy_pack(order, use_grid=True, cell_size=1)
        if vol > best_vol:
            best_vol = vol
            best_result = result
            if time.time() - start_time < DEADLINE - 5:
                r2, v2 = reinsertion_pass(result, placed, vol)
                if v2 > best_vol:
                    best_vol = v2
                    best_result = r2

    if best_result is None:
        best_result = [(-1, 0, 0, 0, 0, 0, 0)] * n

    return best_result
