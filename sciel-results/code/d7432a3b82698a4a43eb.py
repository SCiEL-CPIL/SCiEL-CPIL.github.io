"""3D Container Loading – fast structural packing with multiple heuristics."""
import time
import itertools
import random
import numpy as np

def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 120s kill

    def check_deadline():
        return time.time() - start_time > DEADLINE

    def get_rotations(w, h, d):
        """Return unique axis rotations, sorted by (x,y,z) dim to prefer fitting."""
        seen = set()
        rots = []
        for p in itertools.permutations((w, h, d)):
            if p not in seen:
                seen.add(p)
                rots.append(p)
        return rots

    # Precompute rotations for all boxes
    box_rots = [get_rotations(w, h, d) for w, h, d in boxes]

    def make_grid():
        """Create a 3D occupancy grid if dimensions are small enough."""
        if W <= 250 and H <= 250 and D <= 250:
            return np.zeros((W, H, D), dtype=np.int8)
        return None

    def try_place_box(box_idx, placed, grid):
        """
        Try to place box box_idx. Returns (x, y, z, rw, rh, rd) or None.
        Uses candidate positions generated from placed boxes + boundaries.
        """
        w, h, d = boxes[box_idx]
        rots = box_rots[box_idx]

        # Generate candidate positions
        cands = set()
        cands.add((0, 0, 0))

        for (px, py, pz, pw, ph, pd) in placed:
            # All 8 combinations of face/edge/corner adjacent
            for dx in (0, pw):
                for dy in (0, ph):
                    for dz in (0, pd):
                        cands.add((px + dx, py + dy, pz + dz))

        # Also add positions aligned to container boundaries
        for (px, py, pz, pw, ph, pd) in placed:
            cands.add((0, py, pz))
            cands.add((px, 0, pz))
            cands.add((px, py, 0))

        # Filter valid positions
        valid_cands = []
        for (x, y, z) in cands:
            if 0 <= x < W and 0 <= y < H and 0 <= z < D:
                valid_cands.append((x, y, z))

        # Sort candidates by a heuristic score: prefer lower y (gravity), then x, then z
        def pos_score(x, y, z):
            return y * 10000 + x * 100 + z

        valid_cands.sort(key=pos_score)

        best = None
        best_score = float('inf')

        for (rw, rh, rd) in rots:
            if rw > W or rh > H or rd > D:
                continue
            for (x, y, z) in valid_cands:
                if x + rw > W or y + rh > H or z + rd > D:
                    continue

                # Check overlap
                if grid is not None:
                    if grid[x:x+rw, y:y+rh, z:z+rd].max() > 0:
                        continue
                else:
                    # Fallback: list-based overlap check
                    overlap = False
                    for (px, py, pz, pw, ph, pd) in placed:
                        if not (x + rw <= px or px + pw <= x or
                                y + rh <= py or py + ph <= y or
                                z + rd <= pz or pz + pd <= z):
                            overlap = True
                            break
                    if overlap:
                        continue

                # Score: prefer positions that minimize wasted space
                sc = pos_score(x, y, z)
                if sc < best_score:
                    best_score = sc
                    best = (x, y, z, rw, rh, rd)

        return best

    def pack_order(order):
        """Pack boxes in given order. Returns list of placements."""
        placed = []
        result = [None] * len(boxes)

        if W <= 250 and H <= 250 and D <= 250:
            grid = np.zeros((W, H, D), dtype=np.int8)
        else:
            grid = None

        for idx in order:
            if check_deadline():
                break
            pos = try_place_box(idx, placed, grid)
            if pos is not None:
                x, y, z, rw, rh, rd = pos
                placed.append((x, y, z, rw, rh, rd))
                if grid is not None:
                    grid[x:x+rw, y:y+rh, z:z+rd] = 1
                result[idx] = (0, x, y, z, rw, rh, rd)

        for i in range(len(boxes)):
            if result[i] is None:
                w, h, d = boxes[i]
                result[i] = (-1, 0, 0, 0, w, h, d)
        return result

    def filled_volume(result):
        return sum(w * h * d for (b, x, y, z, w, h, d) in result if b == 0)

    # Prepare indexed boxes
    indexed = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]

    # Sort strategies
    def sort_vol(lst):
        return sorted(lst, key=lambda t: (-t[4], -max(t[1], t[2], t[3])))

    def sort_max_dim(lst):
        return sorted(lst, key=lambda t: (-max(t[1], t[2], t[3]), -t[4]))

    def sort_surface(lst):
        return sorted(lst, key=lambda t: (-(2*(t[1]*t[2]+t[2]*t[3]+t[1]*t[3])), -t[4]))

    def sort_base_area(lst):
        return sorted(lst, key=lambda t: (-(t[1]*t[2]), -t[4]))

    def sort_min_dim(lst):
        return sorted(lst, key=lambda t: (-min(t[1], t[2], t[3]), -t[4]))

    def sort_aspect(lst):
        def aspect(t):
            dims = sorted([t[1], t[2], t[3]])
            return dims[2] / dims[0] if dims[0] > 0 else 0
        return sorted(lst, key=lambda t: (aspect(t), -t[4]))

    def sort_heuristic(lst):
        # Weighted combination: volume * 0.6 + max_dim * 0.2 + surface * 0.2
        def score(t):
            w, h, d = t[1], t[2], t[3]
            vol = w * h * d
            maxd = max(w, h, d)
            surf = 2 * (w*h + h*d + w*d)
            return -(vol * 0.6 + maxd * 0.2 + surf * 0.2)
        return sorted(lst, key=score)

    strategies = [sort_vol, sort_max_dim, sort_surface, sort_base_area, sort_min_dim, sort_aspect, sort_heuristic]

    best_result = None
    best_vol = 0

    # Deterministic passes
    for sf in strategies:
        if check_deadline():
            break
        order = [t[0] for t in sf(indexed)]
        res = pack_order(order)
        vol = filled_volume(res)
        if vol > best_vol:
            best_vol = vol
            best_result = res

    # Random restarts with perturbed ordering
    rng = random.Random(42)
    max_restarts = 500

    for _ in range(max_restarts):
        if check_deadline():
            break

        # Create a perturbed ordering
        shuffled = list(indexed)
        # Sort by volume first
        shuffled.sort(key=lambda t: -t[4])
        n = len(shuffled)

        # Apply random swaps with probability decreasing with index
        for i in range(n - 1):
            if rng.random() < 0.3:
                j = rng.randint(i, n - 1)
                shuffled[i], shuffled[j] = shuffled[j], shuffled[i]

        order_r = [t[0] for t in shuffled]
        res_r = pack_order(order_r)
        vol_r = filled_volume(res_r)
        if vol_r > best_vol:
            best_vol = vol_r
            best_result = res_r

    if best_result is None:
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    return best_result
