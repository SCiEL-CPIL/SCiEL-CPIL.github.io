"""3D Container Loading – grid-accelerated packing with Dijkstra-style position search, multi-strategy + final insertion pass."""
import time
import itertools
import random
import heapq

def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def check_deadline():
        return time.time() - start_time > DEADLINE

    def get_rotations(w, h, d):
        seen = set()
        rots = []
        for p in itertools.permutations((w, h, d)):
            if p not in seen:
                seen.add(p)
                rots.append(p)
        return rots

    # Grid-based occupancy for fast overlap checks
    MAX_GRID_DIM = 200

    def make_grid():
        import numpy as np
        if W <= MAX_GRID_DIM and H <= MAX_GRID_DIM and D <= MAX_GRID_DIM:
            return np.zeros((W, H, D), dtype=bool)
        return None

    def grid_check(grid, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        if grid[x:x+w, y:y+h, z:z+d].any():
            return False
        return True

    def grid_mark(grid, x, y, z, w, h, d):
        grid[x:x+w, y:y+h, z:z+d] = True

    def grid_unmark(grid, x, y, z, w, h, d):
        grid[x:x+w, y:y+h, z:z+d] = False

    def list_check(placed, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        for px, py, pz, pw, ph, pd in placed:
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True

    def generate_candidates(placed, W, H, D):
        """Generate candidate (x, y, z) positions based on placed boxes."""
        cands = set()
        cands.add((0, 0, 0))
        for (px, py, pz, pw, ph, pd) in placed:
            cands.add((px + pw, py, pz))
            cands.add((px, py + ph, pz))
            cands.add((px, py, pz + pd))
            cands.add((px + pw, py + ph, pz))
            cands.add((px + pw, py, pz + pd))
            cands.add((px, py + ph, pz + pd))
            cands.add((px + pw, py + ph, pz + pd))
            cands.add((0, py, pz))
            cands.add((px, 0, pz))
            cands.add((px, py, 0))
        valid = []
        for (x, y, z) in cands:
            if 0 <= x < W and 0 <= y < H and 0 <= z < D:
                valid.append((x, y, z))
        return valid

    def score_pos(x, y, z):
        """Score a position: prefer lower y, then smaller x, then smaller z."""
        return y * 1000 + x * 10 + z

    def try_place_dijkstra(w, h, d, placed, grid, W, H, D):
        """
        Use Dijkstra-style search over coordinate surfaces to find the best placement.
        Nodes are (x, y, z) positions. Cost is weighted: height * 100 + x * 10 + z.
        We explore from origin outward, preferring low positions.
        """
        rots = get_rotations(w, h, d)
        best = None
        best_sc = float('inf')

        # Generate candidate positions
        cands = generate_candidates(placed, W, H, D)
        # Sort by score to explore in order
        cands.sort(key=lambda c: score_pos(c[0], c[1], c[2]))

        for (rw, rh, rd) in rots:
            if rw > W or rh > H or rd > D:
                continue
            for (x, y, z) in cands:
                if x + rw > W or y + rh > H or z + rd > D:
                    continue
                if grid is not None:
                    if not grid_check(grid, x, y, z, rw, rh, rd):
                        continue
                else:
                    if not list_check(placed, x, y, z, rw, rh, rd):
                        continue
                sc = score_pos(x, y, z)
                if sc < best_sc:
                    best_sc = sc
                    best = (x, y, z, rw, rh, rd)
                break  # first valid for this rotation is the best (lowest)
            if best is not None:
                break

        return best

    def try_place(w, h, d, placed, grid, W, H, D):
        """Try to place a box; returns (x,y,z,rw,rh,rd) or None."""
        return try_place_dijkstra(w, h, d, placed, grid, W, H, D)

    def pack_order(order, boxes, W, H, D, grid):
        """Pack boxes in given order, return list of placements."""
        placed = []
        result = [None] * len(boxes)

        for idx in order:
            if check_deadline():
                break
            w, h, d = boxes[idx]
            pos = try_place(w, h, d, placed, grid, W, H, D)
            if pos is not None:
                x, y, z, rw, rh, rd = pos
                placed.append((x, y, z, rw, rh, rd))
                if grid is not None:
                    grid_mark(grid, x, y, z, rw, rh, rd)
                result[idx] = (0, x, y, z, rw, rh, rd)

        for i in range(len(boxes)):
            if result[i] is None:
                w, h, d = boxes[i]
                result[i] = (-1, 0, 0, 0, w, h, d)
        return result

    def final_insertion_pass(result, boxes, W, H, D):
        """
        Attempt to fit unplaced boxes into the best solution using dense candidate positions.
        This can recover volume from boxes left out during the main greedy pass.
        """
        if check_deadline():
            return result

        # Reconstruct placed boxes from result
        placed = []
        for (b, x, y, z, w, h, d) in result:
            if b == 0:
                placed.append((x, y, z, w, h, d))

        # Create grid
        grid = make_grid()
        if grid is not None:
            for (px, py, pz, pw, ph, pd) in placed:
                grid_mark(grid, px, py, pz, pw, ph, pd)

        # Find unplaced boxes
        unplaced = [i for i in range(len(boxes)) if result[i][0] == -1]
        if not unplaced:
            return result

        # Sort unplaced by volume descending
        unplaced.sort(key=lambda i: -boxes[i][0] * boxes[i][1] * boxes[i][2])

        new_result = list(result)
        for idx in unplaced:
            if check_deadline():
                break
            w, h, d = boxes[idx]
            pos = try_place(w, h, d, placed, grid, W, H, D)
            if pos is not None:
                x, y, z, rw, rh, rd = pos
                placed.append((x, y, z, rw, rh, rd))
                if grid is not None:
                    grid_mark(grid, x, y, z, rw, rh, rd)
                new_result[idx] = (0, x, y, z, rw, rh, rd)

        return new_result

    def filled_volume(result):
        return sum(w * h * d for (b, x, y, z, w, h, d) in result if b == 0)

    # Sort strategies
    indexed = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]

    def sort_vol(lst):
        return sorted(lst, key=lambda t: (-t[1]*t[2]*t[3], -max(t[1], t[2], t[3])))

    def sort_max(lst):
        return sorted(lst, key=lambda t: (-max(t[1], t[2], t[3]), -t[1]*t[2]*t[3]))

    def sort_surf(lst):
        return sorted(lst, key=lambda t: (-2*(t[1]*t[2]+t[2]*t[3]+t[1]*t[3]), -t[1]*t[2]*t[3]))

    def sort_min(lst):
        return sorted(lst, key=lambda t: (-min(t[1], t[2], t[3]), -t[1]*t[2]*t[3]))

    def sort_base(lst):
        return sorted(lst, key=lambda t: (-t[1]*t[2], -t[1]*t[2]*t[3]))

    def sort_aspect(lst):
        def aspect(t):
            dims = sorted([t[1], t[2], t[3]])
            return (dims[2]/dims[0] if dims[0] > 0 else 0)
        return sorted(lst, key=lambda t: (aspect(t), -t[1]*t[2]*t[3]))

    strategies = [sort_vol, sort_max, sort_surf, sort_min, sort_base, sort_aspect]

    best_result = None
    best_vol = 0

    # Deterministic passes
    for sf in strategies:
        if check_deadline():
            break
        order = [t[0] for t in sf(indexed)]
        grid = make_grid()
        res = pack_order(order, boxes, W, H, D, grid)
        res = final_insertion_pass(res, boxes, W, H, D)
        vol = filled_volume(res)
        if vol > best_vol:
            best_vol = vol
            best_result = res

    # Random restarts
    rng = random.Random(123)
    max_restarts = 200

    for _ in range(max_restarts):
        if check_deadline():
            break
        shuffled = list(indexed)
        shuffled.sort(key=lambda t: -t[1]*t[2]*t[3])
        n = len(shuffled)
        for _ in range(n // 2):
            i = rng.randint(0, n - 2)
            shuffled[i], shuffled[i+1] = shuffled[i+1], shuffled[i]
        order_r = [t[0] for t in shuffled]

        grid = make_grid()
        res_r = pack_order(order_r, boxes, W, H, D, grid)
        res_r = final_insertion_pass(res_r, boxes, W, H, D)
        vol_r = filled_volume(res_r)
        if vol_r > best_vol:
            best_vol = vol_r
            best_result = res_r

    if best_result is None:
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    return best_result
