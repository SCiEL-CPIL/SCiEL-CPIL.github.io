"""3D Container Loading: Structural change to a fast, robust packer.

Key structural changes vs. the previous search-heavy design:
1) A spatial index (uniform grid) for O(1)-ish overlap checks instead of O(n) scans.
2) A "skyline" candidate-position generator that keeps the candidate set small and
   physically meaningful (floor + top-of-existing-box faces), rather than an
   exploding cartesian product of all extreme coordinates.
3) A deterministic multi-start greedy core with a small number of well-chosen
   orderings, followed by a lightweight local improvement pass that only runs if
   there is ample time remaining. This avoids the previous design's tendency to
   burn the entire budget on many slow greedy passes and then time out.
4) The overlap test uses early-exit axis separation (the fast form) and the grid
   cell lookup to avoid scanning all placed boxes.
"""
import time
import random
import itertools
from collections import defaultdict


def _rotations(w, h, d):
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result


class _SpatialGrid:
    """Uniform-grid spatial index for axis-aligned boxes in [0,W)x[0,H)x[0,D)."""

    def __init__(self, W, H, D, cell=1):
        self.W = W
        self.H = H
        self.D = D
        self.cell = max(1, int(cell))
        self.buckets = defaultdict(list)
        self.boxes = []

    def _key(self, x, y, z):
        return (int(x // self.cell), int(y // self.cell), int(z // self.cell))

    def add(self, box):
        x, y, z, w, h, d = box
        self.boxes.append(box)
        x0, y0, z0 = self._key(x, y, z)
        x1, y1, z1 = self._key(x + w, y + h, z + d)
        for cx in range(x0, x1 + 1):
            for cy in range(y0, y1 + 1):
                for cz in range(z0, z1 + 1):
                    self.buckets[(cx, cy, cz)].append(box)

    def remove(self, box):
        # Remove the first occurrence of the exact box tuple.
        try:
            self.boxes.remove(box)
        except ValueError:
            return
        x, y, z, w, h, d = box
        x0, y0, z0 = self._key(x, y, z)
        x1, y1, z1 = self._key(x + w, y + h, z + d)
        for cx in range(x0, x1 + 1):
            for cy in range(y0, y1 + 1):
                for cz in range(z0, z1 + 1):
                    key = (cx, cy, cz)
                    lst = self.buckets.get(key)
                    if lst:
                        try:
                            lst.remove(box)
                        except ValueError:
                            pass

    def overlaps(self, x, y, z, w, h, d):
        x0, y0, z0 = self._key(x, y, z)
        x1, y1, z1 = self._key(x + w, y + h, z + d)
        for cx in range(x0, x1 + 1):
            for cy in range(y0, y1 + 1):
                for cz in range(z0, z1 + 1):
                    for (px, py, pz, pw, ph, pd) in self.buckets.get((cx, cy, cz), ()):
                        if x >= px + pw or x + w <= px:
                            continue
                        if y >= py + ph or y + h <= py:
                            continue
                        if z >= pz + pd or z + d <= pz:
                            continue
                        return True
        return False


def _fits_fast(grid, x, y, z, w, h, d, W, H, D):
    if x < 0 or y < 0 or z < 0:
        return False
    if x + w > W or y + h > H or z + d > D:
        return False
    return not grid.overlaps(x, y, z, w, h, d)


def _skyline_positions(grid, W, H, D, max_candidates=4000):
    """Generate candidate positions from the 'skyline' of placed boxes.

    The idea: a good position for a new box is either on the floor (z=0) or on
    top of an existing box (z = pz+pd), with x and y similarly snapped to 0 or
    to the right/far face of an existing box. We keep the set small by capping
    the number of distinct x, y, z values we consider.
    """
    if not grid.boxes:
        return [(0, 0, 0)]

    xs = {0}
    ys = {0}
    zs = {0}
    for (px, py, pz, pw, ph, pd) in grid.boxes:
        if px + pw < W:
            xs.add(px + pw)
        if py + ph < H:
            ys.add(py + ph)
        if pz + pd < D:
            zs.add(pz + pd)

    # Limit the number of distinct coordinates to keep the product small.
    xs = sorted(xs)
    ys = sorted(ys)
    zs = sorted(zs)

    # If the product is too large, prune by keeping the first few in each axis
    # (lowest coordinates are preferred by the compact heuristic).
    if len(xs) * len(ys) * len(zs) > max_candidates:
        # Keep enough per axis to stay under the cap.
        target_per_axis = max(1, int(round(max_candidates ** (1.0 / 3.0))))
        xs = xs[:target_per_axis]
        ys = ys[:target_per_axis]
        zs = zs[:target_per_axis]

    points = []
    for x in xs:
        for y in ys:
            for z in zs:
                if x < W and y < H and z < D:
                    points.append((x, y, z))
    return points


def _greedy_place(boxes, container, order, deadline, strategy='compact'):
    """Greedy placement using the spatial grid and skyline positions."""
    W, H, D = container
    n = len(boxes)
    # Choose a cell size roughly matching the median box dimension for efficiency.
    dims = [min(w, h, d) for (w, h, d) in boxes]
    cell = max(1, int(sorted(dims)[len(dims) // 2]) if dims else 1)
    grid = _SpatialGrid(W, H, D, cell=cell)
    placements = [None] * n
    total_vol = 0.0

    for idx in order:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        rots = _rotations(w, h, d)

        candidates = _skyline_positions(grid, W, H, D, max_candidates=4000)

        best_pos = None
        best_rot = None
        best_key = None

        for pos in candidates:
            x, y, z = pos
            for rw, rh, rd in rots:
                if _fits_fast(grid, x, y, z, rw, rh, rd, W, H, D):
                    if strategy == 'compact':
                        key = (z, y, x)
                    elif strategy == 'back':
                        key = (-x, -y, -z)
                    else:  # corner
                        key = (-(x + y + z),)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_pos = (x, y, z)
                        best_rot = (rw, rh, rd)

        if best_pos is not None:
            x, y, z = best_pos
            rw, rh, rd = best_rot
            grid.add((x, y, z, rw, rh, rd))
            placements[idx] = (0, x, y, z, rw, rh, rd)
            total_vol += rw * rh * rd
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)

    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)

    return placements, total_vol


def _local_improve(placements, boxes, container, deadline, rng, max_iters=40):
    """Lightweight local improvement: repeatedly try to move a loaded box to a
    lower position to free space for others. Only runs if there is time left.
    """
    W, H, D = container
    n = len(boxes)
    dims = [min(w, h, d) for (w, h, d) in boxes]
    cell = max(1, int(sorted(dims)[len(dims) // 2]) if dims else 1)
    grid = _SpatialGrid(W, H, D, cell=cell)

    loaded_indices = []
    for i in range(n):
        p = placements[i]
        if p[0] == 0:
            grid.add((p[1], p[2], p[3], p[4], p[5], p[6]))
            loaded_indices.append(i)

    if not loaded_indices:
        return placements, 0.0

    total_vol = sum(boxes[i][0] * boxes[i][1] * boxes[i][2] for i in loaded_indices)

    improved = True
    iters = 0
    while improved and iters < max_iters and time.time() < deadline:
        improved = False
        iters += 1

        for idx in loaded_indices:
            if time.time() > deadline:
                break

            p = placements[idx]
            box_pos = (p[1], p[2], p[3], p[4], p[5], p[6])
            grid.remove(box_pos)

            w, h, d = boxes[idx]
            rots = _rotations(w, h, d)
            candidates = _skyline_positions(grid, W, H, D, max_candidates=3000)

            best_pos = None
            best_rot = None
            best_key = None

            for pos in candidates:
                x, y, z = pos
                for rw, rh, rd in rots:
                    if _fits_fast(grid, x, y, z, rw, rh, rd, W, H, D):
                        key = (z, y, x)
                        if best_key is None or key < best_key:
                            best_key = key
                            best_pos = (x, y, z)
                            best_rot = (rw, rh, rd)

            if best_pos is not None:
                x, y, z = best_pos
                rw, rh, rd = best_rot
                new_pos = (x, y, z, rw, rh, rd)
                old_score = (box_pos[2], box_pos[1], box_pos[0])
                new_score = (z, y, x)
                if new_score < old_score:
                    grid.add(new_pos)
                    placements[idx] = (0, x, y, z, rw, rh, rd)
                    improved = True
                else:
                    grid.add(box_pos)
            else:
                grid.add(box_pos)

    total_vol = 0.0
    for i in range(n):
        if placements[i][0] == 0:
            total_vol += placements[i][4] * placements[i][5] * placements[i][6]

    return placements, total_vol


def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    W, H, D = container
    n = len(boxes)

    vols = [w * h * d for w, h, d in boxes]
    mins = [min(w, h, d) for w, h, d in boxes]
    maxs = [max(w, h, d) for w, h, d in boxes]
    ratios = [max(w, h, d) / max(min(w, h, d), 1e-9) for w, h, d in boxes]

    best_placements = None
    best_vol = 0.0

    # A small, well-chosen set of orderings and strategies.
    sort_strategies = [
        ('vol_desc', lambda i: vols[i], True),
        ('vol_asc', lambda i: vols[i], False),
        ('min_dim_desc', lambda i: mins[i], True),
        ('max_dim_desc', lambda i: maxs[i], True),
        ('ratio_desc', lambda i: ratios[i], True),
    ]

    strategies = ['compact', 'back']

    for sort_name, key_fn, reverse in sort_strategies:
        if time.time() > deadline:
            break
        order = list(range(n))
        order.sort(key=key_fn, reverse=reverse)

        for strat in strategies:
            if time.time() > deadline:
                break
            placements, vol = _greedy_place(boxes, container, order, deadline, strat)
            if vol > best_vol:
                best_vol = vol
                best_placements = [p[:] for p in placements]

    # If we have time, do a local improvement pass.
    if best_placements is not None and time.time() < deadline - 10.0:
        rng = random.Random(42)
        best_placements, best_vol = _local_improve(
            best_placements, boxes, container, deadline, rng, max_iters=30
        )

    # A few random restarts if time permits.
    remaining = deadline - time.time()
    if remaining > 15.0:
        rng = random.Random(123)
        base_order = list(range(n))
        base_order.sort(key=lambda i: vols[i], reverse=True)

        num_restarts = int(min(50, max(5, remaining / 3)))
        for r in range(num_restarts):
            if time.time() > deadline:
                break
            order = base_order[:]
            num_swaps = rng.randint(2, min(8, n))
            for _ in range(num_swaps):
                a, b = rng.randrange(n), rng.randrange(n)
                if a != b:
                    order[a], order[b] = order[b], order[a]
            strat = rng.choice(strategies)
            placements, vol = _greedy_place(boxes, container, order, deadline, strat)
            if vol > best_vol:
                best_vol = vol
                best_placements = [p[:] for p in placements]

    if best_placements is None:
        # Fallback
        order = list(range(n))
        order.sort(key=lambda i: vols[i], reverse=True)
        best_placements, _ = _greedy_place(boxes, container, order, deadline, 'compact')

    return best_placements
