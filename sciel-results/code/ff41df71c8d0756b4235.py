"""Combine grid baseline with vectorized hill-climbing refinement."""
import time
import numpy as np
import random
from collections import Counter

# Hard wall-clock budget for the whole construct() call (safety margin under 90s).
_DEADLINE = 78.0


def _doubled_areas(pts):
    """Return (areas, tri_idx): areas = |det| for each triple, tri_idx = (i,j,k) arrays.

    Vectorized over all C(n,3) triples. n is small (30) so this is cheap.
    """
    n = len(pts)
    p = np.asarray(pts, dtype=np.int64)
    ix, jx, kx = np.indices((n, n, n))
    # keep strictly i<j<k
    mask = (ix < jx) & (jx < kx)
    i = ix[mask].astype(np.int64)
    j = jx[mask].astype(np.int64)
    k = kx[mask].astype(np.int64)
    xi, yi = p[i, 0], p[i, 1]
    xj, yj = p[j, 0], p[j, 1]
    xk, yk = p[k, 0], p[k, 1]
    det = (xi * (yj - yk)) + (xj * (yk - yi)) + (xk * (yi - yj))
    areas = np.abs(det)
    return areas, (i, j, k)


def _best_multiplicity(areas):
    """Largest multiplicity of a single positive doubled area."""
    pos = areas[areas > 0]
    if pos.size == 0:
        return 0
    vals, counts = np.unique(pos, return_counts=True)
    return int(counts.max())


def _evaluate(pts):
    areas, _ = _doubled_areas(pts)
    return _best_multiplicity(areas)


def _hill_climb(pts, deadline, radius=3, random_trials=6):
    """Local search: move a point to a nearby empty integer cell if it improves
    the max positive-area multiplicity (tie-break: total area concentration)."""
    pts = [tuple(p) for p in pts]
    cur = _evaluate(pts)
    cur_set = set(pts)

    # Precompute all candidate cells in range.
    cells = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]

    improved = True
    stagnation = 0
    while improved and time.time() < deadline:
        improved = False
        # Try each point; for each, try a bounded set of nearby moves.
        for idx in range(len(pts)):
            if time.time() >= deadline:
                break
            ox, oy = pts[idx]
            # candidate new positions: neighbors within given radius
            cands = []
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ox + dx, oy + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        if (nx, ny) not in cur_set:
                            cands.append((nx, ny))
            # Also try a few random distant cells to escape local minima.
            for _ in range(random_trials):
                c = random.choice(cells)
                if c not in cur_set:
                    cands.append(c)
            for c in cands:
                if time.time() >= deadline:
                    break
                new_pts = list(pts)
                new_pts[idx] = c
                new_set = cur_set - {pts[idx]} | {c}
                # quick filter: must keep 30 distinct
                if len(new_set) != 30:
                    continue
                new_val = _evaluate(new_pts)
                if new_val > cur:
                    # accept
                    cur_set = new_set
                    pts = new_pts
                    cur = new_val
                    improved = True
                    stagnation = 0
                    break
        if not improved:
            stagnation += 1
            if stagnation >= 3:
                break
    return pts, cur


def _hexagonal_seed(n=30, cx=0, cy=0, scale=3):
    """Generate a hexagonal lattice seed of n points centered at (cx,cy)."""
    pts = []
    # Hexagonal packing: rows offset by half spacing
    row_y = 0
    row_offset = 0
    while len(pts) < n:
        for x in range(-15, 16):
            px = cx + x * scale + (row_offset * scale // 2)
            py = cy + row_y * scale
            if -30 <= px <= 30 and -30 <= py <= 30 and (px, py) not in pts:
                pts.append((px, py))
            if len(pts) >= n:
                break
        row_y += 1
        row_offset = 1 - row_offset
    return pts[:n]


def _grid_seed(rows, cols, cx=0, cy=0, sx=3, sy=3):
    """Generate a rectangular grid seed centered at (cx,cy)."""
    pts = []
    for r in range(rows):
        for c in range(cols):
            x = cx + (c - cols // 2) * sx
            y = cy + (r - rows // 2) * sy
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.append((x, y))
    return pts[:30]


def _two_line_seed(n_per_line=15, spacing=2):
    """Two parallel lines with evenly spaced points."""
    pts = []
    start_x = -n_per_line * spacing // 2
    for i in range(n_per_line):
        pts.append((start_x + i * spacing, 0))
        pts.append((start_x + i * spacing, 1))
    return pts[:30]


def construct():
    start = time.time()
    deadline = start + _DEADLINE
    random.seed(42)

    # Multiple diverse seeds
    seeds = [
        ("grid_6x5", [(x, y) for x in range(-2, 4) for y in range(-2, 3)]),
        ("grid_5x6", _grid_seed(5, 6, cx=0, cy=0, sx=3, sy=3)),
        ("hex_scale3", _hexagonal_seed(30, 0, 0, scale=3)),
        ("hex_scale4", _hexagonal_seed(30, 0, 0, scale=4)),
        ("two_lines", _two_line_seed(15, spacing=2)),
        ("grid_10x3", _grid_seed(10, 3, cx=0, cy=0, sx=5, sy=3)),
    ]

    best_pts = None
    best_val = 0

    # Phase 1: Quick hill-climb from each seed with moderate radius
    per_seed_budget = (deadline - start) / len(seeds)
    for name, seed in seeds:
        if time.time() >= deadline - 2:
            break
        seed_pts = [tuple(p) for p in seed[:30]]
        # Ensure exactly 30 distinct points in range
        seen = set()
        valid_seed = []
        for p in seed_pts:
            if -30 <= p[0] <= 30 and -30 <= p[1] <= 30 and p not in seen:
                seen.add(p)
                valid_seed.append(p)
        # Pad if needed
        if len(valid_seed) < 30:
            for x in range(-30, 31):
                for y in range(-30, 31):
                    if len(valid_seed) >= 30:
                        break
                    if (x, y) not in seen:
                        seen.add((x, y))
                        valid_seed.append((x, y))
        valid_seed = valid_seed[:30]

        seed_deadline = min(deadline - 2, time.time() + per_seed_budget)
        pts, val = _hill_climb(valid_seed, seed_deadline, radius=2, random_trials=4)
        if val > best_val:
            best_val = val
            best_pts = pts

    # Phase 2: Refine the best solution with wider search
    if best_pts is not None and time.time() < deadline - 3:
        refine_deadline = deadline - 1
        pts2, val2 = _hill_climb(best_pts, refine_deadline, radius=4, random_trials=8)
        if val2 > best_val:
            best_val = val2
            best_pts = pts2

    # Phase 3: If time remains, try perturbation restarts from the best
    if best_pts is not None and time.time() < deadline - 5:
        for _ in range(3):
            if time.time() >= deadline - 2:
                break
            # Perturb: randomly move 3-5 points
            perturbed = list(best_pts)
            cur_set = set(perturbed)
            num_moves = random.randint(3, 5)
            for _m in range(num_moves):
                idx = random.randint(0, len(perturbed) - 1)
                ox, oy = perturbed[idx]
                for _attempt in range(20):
                    nx = random.randint(max(-30, ox - 8), min(30, ox + 8))
                    ny = random.randint(max(-30, oy - 8), min(30, oy + 8))
                    if (nx, ny) not in cur_set:
                        cur_set.discard(perturbed[idx])
                        cur_set.add((nx, ny))
                        perturbed[idx] = (nx, ny)
                        break
            p3, v3 = _hill_climb(perturbed, min(deadline - 1, time.time() + 8), radius=3, random_trials=6)
            if v3 > best_val:
                best_val = v3
                best_pts = p3

    # Final guard: ensure exactly 30 distinct points in range.
    if best_pts is None:
        best_pts = [(x, y) for x in range(-2, 4) for y in range(-2, 3)][:30]

    seen = set()
    out = []
    for p in best_pts:
        x, y = p
        if -30 <= x <= 30 and -30 <= y <= 30 and (x, y) not in seen:
            seen.add((x, y))
            out.append((x, y))
    # If somehow short (shouldn't happen), pad with fresh cells.
    if len(out) < 30:
        for c in [(x, y) for x in range(-30, 31) for y in range(-30, 31)]:
            if len(out) >= 30:
                break
            if c not in seen:
                seen.add(c)
                out.append(c)
    return out[:30]
