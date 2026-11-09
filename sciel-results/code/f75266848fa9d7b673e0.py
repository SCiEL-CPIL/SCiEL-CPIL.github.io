"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from functools import cmp_to_key

def _rotations(dims):
    """Return unique rotations of a box (w, h, d)."""
    w, h, d = dims
    seen = set()
    rots = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            rots.append(perm)
    return rots

def _fits(placed, x, y, z, w, h, d, W, H, D):
    """Check if a box at (x,y,z) with size (w,h,d) fits without overlap and within container."""
    if x + w > W or y + h > H or z + d > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        if (x < px + pw and px < x + w and
            y < py + ph and py < y + h and
            z < pz + pd and pz < z + d):
            return False
    return True

def _candidate_positions(placed, ow, oh, od, W, H, D):
    """Generate candidate positions for a box of size (ow, oh, od)."""
    positions = set()
    positions.add((0, 0, 0))
    
    for px, py, pz, pw, ph, pd in placed:
        x_opts = {px, px + pw}
        y_opts = {py, py + ph}
        z_opts = {pz, pz + pd}
        
        for cx in x_opts:
            for cy in y_opts:
                for cz in z_opts:
                    if 0 <= cx <= W - ow and 0 <= cy <= H - oh and 0 <= cz <= D - od:
                        positions.add((cx, cy, cz))
    
    if placed:
        max_x = min(W - ow, max(px + pw for px, py, pz, pw, ph, pd in placed))
        max_y = min(H - oh, max(py + ph for px, py, pz, pw, ph, pd in placed))
        max_z = min(D - od, max(pz + pd for px, py, pz, pw, ph, pd in placed))
        for mx in [0, max_x]:
            for my in [0, max_y]:
                for mz in [0, max_z]:
                    if 0 <= mx <= W - ow and 0 <= my <= H - oh and 0 <= mz <= D - od:
                        positions.add((mx, my, mz))
    
    return list(positions)

def _score_position(x, y, z, ow, oh, od, placed, W, H, D):
    """Score a position: prefer tight packing."""
    base = -(x + y + z)
    adjacency = 0
    for px, py, pz, pw, ph, pd in placed:
        if abs(x - (px + pw)) < 1e-9 or abs(px - (x + ow)) < 1e-9:
            adjacency += 1
        if abs(y - (py + ph)) < 1e-9 or abs(py - (y + oh)) < 1e-9:
            adjacency += 1
        if abs(z - (pz + pd)) < 1e-9 or abs(pz - (z + od)) < 1e-9:
            adjacency += 1
    return base - adjacency * 0.1

def _place_box(placed, boxes, idx, W, H, D, deadline, start_time):
    """Try to place box idx in the best position/orientation."""
    w, h, d = boxes[idx]
    best_orient = None
    best_pos = None
    best_score = float('inf')
    
    for ow, oh, od in _rotations((w, h, d)):
        if ow > W or oh > H or od > D:
            continue
        
        positions = _candidate_positions(placed, ow, oh, od, W, H, D)
        
        for (tx, ty, tz) in positions:
            if not _fits(placed, tx, ty, tz, ow, oh, od, W, H, D):
                continue
            sc = _score_position(tx, ty, tz, ow, oh, od, placed, W, H, D)
            if sc < best_score:
                best_score = sc
                best_orient = (ow, oh, od)
                best_pos = (tx, ty, tz)
    
    if best_pos is not None:
        placed.append((best_pos[0], best_pos[1], best_pos[2], best_orient[0], best_orient[1], best_orient[2]))
        return True
    return False

def _solve_one(boxes, container, order, deadline, start_time):
    """Solve with a given ordering of boxes."""
    W, H, D = container
    placed = []
    result = {}
    
    for idx in order:
        if time.time() - start_time > deadline:
            break
        if _place_box(placed, boxes, idx, W, H, D, deadline, start_time):
            x, y, z, w, h, d = placed[-1]
            result[idx] = (x, y, z, w, h, d)
    
    return result

def _sort_by_volume(boxes):
    return sorted(range(len(boxes)), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)

def _sort_by_max_dim(boxes):
    return sorted(range(len(boxes)), key=lambda i: max(boxes[i]), reverse=True)

def _sort_by_min_dim(boxes):
    return sorted(range(len(boxes)), key=lambda i: min(boxes[i]), reverse=True)

def _sort_by_base_area(boxes):
    return sorted(range(len(boxes)), key=lambda i: boxes[i][0]*boxes[i][1], reverse=True)

def _random_order(n, rng):
    order = list(range(n))
    rng.shuffle(order)
    return order

def load(boxes, container):
    start_time = time.time()
    DEADLINE = 145.0
    W, H, D = container
    n = len(boxes)
    container_vol = W * H * D
    rng = random.Random(42)
    
    best_result = {}
    best_fill = 0.0
    
    strategies = [
        _sort_by_volume,
        _sort_by_max_dim,
        _sort_by_min_dim,
        _sort_by_base_area,
    ]
    
    # Try deterministic strategies first
    for strat in strategies:
        if time.time() - start_time > DEADLINE:
            break
        order = strat(boxes)
        result = _solve_one(boxes, container, order, DEADLINE, start_time)
        fill = sum(w*h*d for (x, y, z, w, h, d) in result.values()) / container_vol
        if fill > best_fill:
            best_fill = fill
            best_result = dict(result)
    
    # Random restarts with volume-sorted base + perturbations
    base_order = _sort_by_volume(boxes)
    n_restarts = 50
    
    for r in range(n_restarts):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Create perturbed order: swap n/3 random pairs
        order = list(base_order)
        n_swaps = max(1, n // 3)
        for _ in range(n_swaps):
            i, j = rng.sample(range(n), 2)
            order[i], order[j] = order[j], order[i]
        
        result = _solve_one(boxes, container, order, DEADLINE, start_time)
        fill = sum(w*h*d for (x, y, z, w, h, d) in result.values()) / container_vol
        if fill > best_fill:
            best_fill = fill
            best_result = dict(result)
    
    # Build final placements
    placements = [(-1, 0, 0, 0, 0, 0, 0)] * n
    for i in range(n):
        if i in best_result:
            x, y, z, w, h, d = best_result[i]
            placements[i] = (0, x, y, z, w, h, d)
        else:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    return placements
# EVOLVE-BLOCK-END
