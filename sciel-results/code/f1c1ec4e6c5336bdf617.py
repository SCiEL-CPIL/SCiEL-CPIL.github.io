"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import permutations
from collections import defaultdict
from heapq import heappush, heappop

# EVOLVE-BLOCK-START

def _rotations(w, h, d):
    """Return unique rotations of a box as (w, h, d) tuples."""
    seen = set()
    result = []
    for perm in permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result


def _rotations_sorted(w, h, d):
    """Return unique rotations sorted by volume (same) then by max-dim ascending (more compact first)."""
    rots = _rotations(w, h, d)
    rots.sort(key=lambda r: (max(r), r[1], r[2]))
    return rots


def _fits_at(x, y, z, bw, bh, bd, placed, W, H, D):
    """Check if a box of size (bw,bh,bd) fits at (x,y,z) without overlap."""
    if x < 0 or y < 0 or z < 0:
        return False
    if x + bw > W or y + bh > H or z + bd > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        if not (x + bw <= px or px + pw <= x):
            if not (y + bh <= py or py + ph <= y):
                if not (z + bd <= pz or pz + pd <= z):
                    return False
    return True


def _fits_at_fast(x, y, z, bw, bh, bd, placed, W, H, D):
    """Faster fit check using precomputed bounds for early termination."""
    if x < 0 or y < 0 or z < 0:
        return False
    if x + bw > W or y + bh > H or z + bd > D:
        return False
    for i in range(len(placed)):
        px, py, pz, pw, ph, pd = placed[i]
        if px + pw <= x:
            continue
        if x + bw <= px:
            continue
        if py + ph <= y:
            continue
        if y + bh <= py:
            continue
        if pz + pd <= z:
            continue
        if z + bd <= pz:
            continue
        return False
    return True


def _generate_candidates(bw, bh, bd, placed, W, H, D):
    """Generate candidate (x, y, z) positions for a box of given size."""
    candidates = set()
    candidates.add((0, 0, 0))
    
    if not placed:
        return [(0, 0, 0)]
    
    # Generate positions based on existing box boundaries
    for (px, py, pz, pw, ph, pd) in placed:
        # Right of box
        if px + pw + bw <= W:
            candidates.add((px + pw, py, pz))
            candidates.add((px + pw, py + ph, pz))
            candidates.add((px + pw, py, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
        # Left of box (if box not at x=0)
        if px >= bw:
            candidates.add((px - bw, py, pz))
            candidates.add((px - bw, py + ph, pz))
            candidates.add((px - bw, py, pz + pd))
            candidates.add((px - bw, py + ph, pz + pd))
        # Above box
        if pz + pd + bd <= D:
            candidates.add((px, py, pz + pd))
            candidates.add((px + pw, py, pz + pd))
            candidates.add((px, py + ph, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
        # In front/behind in y
        if py + ph + bh <= H:
            candidates.add((px, py + ph, pz))
            candidates.add((px + pw, py + ph, pz))
            candidates.add((px, py + ph, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
        if py >= bh:
            candidates.add((px, py - bh, pz))
            candidates.add((px + pw, py - bh, pz))
            candidates.add((px, py - bh, pz + pd))
            candidates.add((px + pw, py - bh, pz + pd))
        # Behind in z
        if pz + pd + bd <= D:
            candidates.add((px, py, pz + pd))
            candidates.add((px + pw, py, pz + pd))
            candidates.add((px, py + ph, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
        if pz >= bd:
            candidates.add((px, py, pz - bd))
            candidates.add((px + pw, py, pz - bd))
            candidates.add((px, py + ph, pz - bd))
            candidates.add((px + pw, py + ph, pz - bd))
    
    # Also try some grid-based positions for diversity
    xs = {0}
    ys = {0}
    zs = {0}
    for (px, py, pz, pw, ph, pd) in placed:
        xs.add(px)
        xs.add(px + pw)
        ys.add(py)
        ys.add(py + ph)
        zs.add(pz)
        zs.add(pz + pd)
    
    xs = sorted(set(x for x in xs if 0 <= x <= W - bw))
    ys = sorted(set(y for y in ys if 0 <= y <= H - bh))
    zs = sorted(set(z for z in zs if 0 <= z <= D - bd))
    
    if len(xs) <= 12 and len(ys) <= 12:
        for x in xs[:8]:
            for y in ys[:8]:
                min_z = 0
                for (px, py, pz, pw, ph, pd) in placed:
                    if not (x + bw <= px or px + pw <= x):
                        if not (y + bh <= py or py + ph <= y):
                            if pz + pd > min_z:
                                min_z = pz + pd
                if min_z + bd <= D:
                    candidates.add((x, y, min_z))
    
    # Filter candidates
    valid = []
    for (cx, cy, cz) in candidates:
        if 0 <= cx and cx + bw <= W and 0 <= cy and cy + bh <= H and 0 <= cz and cz + bd <= D:
            valid.append((cx, cy, cz))
    
    return valid


def _score_position(x, y, z, bw, bh, bd, placed, W, H, D):
    """Score a position: prefer compact, low, and well-connected."""
    # Primary: minimize z (pack from bottom)
    # Secondary: minimize y, then x
    # Tertiary: maximize contact with existing boxes
    contact = 0
    for (px, py, pz, pw, ph, pd) in placed:
        # Check face contacts
        if abs((x + bw) - px) < 1e-9:
            oy = min(y + bh, py + ph) - max(y, py)
            oz = min(z + bd, pz + pd) - max(z, pz)
            if oy > 0 and oz > 0:
                contact += oy * oz
        if abs((y + bh) - py) < 1e-9:
            ox = min(x + bw, px + pw) - max(x, px)
            oz = min(z + bd, pz + pd) - max(z, pz)
            if ox > 0 and oz > 0:
                contact += ox * oz
        if abs((z + bd) - pz) < 1e-9:
            ox = min(x + bw, px + pw) - max(x, px)
            oy = min(y + bh, py + ph) - max(y, py)
            if ox > 0 and oy > 0:
                contact += ox * oy
    
    return (z, y, x, -contact)


def _fast_greedy_place(order, boxes, container, deadline):
    """Fast greedy placement using candidate generation."""
    W, H, D = container
    n = len(boxes)
    placed = []
    result = [None] * n
    
    for idx in order:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        rots = _rotations_sorted(w, h, d)
        
        best_pos = None
        best_score = None
        best_rot = None
        
        for (bw, bh, bd) in rots:
            candidates = _generate_candidates(bw, bh, bd, placed, W, H, D)
            for (cx, cy, cz) in candidates:
                if _fits_at_fast(cx, cy, cz, bw, bh, bd, placed, W, H, D):
                    s = _score_position(cx, cy, cz, bw, bh, bd, placed, W, H, D)
                    if best_score is None or s < best_score:
                        best_score = s
                        best_pos = (cx, cy, cz)
                        best_rot = (bw, bh, bd)
        
        if best_pos is not None:
            x, y, z = best_pos
            bw, bh, bd = best_rot
            placed.append((x, y, z, bw, bh, bd))
            result[idx] = (0, x, y, z, bw, bh, bd)
        else:
            result[idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(n):
        if result[i] is None:
            w, h, d = boxes[i]
            result[i] = (-1, 0, 0, 0, w, h, d)
    
    return result


def _compute_volume(result):
    """Compute total volume of placed boxes."""
    total = 0
    for r in result:
        if r[0] == 0:
            total += r[4] * r[5] * r[6]
    return total


def _heuristic_orders(boxes, n):
    """Generate multiple heuristic orderings."""
    orders = []
    
    vol_order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    orders.append(('vol', vol_order))
    
    maxdim_order = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    orders.append(('maxdim', maxdim_order))
    
    sa_order = sorted(range(n), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)
    orders.append(('sa', sa_order))
    
    mindim_order = sorted(range(n), key=lambda i: min(boxes[i]), reverse=True)
    orders.append(('mindim', mindim_order))
    
    # Compactness: prefer more cube-like boxes first
    compact_order = sorted(range(n), key=lambda i: max(boxes[i]) / max(1, min(boxes[i])), reverse=False)
    orders.append(('compact', compact_order))
    
    # Volume-weighted by max dimension
    vol_maxdim_order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2] * max(boxes[i]), reverse=True)
    orders.append(('vol_maxdim', vol_maxdim_order))
    
    # Aspect ratio: prefer boxes with lower aspect ratio (more compact)
    aspect_order = sorted(range(n), key=lambda i: max(boxes[i])/min(boxes[i]) if min(boxes[i])>0 else 100, reverse=False)
    orders.append(('aspect', aspect_order))
    
    return orders


def _random_orders(boxes, n, count, seed):
    """Generate random orderings."""
    rng = random.Random(seed)
    orders = []
    for _ in range(count):
        order = list(range(n))
        rng.shuffle(order)
        orders.append(('rand', order))
    return orders


def _local_search(result, boxes, container, deadline, max_iters=100):
    """Local search: try to improve by moving individual boxes and placing unplaced boxes."""
    W, H, D = container
    n = len(boxes)
    best_result = [tuple(r) for r in result]
    best_vol = _compute_volume(best_result)
    rng = random.Random(42)
    
    # First pass: try to place unplaced boxes
    unplaced_indices = [i for i in range(n) if best_result[i][0] == -1]
    for _ in range(min(len(unplaced_indices), max_iters // 3)):
        if time.time() > deadline:
            break
        if not unplaced_indices:
            break
        idx = rng.choice(unplaced_indices)
        w, h, d = boxes[idx]
        rots = _rotations_sorted(w, h, d)
        
        placed = []
        for i in range(n):
            if best_result[i][0] == 0:
                x, y, z, bw, bh, bd = best_result[i][1:7]
                placed.append((x, y, z, bw, bh, bd))
        
        if not placed:
            candidates = [(0, 0, 0)]
        else:
            candidates = _generate_candidates(w, h, d, placed, W, H, D)
        
        best_pos = None
        best_score = None
        best_rot = None
        
        for (bw, bh, bd) in rots:
            cands = _generate_candidates(bw, bh, bd, placed, W, H, D)
            for (cx, cy, cz) in cands:
                if _fits_at_fast(cx, cy, cz, bw, bh, bd, placed, W, H, D):
                    s = _score_position(cx, cy, cz, bw, bh, bd, placed, W, H, D)
                    if best_score is None or s < best_score:
                        best_score = s
                        best_pos = (cx, cy, cz)
                        best_rot = (bw, bh, bd)
        
        if best_pos is not None:
            x, y, z = best_pos
            bw, bh, bd = best_rot
            temp_result = list(best_result)
            temp_result[idx] = (0, x, y, z, bw, bh, bd)
            new_vol = _compute_volume(temp_result)
            if new_vol > best_vol:
                best_result = [tuple(r) for r in temp_result]
                best_vol = new_vol
                unplaced_indices = [i for i in unplaced_indices if i != idx]
    
    # Second pass: move placed boxes to improve packing
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        placed_indices = [i for i in range(n) if best_result[i][0] == 0]
        if not placed_indices:
            break
        
        idx = rng.choice(placed_indices)
        
        temp_result = [tuple(r) for r in best_result]
        temp_result[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
        
        placed = []
        for i in range(n):
            if i != idx and temp_result[i][0] == 0:
                x, y, z, bw, bh, bd = temp_result[i][1:7]
                placed.append((x, y, z, bw, bh, bd))
        
        w, h, d = boxes[idx]
        rots = _rotations_sorted(w, h, d)
        
        if not placed:
            candidates = [(0, 0, 0)]
        else:
            candidates = _generate_candidates(w, h, d, placed, W, H, D)
        
        best_pos = None
        best_score = None
        best_rot = None
        
        for (bw, bh, bd) in rots:
            cands = _generate_candidates(bw, bh, bd, placed, W, H, D)
            for (cx, cy, cz) in cands:
                if _fits_at_fast(cx, cy, cz, bw, bh, bd, placed, W, H, D):
                    s = _score_position(cx, cy, cz, bw, bh, bd, placed, W, H, D)
                    if best_score is None or s < best_score:
                        best_score = s
                        best_pos = (cx, cy, cz)
                        best_rot = (bw, bh, bd)
        
        if best_pos is not None:
            x, y, z = best_pos
            bw, bh, bd = best_rot
            new_placement = (0, x, y, z, bw, bh, bd)
            temp_result[idx] = new_placement
            new_vol = _compute_volume(temp_result)
            if new_vol > best_vol:
                best_result = [tuple(r) for r in temp_result]
                best_vol = new_vol
    
    return best_result


def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    deadline = time.time() + 110
    
    if n == 0:
        return []
    
    heuristic_orders = _heuristic_orders(boxes, n)
    
    best_result = None
    best_vol = -1
    
    # Run greedy with different orderings
    for name, order in heuristic_orders:
        if time.time() > deadline:
            break
        result = _fast_greedy_place(order, boxes, container, deadline)
        vol = _compute_volume(result)
        if vol > best_vol:
            best_vol = vol
            best_result = result
    
    # Additional random restarts if time permits
    if time.time() < deadline - 10 and best_result is not None:
        rng_restart = random.Random(123)
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            random_order = list(range(n))
            rng_restart.shuffle(random_order)
            result_rand = _fast_greedy_place(random_order, boxes, container, deadline)
            vol_rand = _compute_volume(result_rand)
            if vol_rand > best_vol:
                best_vol = vol_rand
                best_result = result_rand
    
    # Local search to improve
    remaining_time = deadline - time.time()
    if remaining_time > 5 and best_result is not None:
        max_iters = int(remaining_time * 30)  # Scale iterations with time
        max_iters = max(50, min(max_iters, 500))
        best_result = _local_search(best_result, boxes, container, deadline, max_iters=max_iters)
    
    if best_result is None:
        best_result = []
        for w, h, d in boxes:
            best_result.append((-1, 0, 0, 0, w, h, d))
    
    return [tuple(r) for r in best_result]

# EVOLVE-BLOCK-END
