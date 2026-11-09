import time
import random
import itertools
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START

def _get_orientations(w, h, d):
    """Return all unique orientations of a box (w, h, d)."""
    dims = [w, h, d]
    orients = set()
    for perm in itertools.permutations(dims):
        orients.add(perm)
    return list(orients)

def _can_place(px, py, pz, pw, ph, pd, W, H, D, placed):
    """Check if a box can be placed at (px, py, pz) with dimensions (pw, ph, pd)."""
    if px < 0 or py < 0 or pz < 0:
        return False
    if px + pw > W or py + ph > H or pz + pd > D:
        return False
    for (qx, qy, qz, qw, qh, qd) in placed:
        # Check overlap: boxes overlap if they intersect in all 3 axes
        if not (px + pw <= qx or qx + qw <= px):
            if not (py + ph <= qy or qy + qh <= py):
                if not (pz + pd <= qz or qz + qd <= pz):
                    return False
    return True

def _find_position(box_dims, W, H, D, placed, deadline):
    """Try to find a valid position for a box with given dimensions.
    Returns (x, y, z) or None."""
    pw, ph, pd = box_dims
    
    # Generate candidate positions:
    # 1. Origin
    # 2. Adjacent to each placed box in 6 directions
    candidates = set()
    candidates.add((0, 0, 0))
    
    for (qx, qy, qz, qw, qh, qd) in placed:
        # Right
        candidates.add((qx + qw, qy, qz))
        # Left
        candidates.add((qx - pw, qy, qz))
        # Front
        candidates.add((qx, qy + qh, qz))
        # Back
        candidates.add((qx, qy - ph, qz))
        # Top
        candidates.add((qx, qy, qz + qd))
        # Bottom
        candidates.add((qx, qy, qz - pd))
        
        # Also try corners/edges combinations
        candidates.add((qx + qw, qy + qh, qz))
        candidates.add((qx + qw, qy, qz + qd))
        candidates.add((qx, qy + qh, qz + qd))
        candidates.add((qx + qw, qy + qh, qz + qd))
        candidates.add((qx - pw, qy + qh, qz))
        candidates.add((qx, qy - ph, qz + qd))
        candidates.add((qx + qw, qy - ph, qz))
        candidates.add((qx - pw, qy, qz + qd))
    
    # Sort candidates by a heuristic: prefer lower z, then lower y, then lower x
    # This tends to pack things tightly
    def sort_key(pos):
        x, y, z = pos
        return (z, y, x)
    
    sorted_candidates = sorted(candidates, key=sort_key)
    
    for (px, py, pz) in sorted_candidates:
        if time.time() > deadline:
            return None
        if _can_place(px, py, pz, pw, ph, pd, W, H, D, placed):
            return (px, py, pz)
    
    return None

def _greedy_pack(boxes, container, deadline, order=None):
    """Greedy packing with orientation choice.
    Returns list of (bin, x, y, z, w, h, d) for each box."""
    W, H, D = container
    n = len(boxes)
    
    if order is None:
        order = list(range(n))
    
    placements = [None] * n
    placed = []  # list of (x, y, z, w, h, d)
    
    for idx in order:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        orients = _get_orientations(w, h, d)
        
        best_pos = None
        best_orient = None
        
        for (pw, ph, pd) in orients:
            pos = _find_position((pw, ph, pd), W, H, D, placed, deadline)
            if pos is not None:
                best_pos = pos
                best_orient = (pw, ph, pd)
                break
        
        if best_pos is not None:
            px, py, pz = best_pos
            pw, ph, pd = best_orient
            placements[idx] = (0, px, py, pz, pw, ph, pd)
            placed.append((px, py, pz, pw, ph, pd))
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)
    
    # Fill in any None entries (boxes not yet processed due to deadline)
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    return placements

def _score(placements, boxes, container):
    """Calculate the fraction of container volume filled."""
    W, H, D = container
    container_vol = W * H * D
    if container_vol == 0:
        return 0.0
    
    filled = 0
    for i, (bin_id, x, y, z, w, h, d) in enumerate(placements):
        if bin_id == 0:
            filled += w * h * d
    
    return filled / container_vol

def _sort_by_volume_desc(boxes):
    """Sort box indices by volume descending."""
    n = len(boxes)
    volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    return sorted(range(n), key=lambda i: -volumes[i])

def _sort_by_max_dim_desc(boxes):
    """Sort box indices by maximum dimension descending."""
    n = len(boxes)
    max_dims = [max(boxes[i]) for i in range(n)]
    return sorted(range(n), key=lambda i: -max_dims[i])

def _sort_by_volume_asc(boxes):
    """Sort box indices by volume ascending."""
    n = len(boxes)
    volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    return sorted(range(n), key=lambda i: volumes[i])

def _random_order(n, seed):
    """Generate a random order with given seed."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    return order

def load(boxes, container):
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin for 120s limit
    
    W, H, D = container
    container_vol = W * H * D
    if container_vol == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    
    # Try multiple strategies and keep the best
    best_score = -1
    best_placements = None
    
    # Strategy 1: Volume descending
    order1 = _sort_by_volume_desc(boxes)
    if time.time() < DEADLINE:
        placements1 = _greedy_pack(boxes, container, DEADLINE, order1)
        score1 = _score(placements1, boxes, container)
        if score1 > best_score:
            best_score = score1
            best_placements = placements1
    
    # Strategy 2: Max dimension descending
    order2 = _sort_by_max_dim_desc(boxes)
    if time.time() < DEADLINE:
        placements2 = _greedy_pack(boxes, container, DEADLINE, order2)
        score2 = _score(placements2, boxes, container)
        if score2 > best_score:
            best_score = score2
            best_placements = placements2
    
    # Strategy 3: Volume ascending
    order3 = _sort_by_volume_asc(boxes)
    if time.time() < DEADLINE:
        placements3 = _greedy_pack(boxes, container, DEADLINE, order3)
        score3 = _score(placements3, boxes, container)
        if score3 > best_score:
            best_score = score3
            best_placements = placements3
    
    # Strategy 4: Random orders with different seeds
    remaining_time = DEADLINE - time.time()
    if remaining_time > 5:
        num_random_tries = max(1, int(remaining_time / 2))
        for seed in range(num_random_tries):
            if time.time() > DEADLINE - 2:
                break
            order_r = _random_order(n, seed)
            placements_r = _greedy_pack(boxes, container, DEADLINE, order_r)
            score_r = _score(placements_r, boxes, container)
            if score_r > best_score:
                best_score = score_r
                best_placements = placements_r
    
    if best_placements is None:
        # Fallback: simple shelf packing
        W, H, D = container
        placements = []
        x, y, z, row_h, layer_d = 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d, x, row_h = 0, z + layer_d, 0, 0, 0
            if z + d > D:
                placements.append((-1, 0, 0, 0, w, h, d))
                continue
            placements.append((0, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        best_placements = placements
    
    return best_placements

# EVOLVE-BLOCK-END
