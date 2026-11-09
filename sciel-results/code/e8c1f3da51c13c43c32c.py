"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import permutations

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


def _find_position(bw, bh, bd, placed, W, H, D):
    """Find the lowest feasible position for a box of size (bw,bh,bd)."""
    # Generate candidate positions from face points of placed boxes
    candidates = [(0, 0, 0)]
    for (px, py, pz, pw, ph, pd) in placed:
        # Points along x
        for cx in [px, px + pw]:
            if 0 <= cx < W:
                for cy in [py, py + ph]:
                    if 0 <= cy < H:
                        for cz in [pz, pz + pd]:
                            if 0 <= cz < D:
                                if cx + bw <= W and cy + bh <= H and cz + bd <= D:
                                    candidates.append((cx, cy, cz))
    
    # Sort by z, then y, then x (lowest first)
    candidates.sort(key=lambda p: (p[2], p[1], p[0]))
    
    for (cx, cy, cz) in candidates:
        if _fits_at(cx, cy, cz, bw, bh, bd, placed, W, H, D):
            return (cx, cy, cz)
    
    return None


def _greedy_place(boxes, container, order):
    """Greedy placement using a given order of boxes."""
    W, H, D = container
    placed = []  # list of (x, y, z, w, h, d)
    result = [None] * len(boxes)
    
    for idx in order:
        w, h, d = boxes[idx]
        rotations = _rotations(w, h, d)
        best_pos = None
        best_vol = -1
        
        for (bw, bh, bd) in rotations:
            pos = _find_position(bw, bh, bd, placed, W, H, D)
            if pos is not None:
                if best_pos is None:
                    best_pos = (pos, bw, bh, bd)
                # Prefer lower z, then lower y, then lower x
                pz, py, px = pos
                if best_pos[0] is None or (pz, py, px) < (best_pos[0][2], best_pos[0][1], best_pos[0][0]):
                    best_pos = (pos, bw, bh, bd)
        
        if best_pos is not None:
            (px, py, pz), bw, bh, bd = best_pos
            placed.append((px, py, pz, bw, bh, bd))
            result[idx] = (0, px, py, pz, bw, bh, bd)
        else:
            result[idx] = (-1, 0, 0, 0, w, h, d)
    
    return result


def _score_placement(placements, container):
    """Calculate the fraction of container volume filled."""
    W, H, D = container
    total_vol = W * H * D
    if total_vol == 0:
        return 0.0
    filled = 0
    for p in placements:
        if p[0] == 0:
            filled += p[4] * p[5] * p[6]
    return filled / total_vol


def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Precompute volumes and rotations
    volumes = [w * h * d for w, h, d in boxes]
    rotations_list = [_rotations(w, h, d) for w, h, d in boxes]
    
    best_result = None
    best_score = -1
    
    # Strategy 1: Sort by volume descending
    order_vol_desc = sorted(range(n), key=lambda i: volumes[i], reverse=True)
    result = _greedy_place(boxes, container, order_vol_desc)
    score = _score_placement(result, container)
    if score > best_score:
        best_score = score
        best_result = result
    
    if time.time() > deadline:
        return best_result
    
    # Strategy 2: Sort by volume ascending
    order_vol_asc = sorted(range(n), key=lambda i: volumes[i])
    result = _greedy_place(boxes, container, order_vol_asc)
    score = _score_placement(result, container)
    if score > best_score:
        best_score = score
        best_result = result
    
    if time.time() > deadline:
        return best_result
    
    # Strategy 3: Sort by longest side descending
    def max_side(i):
        w, h, d = boxes[i]
        return max(w, h, d)
    
    order_max_side = sorted(range(n), key=lambda i: max_side(i), reverse=True)
    result = _greedy_place(boxes, container, order_max_side)
    score = _score_placement(result, container)
    if score > best_score:
        best_score = score
        best_result = result
    
    if time.time() > deadline:
        return best_result
    
    # Strategy 4: Random restarts with different orderings
    random.seed(42)
    num_restarts = 0
    while time.time() < deadline:
        # Create a random ordering with some bias toward larger boxes
        order = list(range(n))
        random.shuffle(order)
        # Bias: put larger boxes earlier with some probability
        if random.random() < 0.5:
            order.sort(key=lambda i: volumes[i], reverse=True)
            # Shuffle the first half
            mid = n // 2
            random.shuffle(order[:mid])
        
        result = _greedy_place(boxes, container, order)
        score = _score_placement(result, container)
        if score > best_score:
            best_score = score
            best_result = result
        
        num_restarts += 1
        # Check deadline more frequently
        if num_restarts % 5 == 0 and time.time() > deadline:
            break
    
    return best_result

# EVOLVE-BLOCK-END
