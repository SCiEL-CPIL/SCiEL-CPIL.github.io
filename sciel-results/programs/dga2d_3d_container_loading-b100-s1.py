"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import permutations
import numpy as np

def _get_rotations(w, h, d):
    """Get all unique rotations of a box."""
    rots = set()
    for perm in permutations((w, h, d)):
        rots.add(perm)
    return list(rots)

def _pack_boxes(boxes, container, order_indices, time_limit):
    """
    Attempt to place boxes into the container using Extreme Point heuristic.
    Returns (result, filled_volume) where result is a list of placements.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return [], 0
    
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Precompute rotations for each box
    box_rotations = []
    for w, h, d in boxes:
        rots = _get_rotations(w, h, d)
        # Sort rotations: prefer lower height (z-dim) to stack better, then lower depth, then lower width
        rots.sort(key=lambda r: (r[2], r[1], r[0]))
        box_rotations.append(rots)
    
    placed = []  # List of (x, y, z, w, h, d)
    result = [(-1, 0, 0, 0, 0, 0, 0)] * n
    filled_volume = 0
    
    # Candidate points for placement (Extreme Points)
    candidates = [(0, 0, 0)]
    
    def overlaps_any(px, py, pz, pw, ph, pd):
        for ex, ey, ez, ew, eh, ed in placed:
            if (px < ex + ew and px + pw > ex and
                py < ey + eh and py + ph > ey and
                pz < ez + ed and pz + pd > ez):
                return True
        return False

    def add_candidate(x, y, z):
        if 0 <= x <= W and 0 <= y <= H and 0 <= z <= D:
            # Check if this point is strictly inside an existing box
            inside = False
            for ex, ey, ez, ew, eh, ed in placed:
                if x > ex and x < ex + ew and y > ey and y < ey + eh and z > ez and z < ez + ed:
                    inside = True
                    break
            if not inside:
                if (x, y, z) not in candidates:
                    candidates.append((x, y, z))

    for idx in order_indices:
        if time.time() > deadline:
            break
        
        w, h, d = boxes[idx]
        rots = box_rotations[idx]
        
        placed_this = False
        
        # Sort candidates by Z (lowest), then Y (lowest), then X (leftmost)
        valid_candidates = [c for c in candidates if 0 <= c[0] <= W and 0 <= c[1] <= H and 0 <= c[2] <= D]
        valid_candidates.sort(key=lambda p: (p[2], p[1], p[0]))
        
        for bw, bh, bd in rots:
            if time.time() > deadline:
                break
            
            found = False
            for cx, cy, cz in valid_candidates:
                if cx + bw > W or cy + bh > H or cz + bd > D:
                    continue
                
                if not overlaps_any(cx, cy, cz, bw, bh, bd):
                    placed.append((cx, cy, cz, bw, bh, bd))
                    result[idx] = (0, cx, cy, cz, bw, bh, bd)
                    filled_volume += bw * bh * bd
                    
                    add_candidate(cx + bw, cy, cz)
                    add_candidate(cx, cy + bh, cz)
                    add_candidate(cx, cy, cz + bd)
                    
                    if (cx, cy, cz) in candidates:
                        candidates.remove((cx, cy, cz))
                    
                    placed_this = True
                    found = True
                    break
            
            if found:
                break
        
        if not placed_this:
            result[idx] = (-1, 0, 0, 0, w, h, d)
    
    return result, filled_volume

def load(boxes, container):
    """
    Main entry point: place boxes into container.
    Returns list of (bin, x, y, z, w, h, d) for each box.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    box_volumes = [w * h * d for w, h, d in boxes]
    
    # Strategy 1: Volume descending
    order1 = list(range(n))
    order1.sort(key=lambda i: -box_volumes[i])
    
    # Strategy 2: Max dimension descending
    order2 = list(range(n))
    order2.sort(key=lambda i: -max(boxes[i]))
    
    # Strategy 3: Min dimension descending
    order3 = list(range(n))
    order3.sort(key=lambda i: -min(boxes[i]))
    
    # Strategy 4: Sum of dimensions descending
    order4 = list(range(n))
    order4.sort(key=lambda i: -sum(boxes[i]))
    
    # Strategy 5: Volume ascending (smaller first to fill gaps)
    order5 = list(range(n))
    order5.sort(key=lambda i: box_volumes[i])
    
    # Strategy 6: Random shuffle
    random.seed(42)
    order6 = list(range(n))
    random.shuffle(order6)
    
    # Strategy 7: Sort by aspect ratio (max/min)
    order7 = list(range(n))
    order7.sort(key=lambda i: -max(boxes[i]) / max(min(boxes[i]), 1e-9))
    
    # Strategy 8: Sort by surface area
    order8 = list(range(n))
    order8.sort(key=lambda i: -(2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2])))
    
    orders = [order1, order2, order3, order4, order5, order6, order7, order8]
    
    best_result = None
    best_filled = -1
    
    # Time budget per trial
    # We have 115 seconds total. Let's allocate time dynamically.
    # Start with a few fast trials, then use remaining time for the best candidate if needed?
    # No, just run multiple trials.
    # With 60 boxes, each trial is fast. Let's run all 8 orders.
    # If time remains, we can do more random shuffles.
    
    for order in orders:
        if time.time() > deadline:
            break
        # Allocate remaining time to this trial, but cap it to allow multiple trials
        # Actually, just run it. It should be fast.
        res, filled = _pack_boxes(boxes, container, order, 110.0)
        if filled > best_filled:
            best_filled = filled
            best_result = res
    
    # If we have time left, do more random shuffles
    if best_result is not None and time.time() < deadline - 10.0:
        # Try a few more random shuffles
        for _ in range(20):
            if time.time() > deadline:
                break
            random_order = list(range(n))
            random.shuffle(random_order)
            res, filled = _pack_boxes(boxes, container, random_order, 100.0)
            if filled > best_filled:
                best_filled = filled
                best_result = res
    
    if best_result is None:
        # Fallback: return all -1
        best_result = [(-1, 0, 0, 0, 0, 0, 0)] * n
    
    return best_result
# EVOLVE-BLOCK-END
