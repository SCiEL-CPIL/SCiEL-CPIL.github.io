"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import permutations

def _get_rotations(w, h, d):
    """Get all unique rotations of a box."""
    rots = set()
    for perm in permutations((w, h, d)):
        rots.add(perm)
    return list(rots)

def _try_place(boxes, container, sort_key=None):
    """
    Attempt to place boxes into the container using Extreme Point heuristic.
    Returns a list of placements: (bin, x, y, z, w, h, d) for each box.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Precompute rotations for each box
    box_rotations = []
    for w, h, d in boxes:
        rots = _get_rotations(w, h, d)
        # Sort rotations: prefer lower height (z-dim) to stack better, then lower depth, then lower width
        rots.sort(key=lambda r: (r[2], r[1], r[0]))
        box_rotations.append(rots)
    
    # Sort boxes based on sort_key
    box_indices = list(range(n))
    if sort_key is None:
        box_volumes = [w * h * d for w, h, d in boxes]
        box_indices.sort(key=lambda i: -box_volumes[i])
    else:
        box_indices.sort(key=sort_key)
    
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
            inside = False
            for ex, ey, ez, ew, eh, ed in placed:
                if x > ex and x < ex + ew and y > ey and y < ey + eh and z > ez and z < ez + ed:
                    inside = True
                    break
            if not inside:
                if (x, y, z) not in candidates:
                    candidates.append((x, y, z))

    for idx in box_indices:
        if time.time() > deadline:
            break
        
        w, h, d = boxes[idx]
        rots = box_rotations[idx]
        
        placed_this = False
        
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
    
    return result

def _evaluate_result(placement, container):
    """Evaluate the fraction of container volume filled."""
    W, H, D = container
    total_volume = W * H * D
    if total_volume == 0:
        return 0.0
    
    filled = 0
    for bin_id, x, y, z, w, h, d in placement:
        if bin_id == 0:
            if x >= 0 and y >= 0 and z >= 0 and \
               x + w <= W and y + h <= H and z + d <= D:
                filled += w * h * d
    
    return filled / total_volume

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
    
    # Define multiple sorting strategies
    strategies = [
        lambda i: -(boxes[i][0] * boxes[i][1] * boxes[i][2]),  # Volume desc
        lambda i: -boxes[i][0],  # Width desc
        lambda i: -boxes[i][1],  # Height desc
        lambda i: -boxes[i][2],  # Depth desc
        lambda i: -(boxes[i][0] + boxes[i][1] + boxes[i][2]),  # Sum of dims desc
        lambda i: -max(boxes[i]),  # Max dim desc
        lambda i: -min(boxes[i]),  # Min dim desc
    ]
    
    best_placement = None
    best_score = -1
    
    # Try each strategy, keeping the best result
    for strategy in strategies:
        if time.time() > deadline:
            break
        
        placement = _try_place(boxes, container, sort_key=strategy)
        score = _evaluate_result(placement, container)
        
        if score > best_score:
            best_score = score
            best_placement = placement
    
    return best_placement if best_placement is not None else [(-1, 0, 0, 0, 0, 0, 0)] * n
# EVOLVE-BLOCK-END
