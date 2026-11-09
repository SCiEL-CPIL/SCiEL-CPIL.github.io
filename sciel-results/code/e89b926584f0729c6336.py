"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import permutations

def _rotations(w, h, d):
    """Return unique rotations of a box as (w, h, d) tuples, sorted by volume (all same) then by dimensions."""
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

def _face_points(placed, W, H, D):
    """Generate candidate corner points from placed boxes and container origin."""
    points = set()
    points.add((0, 0, 0))
    for (px, py, pz, pw, ph, pd) in placed:
        # All 8 corners of each placed box
        for dx in (0, pw):
            for dy in (0, ph):
                for dz in (0, pd):
                    cx, cy, cz = px + dx, py + dy, pz + dz
                    if cx <= W and cy <= H and cz <= D:
                        points.add((cx, cy, cz))
    return list(points)

def _score_position(x, y, z, bw, bh, bd):
    """Score a position: prefer lower z, then lower y, then lower x."""
    return (z, y, x)

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    placements = [(-1, 0, 0, 0, 0, 0, 0)] * n
    placed = []
    
    # Precompute rotations for each box
    box_rots = [_rotations(w, h, d) for w, h, d in boxes]
    
    # Sort boxes by volume descending for better packing
    order = list(range(n))
    order.sort(key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)
    
    # Deadline guard: aim to finish well within 120s
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    for idx in order:
        if time.time() > deadline:
            break
        
        w, h, d = boxes[idx]
        rots = box_rots[idx]
        
        best_pos = None
        best_score = None
        best_rot = None
        
        # Generate candidate points
        points = _face_points(placed, W, H, D)
        
        # Try each rotation and each candidate point
        for (bw, bh, bd) in rots:
            for (px, py, pz) in points:
                if _fits_at(px, py, pz, bw, bh, bd, placed, W, H, D):
                    score = _score_position(px, py, pz, bw, bh, bd)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pos = (px, py, pz)
                        best_rot = (bw, bh, bd)
        
        if best_pos is not None:
            px, py, pz = best_pos
            bw, bh, bd = best_rot
            placements[idx] = (0, px, py, pz, bw, bh, bd)
            placed.append((px, py, pz, bw, bh, bd))
    
    return placements
# EVOLVE-BLOCK-END
