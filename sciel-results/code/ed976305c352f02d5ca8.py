"""3D Container Loading - multi-strategy packing with shelf, skyline, and local search."""
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


def _shelf_pack_indexed(boxes, container, order):
    """Shelf-based packing with index tracking. Returns list of (box_idx, x, y, z, w, h, d) or None."""
    W, H, D = container
    placed = []  # list of (box_idx, x, y, z, w, h, d)
    
    for idx in order:
        w, h, d = boxes[idx]
        rots = _rotations(w, h, d)
        best_pos = None
        best_vol = -1
        
        for (rw, rh, rd) in rots:
            # Try positions based on existing placements
            candidates = [(0, 0, 0)]
            for (_, px, py, pz, pw, ph, pd) in placed:
                candidates.append((px + pw, py, pz))
                candidates.append((px, py + ph, pz))
                candidates.append((px, py, pz + pd))
            
            for (cx, cy, cz) in candidates:
                if _fits_at(cx, cy, cz, rw, rh, rd, [(p[1],p[2],p[3],p[4],p[5],p[6]) for p in placed], W, H, D):
                    vol = rw * rh * rd
                    # Score: prefer larger volume, then lower position
                    score = vol * 10000 - (cz * 1000 + cy * 10 + cx)
                    if score > best_vol:
                        best_vol = score
                        best_pos = (idx, cx, cy, cz, rw, rh, rd)
        
        if best_pos is not None:
            placed.append(best_pos)
    
    return placed


def _skyline_pack_indexed(boxes, container, order):
    """Skyline-based packing with index tracking."""
    W, H, D = container
    placed = []  # list of (box_idx, x, y, z, w, h, d)
    
    for idx in order:
        w, h, d = boxes[idx]
        rots = _rotations(w, h, d)
        best_pos = None
        best_score = -1
        
        for (rw, rh, rd) in rots:
            candidates = [(0, 0, 0)]
            for (_, px, py, pz, pw, ph, pd) in placed:
                candidates.append((px + pw, py, pz))
                candidates.append((px, py + ph, pz))
                candidates.append((px, py, pz + pd))
            
            for (cx, cy, cz) in candidates:
                if _fits_at(cx, cy, cz, rw, rh, rd, [(p[1],p[2],p[3],p[4],p[5],p[6]) for p in placed], W, H, D):
                    score = -(cz * 10000 + cy * 100 + cx)
                    if score > best_score:
                        best_score = score
                        best_pos = (idx, cx, cy, cz, rw, rh, rd)
        
        if best_pos is not None:
            placed.append(best_pos)
    
    return placed


def _local_search_indexed(placed, boxes, container, max_iters=30):
    """Local search to improve placement."""
    W, H, D = container
    best_placed = list(placed)
    best_vol = sum(p[4]*p[5]*p[6] for p in best_placed)
    
    for _ in range(max_iters):
        improved = False
        for i in range(len(best_placed)):
            idx, x, y, z, w, h, d = best_placed[i]
            remaining = [p for j, p in enumerate(best_placed) if j != i]
            remaining_geo = [(p[1],p[2],p[3],p[4],p[5],p[6]) for p in remaining]
            
            candidates = [(0, 0, 0)]
            for (px, py, pz, pw, ph, pd) in remaining_geo:
                candidates.append((px + pw, py, pz))
                candidates.append((px, py + ph, pz))
                candidates.append((px, py, pz + pd))
            
            best_new_pos = None
            best_new_score = -1
            for (cx, cy, cz) in candidates:
                if _fits_at(cx, cy, cz, w, h, d, remaining_geo, W, H, D):
                    score = -(cz * 10000 + cy * 100 + cx)
                    if score > best_new_score:
                        best_new_score = score
                        best_new_pos = (idx, cx, cy, cz, w, h, d)
            
            if best_new_pos is not None:
                new_placed = remaining + [best_new_pos]
                new_vol = sum(p[4]*p[5]*p[6] for p in new_placed)
                if new_vol > best_vol:
                    best_placed = new_placed
                    best_vol = new_vol
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_placed


def load(boxes, container):
    """Main entry point: place boxes in container, return placements."""
    start_time = time.time()
    deadline = start_time + 140
    W, H, D = container
    n = len(boxes)
    
    # Strategy 1: Volume-decreasing order with shelf packing
    order1 = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    placed1 = _shelf_pack_indexed(boxes, container, order1)
    
    # Strategy 2: Volume-decreasing order with skyline packing
    placed2 = _skyline_pack_indexed(boxes, container, order1)
    
    best_placed = max([placed1, placed2], key=lambda p: sum(x[4]*x[5]*x[6] for x in p))
    
    # Try random orderings
    rng = random.Random(42)
    for trial in range(10):
        if time.time() > deadline - 30:
            break
        order = list(range(n))
        order.sort(key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
        # Perturb the order
        for i in range(n - 1, 0, -1):
            if rng.random() < 0.3:
                j = rng.randint(0, i)
                order[i], order[j] = order[j], order[i]
        
        placed = _shelf_pack_indexed(boxes, container, order)
        vol = sum(p[4]*p[5]*p[6] for p in placed)
        best_vol = sum(p[4]*p[5]*p[6] for p in best_placed)
        if vol > best_vol:
            best_placed = placed
    
    # Local search
    if time.time() < deadline - 10 and best_placed:
        improved = _local_search_indexed(best_placed, boxes, container, max_iters=20)
        if sum(p[4]*p[5]*p[6] for p in improved) > sum(p[4]*p[5]*p[6] for p in best_placed):
            best_placed = improved
    
    # Build output
    placed_indices = set(p[0] for p in best_placed)
    placements = []
    for i in range(n):
        if i in placed_indices:
            for p in best_placed:
                if p[0] == i:
                    _, x, y, z, w, h, d = p
                    placements.append((0, x, y, z, w, h, d))
                    break
        else:
            w, h, d = boxes[i]
            placements.append((-1, 0, 0, 0, w, h, d))
    
    return placements

# EVOLVE-BLOCK-END
