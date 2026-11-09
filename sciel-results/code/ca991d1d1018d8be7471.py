"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import permutations

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

def _candidate_positions(placed, W, H, D):
    """Generate candidate (x,y,z) positions for placing a new box."""
    positions = {(0, 0, 0)}
    for (px, py, pz, pw, ph, pd) in placed:
        if px + pw < W:
            positions.add((px + pw, py, pz))
        if py + ph < H:
            positions.add((px, py + ph, pz))
        if pz + pd < D:
            positions.add((px, py, pz + pd))
    # Also add some "floor" positions
    for (px, py, pz, pw, ph, pd) in placed:
        if px + pw < W and py + ph < H:
            positions.add((px + pw, py + ph, 0))
        if px + pw < W and pz + pd < D:
            positions.add((px + pw, 0, pz + pd))
        if py + ph < H and pz + pd < D:
            positions.add((0, py + ph, pz + pd))
    return list(positions)

def _pack_boxes(boxes, container, seed=42, time_limit=110):
    W, H, D = container
    rng = random.Random(seed)
    
    # Sort boxes by volume descending (larger first)
    box_list = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    box_list.sort(key=lambda x: -x[4])
    
    best_volume = 0
    best_placements = None
    
    # Try multiple strategies
    strategies = [
        'volume_desc',
        'volume_asc',
        'longest_side_desc',
        'random',
    ]
    
    start_time = time.time()
    
    for strategy in strategies:
        if time.time() - start_time > time_limit:
            break
            
        # Order boxes according to strategy
        if strategy == 'volume_desc':
            ordered = sorted(box_list, key=lambda x: -x[4])
        elif strategy == 'volume_asc':
            ordered = sorted(box_list, key=lambda x: x[4])
        elif strategy == 'longest_side_desc':
            ordered = sorted(box_list, key=lambda x: -max(x[1], x[2], x[3]))
        else:  # random
            ordered = box_list[:]
            rng.shuffle(ordered)
        
        placed = []  # list of (x, y, z, w, h, d, box_index)
        placed_indices = set()
        
        for idx, w, h, d, vol in ordered:
            if time.time() - start_time > time_limit:
                break
            
            rotations = _rotations(w, h, d)
            best_pos = None
            best_rot = None
            
            # Try each rotation
            for (rw, rh, rd) in rotations:
                # Generate candidate positions
                candidates = _candidate_positions(placed, W, H, D)
                
                # Sort candidates by some heuristic (e.g., x+y+z to prefer corners)
                candidates.sort(key=lambda p: p[0] + p[1] + p[2])
                
                for (cx, cy, cz) in candidates:
                    if _fits_at(cx, cy, cz, rw, rh, rd, placed, W, H, D):
                        best_pos = (cx, cy, cz)
                        best_rot = (rw, rh, rd)
                        break
                if best_pos is not None:
                    break
            
            if best_pos is not None:
                bx, by, bz = best_pos
                bw, bh, bd = best_rot
                placed.append((bx, by, bz, bw, bh, bd, idx))
                placed_indices.add(idx)
        
        # Calculate total volume placed
        total_vol = sum(w*h*d for (w, h, d) in boxes if boxes.index((w,h,d)) in placed_indices)
        # Actually, let's compute it properly
        total_vol = 0
        for (bx, by, bz, bw, bh, bd, idx) in placed:
            total_vol += bw * bh * bd
        
        if total_vol > best_volume:
            best_volume = total_vol
            best_placements = placed[:]
    
    # Build final result
    container_vol = W * H * D
    result = [(-1, 0, 0, 0, 0, 0, 0)] * len(boxes)
    
    if best_placements:
        for (bx, by, bz, bw, bh, bd, idx) in best_placements:
            orig_w, orig_h, orig_d = boxes[idx]
            result[idx] = (0, bx, by, bz, bw, bh, bd)
    
    return result

def load(boxes, container):
    W, H, D = container
    start_time = time.time()
    
    # Use the packing algorithm
    placements = _pack_boxes(boxes, container, seed=42, time_limit=105)
    
    # Ensure we're within time limit
    if time.time() - start_time > 115:
        # Fallback: simple greedy
        placements = [(-1, 0, 0, 0, 0, 0, 0)] * len(boxes)
        x, y, z, row_h, layer_d = 0, 0, 0, 0, 0
        for i, (w, h, d) in enumerate(boxes):
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d, x, row_h = 0, z + layer_d, 0, 0, 0
            if z + d > D:
                continue
            placements[i] = (0, x, y, z, w, h, d)
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
    
    return placements
# EVOLVE-BLOCK-END
