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


def _can_place(ox, oy, oz, bw, bh, bd, placed, W, H, D):
    """Check if a box of size (bw,bh,bd) at (ox,oy,oz) doesn't overlap placed boxes."""
    if ox < 0 or oy < 0 or oz < 0:
        return False
    if ox + bw > W or oy + bh > H or oz + bd > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        # Separation check: if separated on any axis, no overlap
        if ox + bw <= px or px + pw <= ox:
            continue
        if oy + bh <= py or py + ph <= oy:
            continue
        if oz + bd <= pz or pz + pd <= oz:
            continue
        # Overlap on all three axes
        return False
    return True


def _score_position(x, y, z, bw, bh, bd, placed, W, H, D):
    """
    Score a candidate position. Lower is better.
    Uses distance to walls (bottom, left, back) and contact with existing boxes.
    """
    score = 0.0
    # Prefer positions close to bottom, left, back
    score += x * 0.5
    score += y * 0.5
    score += z * 1.0  # Strong preference for lower z
    
    # Bonus for touching existing boxes (compactness)
    contact_bonus = 0.0
    for (px, py, pz, pw, ph, pd) in placed:
        # Check if faces are touching
        if abs((ox + bw) - px) < 1e-9 or abs(ox - (px + pw)) < 1e-9:
            # x-face touching
            if not (oy + bh <= py or oy >= py + ph):
                if not (oz + bd <= pz or oz >= pz + pd):
                    contact_bonus += 1.0
        if abs((oy + bh) - py) < 1e-9 or abs(oy - (py + ph)) < 1e-9:
            if not (ox + bw <= px or ox >= px + pw):
                if not (oz + bd <= pz or oz >= pz + pd):
                    contact_bonus += 1.0
        if abs((oz + bd) - pz) < 1e-9 or abs(oz - (pz + pd)) < 1e-9:
            if not (ox + bw <= px or ox >= px + pw):
                if not (oy + bh <= py or oy >= py + ph):
                    contact_bonus += 1.0
    
    return score - contact_bonus * 2.0


def _best_position_for_box(bw, bh, bd, placed, W, H, D, candidate_points):
    """
    Find the best position for a box with given dimensions.
    Returns (x, y, z) or None if no position found.
    """
    best_pos = None
    best_score = float('inf')
    
    for (cx, cy, cz) in candidate_points:
        if _can_place(cx, cy, cz, bw, bh, bd, placed, W, H, D):
            s = _score_position(cx, cy, cz, bw, bh, bd, placed, W, H, D)
            if s < best_score:
                best_score = s
                best_pos = (cx, cy, cz)
    
    return best_pos


def _generate_candidate_points(placed, W, H, D, box_dims_list):
    """
    Generate candidate (x,y,z) positions where a new box could be placed.
    Based on face points of existing boxes plus origin.
    """
    points = set()
    # Origin
    points.add((0, 0, 0))
    
    for (px, py, pz, pw, ph, pd) in placed:
        # Points on each face
        points.add((px + pw, py, pz))
        points.add((px, py + ph, pz))
        points.add((px, py, pz + pd))
        
        # Also add combinations
        points.add((px + pw, py + ph, pz))
        points.add((px + pw, py, pz + pd))
        points.add((px, py + ph, pz + pd))
        points.add((px + pw, py + ph, pz + pd))
        
        # Points aligned with box edges from other boxes
        for (qx, qy, qz, qw, qh, qd) in placed:
            if q is not p:  # This won't work, need different approach
                pass
    
    # Add more systematic points
    x_vals = set([0, W])
    y_vals = set([0, H])
    z_vals = set([0, D])
    
    for (px, py, pz, pw, ph, pd) in placed:
        x_vals.add(px)
        x_vals.add(px + pw)
        y_vals.add(py)
        y_vals.add(py + ph)
        z_vals.add(pz)
        z_vals.add(pz + pd)
    
    # Generate grid of candidate points
    for x in x_vals:
        for y in y_vals:
            for z in z_vals:
                if x < W and y < H and z < D:
                    points.add((x, y, z))
    
    return list(points)


def _greedy_place(boxes, container, sort_key, deadline):
    """
    Greedy placement with given sorting strategy.
    Returns (placements, volume_filled).
    """
    W, H, D = container
    placed = []
    placements = [None] * len(boxes)
    volume_filled = 0.0
    
    # Sort boxes by given key
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: sort_key(x[1]))
    
    candidate_points = [(0, 0, 0)]
    
    for idx, (w, h, d) in indexed_boxes:
        if time.time() > deadline:
            break
        
        rots = _rotations(w, h, d)
        best_pos = None
        best_rot = None
        best_score = float('inf')
        
        for (rw, rh, rd) in rots:
            # Try to find best position for this rotation
            pos = _best_position_for_box(rw, rh, rd, placed, W, H, D, candidate_points)
            if pos is not None:
                s = _score_position(pos[0], pos[1], pos[2], rw, rh, rd, placed, W, H, D)
                if s < best_score:
                    best_score = s
                    best_pos = pos
                    best_rot = (rw, rh, rd)
        
        if best_pos is not None:
            bx, by, bz = best_pos
            bw, bh, bd = best_rot
            placed.append((bx, by, bz, bw, bh, bd))
            placements[idx] = (0, bx, by, bz, bw, bh, bd)
            volume_filled += bw * bh * bd
            
            # Update candidate points
            new_points = set()
            new_points.add((bx + bw, by, bz))
            new_points.add((bx, by + bh, bz))
            new_points.add((bx, by, bz + bd))
            new_points.add((bx + bw, by + bh, bz))
            new_points.add((bx + bw, by, bz + bd))
            new_points.add((bx, by + bh, bz + bd))
            new_points.add((bx + bw, by + bh, bz + bd))
            
            for pt in new_points:
                if pt not in candidate_points:
                    candidate_points.append(pt)
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)
    
    # Fill in any unplaced boxes
    for i in range(len(boxes)):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    return placements, volume_filled


def _sort_by_volume_desc(box):
    w, h, d = box
    return -(w * h * d)


def _sort_by_max_dim_desc(box):
    w, h, d = box
    return -max(w, h, d)


def _sort_by_min_dim_desc(box):
    w, h, d = box
    return -min(w, h, d)


def _sort_by_volume_asc(box):
    w, h, d = box
    return w * h * d


def _sort_by_max_dim_asc(box):
    w, h, d = box
    return max(w, h, d)


def _sort_by_min_dim_asc(box):
    w, h, d = box
    return min(w, h, d)


def _sort_by_sum_desc(box):
    w, h, d = box
    return -(w + h + d)


def _sort_by_sum_asc(box):
    w, h, d = box
    return w + h + d


def load(boxes, container):
    W, H, D = container
    container_volume = W * H * D
    
    # Deadline with safety margin
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin
    
    # Multiple sorting strategies
    sort_strategies = [
        _sort_by_volume_desc,
        _sort_by_max_dim_desc,
        _sort_by_min_dim_desc,
        _sort_by_sum_desc,
        _sort_by_volume_asc,
        _sort_by_max_dim_asc,
        _sort_by_min_dim_asc,
        _sort_by_sum_asc,
    ]
    
    best_placements = None
    best_volume = -1.0
    
    for sort_key in sort_strategies:
        if time.time() > deadline:
            break
        
        placements, vol = _greedy_place(boxes, container, sort_key, deadline)
        
        if vol > best_volume:
            best_volume = vol
            best_placements = placements
        
        # Local search: try to improve by removing and reinserting
        if time.time() < deadline - 10.0:
            placements, vol = _local_search(placements, boxes, container, deadline)
            if vol > best_volume:
                best_volume = vol
                best_placements = placements
    
    if best_placements is None:
        # Fallback: simple row packing
        placements, _ = _greedy_place(boxes, container, _sort_by_volume_desc, deadline)
        best_placements = placements
    
    return best_placements


def _local_search(placements, boxes, container, deadline):
    """
    Simple local search: try removing each placed box and reinserting it
    to see if we can improve the packing.
    """
    W, H, D = container
    current_volume = 0.0
    
    # Build placed list from placements
    placed = []
    for i, p in enumerate(placements):
        if p[0] == 0:
            _, x, y, z, w, h, d = p
            placed.append((i, x, y, z, w, h, d))
            current_volume += w * h * d
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        for i in range(len(placed) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            idx, bx, by, bz, bw, bh, bd = placed[i]
            
            # Remove this box
            placed_without = placed[:i] + placed[i+1:]
            
            # Try to reinsert at a different position
            best_pos = None
            best_rot = None
            best_score = float('inf')
            
            rots = _rotations(bw, bh, bd)
            candidate_points = _generate_candidate_points(placed_without, W, H, D, boxes)
            
            for (rw, rh, rd) in rots:
                pos = _best_position_for_box(rw, rh, rd, placed_without, W, H, D, candidate_points)
                if pos is not None:
                    s = _score_position(pos[0], pos[1], pos[2], rw, rh, rd, placed_without, W, H, D)
                    if s < best_score:
                        best_score = s
                        best_pos = pos
                        best_rot = (rw, rh, rd)
            
            if best_pos is not None:
                # Check if this is actually different from current position
                if best_pos != (bx, by, bz) or best_rot != (bw, bh, bd):
                    # Replace
                    placed[i] = (idx, best_pos[0], best_pos[1], best_pos[2], best_rot[0], best_rot[1], best_rot[2])
                    improved = True
    
    # Rebuild placements
    new_placements = list(placements)
    for i, (idx, x, y, z, w, h, d) in enumerate(placed):
        new_placements[idx] = (0, x, y, z, w, h, d)
    
    new_volume = 0.0
    for p in new_placements:
        if p[0] == 0:
            _, _, _, _, w, h, d = p
            new_volume += w * h * d
    
    return new_placements, new_volume

# EVOLVE-BLOCK-END
