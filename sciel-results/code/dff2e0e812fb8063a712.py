"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import permutations

def _rotations(w, h, d):
    """Return unique rotations of a box as (w, h, d) tuples, sorted by volume footprint."""
    seen = set()
    result = []
    for perm in permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    # Sort by footprint area (w*h) ascending to prefer compact orientations
    result.sort(key=lambda p: p[0] * p[1])
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
    """Generate candidate positions from face points of already-placed boxes and container walls."""
    points = set()
    # Container bottom-front-left corner
    points.add((0, 0, 0))
    # Points from each placed box's corners and face centers
    for (px, py, pz, pw, ph, pd) in placed:
        # All 8 corners of the placed box
        for dx in [0, pw]:
            for dy in [0, ph]:
                for dz in [0, pd]:
                    nx, ny, nz = px + dx, py + dy, pz + dz
                    if 0 <= nx <= W and 0 <= ny <= H and 0 <= nz <= D:
                        points.add((nx, ny, nz))
        # Also add points just past each face
        if px + pw < W:
            points.add((px + pw, py, pz))
            points.add((px + pw, py + ph, pz))
            points.add((px + pw, py, pz + pd))
            points.add((px + pw, py + ph, pz + pd))
        if py + ph < H:
            points.add((px, py + ph, pz))
            points.add((px + pw, py + ph, pz))
            points.add((px, py + ph, pz + pd))
            points.add((px + pw, py + ph, pz + pd))
        if pz + pd < D:
            points.add((px, py, pz + pd))
            points.add((px + pw, py, pz + pd))
            points.add((px, py + ph, pz + pd))
            points.add((px + pw, py + ph, pz + pd))
    return list(points)


def _score_position(x, y, z, bw, bh, bd, placed, W, H, D):
    """Score a position: prefer positions that minimize wasted space (lower is better)."""
    # Penalize distance from origin (prefer tight packing)
    score = x + y + z
    # Penalize gaps: check how much space is left in each direction
    score += (W - x - bw) * 0.1
    score += (H - y - bh) * 0.1
    score += (D - z - bd) * 0.1
    return score


def _greedy_place(boxes, container, deadline, seed=0):
    """Greedy placement using face points and rotation sorting."""
    random.seed(seed)
    W, H, D = container
    placed = []  # list of (x, y, z, w, h, d)
    result = [None] * len(boxes)
    
    # Sort boxes by volume descending (larger first)
    box_indices = sorted(range(len(boxes)), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    for idx in box_indices:
        w, h, d = boxes[idx]
        rots = _rotations(w, h, d)
        
        # Generate candidate positions
        candidates = _face_points(placed, W, H, D)
        
        best_pos = None
        best_score = float('inf')
        best_rot = None
        
        for rot in rots:
            bw, bh, bd = rot
            for (cx, cy, cz) in candidates:
                if _fits_at(cx, cy, cz, bw, bh, bd, placed, W, H, D):
                    s = _score_position(cx, cy, cz, bw, bh, bd, placed, W, H, D)
                    if s < best_score:
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
        
        if time.time() > deadline:
            # Fill remaining with -1
            for j in range(idx + 1, len(boxes)):
                if result[j] is None:
                    result[j] = (-1, 0, 0, 0, boxes[j][0], boxes[j][1], boxes[j][2])
            break
    
    # Ensure all are filled
    for i in range(len(boxes)):
        if result[i] is None:
            result[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
    
    return result


def _compute_score(placements, boxes, container):
    """Compute the fraction of container volume filled."""
    W, H, D = container
    total_vol = W * H * D
    if total_vol == 0:
        return 0.0
    filled = 0
    for (bin_, x, y, z, w, h, d) in placements:
        if bin_ == 0:
            filled += w * h * d
    return filled / total_vol


def _local_search(placements, boxes, container, deadline, seed=0):
    """Try to improve placement by moving boxes to better positions."""
    random.seed(seed)
    W, H, D = container
    n = len(boxes)
    
    # Separate placed and unplaced
    placed_indices = [i for i in range(n) if placements[i][0] == 0]
    unplaced_indices = [i for i in range(n) if placements[i][0] == -1]
    
    # Build placed list
    placed = [(placements[i][1], placements[i][2], placements[i][3], 
               placements[i][4], placements[i][5], placements[i][6]) 
              for i in placed_indices]
    
    current_score = _compute_score(placements, boxes, container)
    
    # Try to place unplaced boxes first
    for idx in unplaced_indices:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        rots = _rotations(w, h, d)
        candidates = _face_points(placed, W, H, D)
        
        best_pos = None
        best_score = float('inf')
        best_rot = None
        
        for rot in rots:
            bw, bh, bd = rot
            for (cx, cy, cz) in candidates:
                if _fits_at(cx, cy, cz, bw, bh, bd, placed, W, H, D):
                    s = _score_position(cx, cy, cz, bw, bh, bd, placed, W, H, D)
                    if s < best_score:
                        best_score = s
                        best_pos = (cx, cy, cz)
                        best_rot = (bw, bh, bd)
        
        if best_pos is not None:
            x, y, z = best_pos
            bw, bh, bd = best_rot
            placed.append((x, y, z, bw, bh, bd))
            placements[idx] = (0, x, y, z, bw, bh, bd)
    
    # Try to improve positions of placed boxes
    for _ in range(5):  # Limited iterations
        if time.time() > deadline:
            break
        improved = False
        for idx in placed_indices:
            if time.time() > deadline:
                break
            w, h, d = boxes[idx]
            rots = _rotations(w, h, d)
            
            # Remove from placed
            old_placement = (placements[idx][1], placements[idx][2], placements[idx][3],
                           placements[idx][4], placements[idx][5], placements[idx][6])
            placed_without = [p for p in placed if p != old_placement]
            
            candidates = _face_points(placed_without, W, H, D)
            
            best_pos = None
            best_score = float('inf')
            best_rot = None
            
            for rot in rots:
                bw, bh, bd = rot
                for (cx, cy, cz) in candidates:
                    if _fits_at(cx, cy, cz, bw, bh, bd, placed_without, W, H, D):
                        s = _score_position(cx, cy, cz, bw, bh, bd, placed_without, W, H, D)
                        if s < best_score:
                            best_score = s
                            best_pos = (cx, cy, cz)
                            best_rot = (bw, bh, bd)
            
            if best_pos is not None:
                x, y, z = best_pos
                bw, bh, bd = best_rot
                # Replace in placed
                placed.remove(old_placement)
                placed.append((x, y, z, bw, bh, bd))
                placements[idx] = (0, x, y, z, bw, bh, bd)
                improved = True
        
        if not improved:
            break
    
    return placements


def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # Deadline guard: leave 5 seconds margin before 120s limit
    start_time = time.time()
    deadline = start_time + 115.0
    
    # First pass: greedy
    placements = _greedy_place(boxes, container, deadline, seed=42)
    
    if time.time() > deadline:
        return placements
    
    # Second pass: local search to improve
    placements = _local_search(placements, boxes, container, deadline, seed=42)
    
    if time.time() > deadline:
        return placements
    
    # Try a few more greedy passes with different seeds and keep the best
    best_placements = placements
    best_score = _compute_score(placements, boxes, container)
    
    for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        if time.time() > deadline - 5.0:
            break
        p = _greedy_place(boxes, container, deadline, seed=seed)
        s = _compute_score(p, boxes, container)
        if s > best_score:
            best_score = s
            best_placements = p
            # Try local search on the best
            p = _local_search(p, boxes, container, deadline, seed=seed)
            s2 = _compute_score(p, boxes, container)
            if s2 > best_score:
                best_score = s2
                best_placements = p
    
    return best_placements
# EVOLVE-BLOCK-END
