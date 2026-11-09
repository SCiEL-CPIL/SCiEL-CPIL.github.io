"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import heapq
import random

def _unique_rotations(w, h, d):
    """Return unique rotations of a box as (w, h, d) tuples."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _overlaps(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
    """Check if two boxes overlap."""
    if x1 + w1 <= x2 or x2 + w2 <= x1:
        return False
    if y1 + h1 <= y2 or y2 + h2 <= y1:
        return False
    if z1 + d1 <= z2 or z2 + d2 <= z1:
        return False
    return True

def _can_place(x, y, z, bw, bh, bd, placed, W, H, D):
    """Check if a box can be placed at (x,y,z) with dimensions (bw,bh,bd)."""
    if x < 0 or y < 0 or z < 0:
        return False
    if x + bw > W or y + bh > H or z + bd > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        if _overlaps(x, y, z, bw, bh, bd, px, py, pz, pw, ph, pd):
            return False
    return True

def _compute_fill(placed, W, H, D):
    """Compute the fraction of container volume filled."""
    total_vol = W * H * D
    if total_vol == 0:
        return 0.0
    used_vol = sum(pw * ph * pd for (_, _, _, pw, ph, pd) in placed)
    return used_vol / total_vol

def _greedy_place(boxes, container, order_indices):
    """
    Greedy placement using a best-fit strategy with rotations.
    Returns list of placements and the fill fraction.
    """
    W, H, D = container
    placed = []  # list of (x, y, z, w, h, d)
    box_results = [None] * len(boxes)  # will store (x,y,z,w,h,d) or None
    
    # Sort boxes by order_indices (already sorted by some heuristic)
    for idx in order_indices:
        w, h, d = boxes[idx]
        rotations = _unique_rotations(w, h, d)
        
        best_pos = None
        best_score = None
        best_rot = None
        
        for (rw, rh, rd) in rotations:
            # Generate candidate positions: try to place at origin, then along edges
            # For efficiency, we'll try a limited set of candidate positions
            # Strategy: try to place as close to (0,0,0) as possible, preferring
            # positions that fill gaps
            
            # Candidate positions: try all combinations of "faces" of existing boxes
            # plus the origin
            candidate_positions = set()
            
            # Always try origin
            candidate_positions.add((0, 0, 0))
            
            # Try positions adjacent to each placed box
            for (px, py, pz, pw, ph, pd) in placed:
                # Place to the right, above, or in front of each box
                candidate_positions.add((px + pw, py, pz))
                candidate_positions.add((px, py + ph, pz))
                candidate_positions.add((px, py, pz + pd))
            
            # Also try some grid-based positions for better packing
            # Sample a few positions along the axes
            for x in [0, W - rw]:
                for y in [0, H - rh]:
                    for z in [0, D - rd]:
                        candidate_positions.add((x, y, z))
            
            # Limit candidate positions for performance
            if len(candidate_positions) > 200:
                # Keep only the most promising ones (closest to origin)
                candidate_positions = sorted(candidate_positions, key=lambda p: p[0] + p[1] + p[2])[:200]
            
            for (cx, cy, cz) in candidate_positions:
                if _can_place(cx, cy, cz, rw, rh, rd, placed, W, H, D):
                    # Score: prefer positions closer to origin (better packing)
                    score = cx + cy + cz
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pos = (cx, cy, cz)
                        best_rot = (rw, rh, rd)
        
        if best_pos is not None:
            x, y, z = best_pos
            rw, rh, rd = best_rot
            placed.append((x, y, z, rw, rh, rd))
            box_results[idx] = (x, y, z, rw, rh, rd)
    
    # Build the final placement list
    placements = []
    for i in range(len(boxes)):
        w, h, d = boxes[i]
        if box_results[i] is not None:
            x, y, z, rw, rh, rd = box_results[i]
            placements.append((0, x, y, z, rw, rh, rd))
        else:
            placements.append((-1, 0, 0, 0, w, h, d))
    
    fill = _compute_fill(placed, W, H, D)
    return placements, fill

def _random_order(boxes, rng):
    """Generate a random order of box indices."""
    indices = list(range(len(boxes)))
    rng.shuffle(indices)
    return indices

def _heuristic_order(boxes, container, strategy):
    """
    Generate a heuristic order of box indices.
    strategy: 0=largest volume first, 1=largest dimension first, 2=smallest first
    """
    W, H, D = container
    indices = list(range(len(boxes)))
    
    if strategy == 0:
        # Largest volume first
        indices.sort(key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)
    elif strategy == 1:
        # Largest max dimension first
        indices.sort(key=lambda i: max(boxes[i]), reverse=True)
    elif strategy == 2:
        # Smallest volume first
        indices.sort(key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2])
    elif strategy == 3:
        # Sort by aspect ratio (most cubic first)
        def aspect_ratio(i):
            w, h, d = boxes[i]
            max_dim = max(w, h, d)
            min_dim = min(w, h, d)
            return max_dim / max(min_dim, 1e-9)
        indices.sort(key=aspect_ratio)
    
    return indices

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # Deadline guard: 140 seconds (10s safety margin)
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try multiple strategies and keep the best result
    best_placements = None
    best_fill = 0.0
    
    # Strategy 1: Greedy with largest volume first
    order = _heuristic_order(boxes, container, 0)
    placements, fill = _greedy_place(boxes, container, order)
    if fill > best_fill:
        best_fill = fill
        best_placements = placements
    
    if time.time() > deadline:
        return best_placements
    
    # Strategy 2: Greedy with largest dimension first
    order = _heuristic_order(boxes, container, 1)
    placements, fill = _greedy_place(boxes, container, order)
    if fill > best_fill:
        best_fill = fill
        best_placements = placements
    
    if time.time() > deadline:
        return best_placements
    
    # Strategy 3: Greedy with smallest volume first
    order = _heuristic_order(boxes, container, 2)
    placements, fill = _greedy_place(boxes, container, order)
    if fill > best_fill:
        best_fill = fill
        best_placements = placements
    
    if time.time() > deadline:
        return best_placements
    
    # Strategy 4: Greedy with most cubic first
    order = _heuristic_order(boxes, container, 3)
    placements, fill = _greedy_place(boxes, container, order)
    if fill > best_fill:
        best_fill = fill
        best_placements = placements
    
    if time.time() > deadline:
        return best_placements
    
    # Strategy 5-10: Random orders with multiple restarts
    rng = random.Random(42)
    for _ in range(10):
        if time.time() > deadline:
            break
        order = _random_order(boxes, rng)
        placements, fill = _greedy_place(boxes, container, order)
        if fill > best_fill:
            best_fill = fill
            best_placements = placements
    
    return best_placements
# EVOLVE-BLOCK-END
