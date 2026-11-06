"""Rewritten 3D container loading program."""
import time
import math
import random
from itertools import permutations

def _get_rotations(w, h, d):
    """Get all unique rotations of a box, sorted by preferred orientation (lower z first)."""
    rots = set()
    for perm in permutations((w, h, d)):
        rots.add(perm)
    rots = sorted(rots, key=lambda r: (r[2], r[1], r[0]))
    return list(rots)

def _check_overlap(cx, cy, cz, rw, rh, rd, placed):
    """Check if a box at (cx,cy,cz) with size (rw,rh,rd) overlaps any placed box."""
    for px, py, pz, pw, ph, pd in placed:
        if not (cx + rw <= px or px + pw <= cx or
                cy + rh <= py or py + ph <= cy or
                cz + rd <= pz or pz + pd <= cz):
            return True
    return False

def _generate_candidates(placed, W, H, D, max_candidates=800):
    """
    Generate candidate positions from placed boxes using extreme points.
    Returns a list of candidate (x, y, z) positions.
    """
    candidates = set()
    candidates.add((0, 0, 0))
    
    for px, py, pz, pw, ph, pd in placed:
        new_pts = [
            (px + pw, py, pz),
            (px, py + ph, pz),
            (px, py, pz + pd),
            (px + pw, py + ph, pz),
            (px + pw, py, pz + pd),
            (px, py + ph, pz + pd),
            (px + pw, py + ph, pz + pd),
        ]
        for pt in new_pts:
            if 0 <= pt[0] <= W and 0 <= pt[1] <= H and 0 <= pt[2] <= D:
                candidates.add(pt)
    
    cand_list = list(candidates)
    if len(cand_list) > max_candidates:
        # Sort by priority: lower z first, then lower x, then lower y
        cand_list.sort(key=lambda c: (c[2], c[0], c[1]))
        cand_list = cand_list[:max_candidates]
    
    return cand_list

def _try_place(boxes, container, sort_key='volume', seed=0):
    """
    Attempt to place boxes into the container using an Extreme Point heuristic.
    Returns a list of placements: (bin, x, y, z, w, h, d) for each box.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    # Precompute rotations for each box
    box_rotations = []
    for w, h, d in boxes:
        rots = _get_rotations(w, h, d)
        box_rotations.append(rots)
    
    # Sort boxes based on the chosen key
    box_indices = list(range(n))
    box_volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    
    if sort_key == 'volume':
        box_indices.sort(key=lambda i: box_volumes[i], reverse=True)
    elif sort_key == 'aspect_ratio':
        def aspect(i):
            w, h, d = boxes[i]
            dims = sorted([w, h, d])
            return dims[2] / max(dims[0], 1)
        box_indices.sort(key=aspect)
    elif sort_key == 'surface_area':
        def surf(i):
            w, h, d = boxes[i]
            return 2 * (w*h + h*d + w*d)
        box_indices.sort(key=surf, reverse=True)
    elif sort_key == 'max_dim':
        def maxd(i):
            w, h, d = boxes[i]
            return max(w, h, d)
        box_indices.sort(key=maxd, reverse=True)
    elif sort_key == 'min_dim':
        def mind(i):
            w, h, d = boxes[i]
            return min(w, h, d)
        box_indices.sort(key=mind, reverse=True)
    elif sort_key == 'random':
        random.seed(seed)
        random.shuffle(box_indices)
    else:
        box_indices.sort(key=lambda i: box_volumes[i], reverse=True)
    
    # Placements: list of (x, y, z, w, h, d) for placed boxes
    placed = []
    # Result: for each original box index, store placement or None
    result = [None] * n
    
    for idx in box_indices:
        w_orig, h_orig, d_orig = boxes[idx]
        rots = box_rotations[idx]
        
        # Generate candidate positions
        candidates = _generate_candidates(placed, W, H, D)
        
        best_pos = None
        best_rot = None
        best_score = None
        
        # Try each rotation
        for rw, rh, rd in rots:
            # Try each candidate position
            for cx, cy, cz in candidates:
                if cx + rw > W or cy + rh > H or cz + rd > D:
                    continue
                
                # Check overlap with all placed boxes
                if _check_overlap(cx, cy, cz, rw, rh, rd, placed):
                    continue
                
                # Score: strongly prefer lower z (stacking), then lower x, then lower y
                # Use tuple comparison for lexicographic ordering
                score_tuple = (cz, cx, cy)
                if best_score is None or score_tuple < best_score:
                    best_score = score_tuple
                    best_pos = (cx, cy, cz)
                    best_rot = (rw, rh, rd)
        
        if best_pos is not None:
            bx, by, bz = best_pos
            bw, bh, bd = best_rot
            placed.append((bx, by, bz, bw, bh, bd))
            result[idx] = (0, bx, by, bz, bw, bh, bd)
        else:
            result[idx] = (-1, 0, 0, 0, w_orig, h_orig, d_orig)
        
        # Check time
        if time.time() > deadline:
            # Fill remaining with -1
            for i in range(n):
                if result[i] is None:
                    w_i, h_i, d_i = boxes[i]
                    result[i] = (-1, 0, 0, 0, w_i, h_i, d_i)
            break
    
    # Fill any remaining None entries
    for i in range(n):
        if result[i] is None:
            w_i, h_i, d_i = boxes[i]
            result[i] = (-1, 0, 0, 0, w_i, h_i, d_i)
    
    return result

def _try_place_diverse(boxes, container):
    """
    Try multiple sorting strategies and return the best result.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Define different sorting strategies
    strategies = ['volume', 'aspect_ratio', 'surface_area', 'max_dim', 'min_dim']
    
    best_result = None
    best_filled = 0
    
    for strat_idx, strategy in enumerate(strategies):
        if time.time() > deadline:
            break
        
        # Place boxes using the extreme point heuristic
        result = _try_place(boxes, container, sort_key=strategy)
        
        # Calculate filled volume
        filled = 0
        for r in result:
            if r[0] == 0:
                filled += r[4] * r[5] * r[6]
        
        if filled > best_filled:
            best_filled = filled
            best_result = result
    
    if best_result is None:
        # Fallback: all boxes left out
        best_result = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    
    return best_result

def _try_place_with_shake(boxes, container, seed=0):
    """
    Try to improve a placement by shaking (removing and re-placing) boxes.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Get initial placement
    result = _try_place(boxes, container, sort_key='volume')
    
    # Calculate initial filled volume
    filled = sum(r[4] * r[5] * r[6] for r in result if r[0] == 0)
    
    # Build placed list from result
    placed = []
    box_to_placed_idx = {}
    for i, r in enumerate(result):
        if r[0] == 0:
            placed.append((r[1], r[2], r[3], r[4], r[5], r[6]))
            box_to_placed_idx[i] = len(placed) - 1
    
    # Precompute rotations
    box_rotations = []
    for w, h, d in boxes:
        rots = _get_rotations(w, h, d)
        box_rotations.append(rots)
    
    # Try to shake: remove a box and try to re-place it in a different position
    improved = True
    max_iterations = 20
    
    while improved and time.time() < deadline:
        improved = False
        
        # Try each placed box in reverse order (larger boxes first)
        for i in range(n - 1, -1, -1):
            if time.time() > deadline:
                break
            
            if result[i][0] != 0:
                continue
            
            # Remove this box from placed
            placed_idx = box_to_placed_idx.get(i)
            if placed_idx is None:
                continue
            
            bx, by, bz, bw, bh, bd = placed[placed_idx]
            placed.pop(placed_idx)
            
            # Update box_to_placed_idx for remaining boxes
            for k in list(box_to_placed_idx.keys()):
                if box_to_placed_idx[k] > placed_idx:
                    box_to_placed_idx[k] -= 1
            del box_to_placed_idx[i]
            
            # Try to re-place the box in a different position
            w_orig, h_orig, d_orig = boxes[i]
            rots = box_rotations[i]
            
            candidates = _generate_candidates(placed, W, H, D)
            
            best_pos = None
            best_rot = None
            best_score = None
            
            for rw, rh, rd in rots:
                for cx, cy, cz in candidates:
                    if cx + rw > W or cy + rh > H or cz + rd > D:
                        continue
                    
                    if _check_overlap(cx, cy, cz, rw, rh, rd, placed):
                        continue
                    
                    # Prefer different position than original
                    score_tuple = (cz, cx, cy)
                    if (cx, cy, cz, rw, rh, rd) == (bx, by, bz, bw, bh, bd):
                        score_tuple = (cz, cx, cy, 1)  # Penalize same position
                    
                    if best_score is None or score_tuple < best_score:
                        best_score = score_tuple
                        best_pos = (cx, cy, cz)
                        best_rot = (rw, rh, rd)
            
            if best_pos is not None:
                bx2, by2, bz2 = best_pos
                bw2, bh2, bd2 = best_rot
                placed.append((bx2, by2, bz2, bw2, bh2, bd2))
                box_to_placed_idx[i] = len(placed) - 1
                result[i] = (0, bx2, by2, bz2, bw2, bh2, bd2)
                
                # Check if this led to improvement
                new_filled = sum(r[4] * r[5] * r[6] for r in result if r[0] == 0)
                if new_filled > filled:
                    improved = True
                    filled = new_filled
            else:
                # Put back in original position
                placed.append((bx, by, bz, bw, bh, bd))
                box_to_placed_idx[i] = len(placed) - 1
                result[i] = (0, bx, by, bz, bw, bh, bd)
        
        if not improved:
            break
    
    return result

def load(boxes, container):
    """
    Place boxes into container. Returns list of (bin, x, y, z, w, h, d) for each box.
    bin=0 for loaded, bin=-1 for left out.
    """
    W, H, D = container
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin before 150s limit
    
    # Try the diverse strategy which explores multiple sorting orders
    placements = _try_place_diverse(boxes, container)
    
    # If we have time, try shaking to improve
    if time.time() < deadline - 5.0:
        improved = _try_place_with_shake(boxes, container)
        
        # Compare and use the better result
        filled_diverse = sum(r[4] * r[5] * r[6] for r in placements if r[0] == 0)
        filled_shake = sum(r[4] * r[5] * r[6] for r in improved if r[0] == 0)
        
        if filled_shake > filled_diverse:
            placements = improved
    
    return placements
