"""3D Container Loading with optimized multi-strategy search."""
# EVOLVE-BLOCK-START
import time
import itertools
import sys
import math

def _orientations(w, h, d):
    """Generate all unique orientations of a box."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _fits(box, pos, placed, W, H, D):
    """Check if box at pos doesn't overlap with placed boxes and fits in container."""
    x, y, z = pos
    bw, bh, bd = box
    if x + bw > W or y + bh > H or z + bd > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        # Check overlap: boxes overlap if they intersect in all 3 axes
        if not (x + bw <= px or px + pw <= x or
                y + bh <= py or py + ph <= y or
                z + bd <= pz or pz + pd <= z):
            return False
    return True

def _candidate_positions(placed, W, H, D):
    """Generate candidate positions based on corners of placed boxes and container floor."""
    candidates = set()
    # Bottom corners of container
    candidates.add((0, 0, 0))
    # For each placed box, generate 3 candidate positions adjacent to it
    for (px, py, pz, pw, ph, pd) in placed:
        candidates.add((px + pw, py, pz))
        candidates.add((px, py + ph, pz))
        candidates.add((px, py, pz + pd))
    # Filter to only valid positions within container
    valid = []
    for (x, y, z) in candidates:
        if 0 <= x < W and 0 <= y < H and 0 <= z < D:
            valid.append((x, y, z))
    return valid

def _select_candidates(placed, W, H, D, max_candidates=300):
    """Select candidate positions, prioritizing floor-level (low z) positions."""
    candidates = _candidate_positions(placed, W, H, D)
    if len(candidates) <= max_candidates:
        return candidates
    # Sort: primary by z (floor first), secondary by x+y
    candidates.sort(key=lambda p: (p[2], p[0] + p[1]))
    return candidates[:max_candidates]

def _score_placement(placed, W, H, D):
    """Score a placement by minimizing the sum of corner coordinates (prefer lower/tighter)."""
    if not placed:
        return 0
    return sum(p[0] + p[1] + p[2] for p in placed)

def _score_max_extent(placed, W, H, D):
    """Score by minimizing the max extent of all placed boxes (bounding box volume proxy)."""
    if not placed:
        return 0
    max_x = max(p[0] for p in placed)
    max_y = max(p[1] for p in placed)
    max_z = max(p[2] for p in placed)
    return max_x + max_y + max_z

def _best_fit_decreasing(boxes, container, deadline, max_candidates=300, score_func=None):
    """Best-Fit-Decreasing with corner-based candidate generation."""
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    if score_func is None:
        _sc = _score_placement
    elif score_func == 'max_extent':
        _sc = _score_max_extent
    else:
        _sc = _score_placement
    
    # Sort boxes by volume descending
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    placed = []  # list of (x, y, z, w, h, d)
    placement_map = {}  # box_index -> (x, y, z, w, h, d)
    
    for i, w, h, d in indexed_boxes:
        if time.time() > deadline:
            break
        
        # Try all orientations
        orients = _orientations(w, h, d)
        best_pos = None
        best_score = float('inf')
        best_orient = None
        
        # Generate and select candidate positions
        candidates = _select_candidates(placed, W, H, D, max_candidates)
        
        for orient in orients:
            for pos in candidates:
                if _fits(orient, pos, placed, W, H, D):
                    score = _sc([pos], W, H, D)
                    if score < best_score:
                        best_score = score
                        best_pos = pos
                        best_orient = orient
        
        if best_pos is not None:
            x, y, z = best_pos
            bw, bh, bd = best_orient
            placed.append((x, y, z, bw, bh, bd))
            placement_map[i] = (x, y, z, bw, bh, bd)
    
    # Build result
    result = []
    for i, (w, h, d) in enumerate(boxes):
        if i in placement_map:
            x, y, z, bw, bh, bd = placement_map[i]
            result.append((0, x, y, z, bw, bh, bd))
        else:
            result.append((-1, 0, 0, 0, w, h, d))
    return result

def _shelf_packing(boxes, container):
    """Simple shelf packing as fallback."""
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Sort by height descending
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[2], reverse=True)
    
    placements = []
    placement_map = {}
    x, y, z, row_h, layer_d = 0, 0, 0, 0, 0
    
    for i, w, h, d in indexed_boxes:
        if x + w > W:
            x, y, row_h = 0, y + row_h, 0
        if y + h > H:
            y, z, layer_d, x, row_h = 0, z + layer_d, 0, 0, 0
        if z + d > D:
            continue
        placements.append((x, y, z, w, h, d))
        placement_map[i] = (x, y, z, w, h, d)
        x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
    
    result = []
    for i, (w, h, d) in enumerate(boxes):
        if i in placement_map:
            x, y, z, bw, bh, bd = placement_map[i]
            result.append((0, x, y, z, bw, bh, bd))
        else:
            result.append((-1, 0, 0, 0, w, h, d))
    return result

def _volume_filling(boxes, container):
    """Calculate volume fraction filled."""
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return 0.0
    container_vol = W * H * D
    if container_vol == 0:
        return 0.0
    filled = 0
    for (bin_id, x, y, z, w, h, d) in boxes:
        if bin_id == 0:
            filled += w * h * d
    return filled / container_vol

def _improve_placement(result, boxes, container, deadline, max_iters=50):
    """Local search: try to swap or reposition boxes to improve packing."""
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return result
    
    # Extract placed boxes
    placed = []
    placement_map = {}
    for i, (bin_id, x, y, z, w, h, d) in enumerate(result):
        if bin_id == 0:
            placed.append((x, y, z, w, h, d))
            placement_map[i] = (x, y, z, w, h, d)
    
    # Try to improve by moving boxes to better positions
    improved = False
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        best_gain = 0
        best_swap = None
        
        for i in list(placement_map.keys()):
            if time.time() > deadline:
                break
            x, y, z, w, h, d = placement_map[i]
            # Remove box temporarily
            remaining = [p for p in placed if p != (x, y, z, w, h, d)]
            del placement_map[i]
            placed = remaining
            
            # Try to find a better position
            orients = _orientations(w, h, d)
            candidates = _select_candidates(placed, W, H, D, max_candidates=200)
            
            best_pos = None
            best_score = float('inf')
            best_orient = None
            
            for orient in orients:
                for pos in candidates:
                    if _fits(orient, pos, placed, W, H, D):
                        score = _score_placement([pos], W, H, D)
                        if score < best_score:
                            best_score = score
                            best_pos = pos
                            best_orient = orient
            
            if best_pos is not None:
                # Check if this is an improvement
                old_score = _score_placement([(x, y, z)], W, H, D)
                if best_score < old_score - 1e-9:
                    gain = old_score - best_score
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = (i, best_pos, best_orient)
            
            # Restore box
            placement_map[i] = (x, y, z, w, h, d)
            placed = list(remaining) + [(x, y, z, w, h, d)]
        
        if best_swap is not None:
            i, new_pos, new_orient = best_swap
            x, y, z, w, h, d = placement_map[i]
            # Remove old position
            placed = [p for p in placed if p != (x, y, z, w, h, d)]
            # Add new position
            bx, by, bz = new_pos
            bw, bh, bd = new_orient
            placed.append((bx, by, bz, bw, bh, bd))
            placement_map[i] = (bx, by, bz, bw, bh, bd)
            improved = True
        else:
            break
    
    # Build result
    result = []
    for i, (w, h, d) in enumerate(boxes):
        if i in placement_map:
            x, y, z, bw, bh, bd = placement_map[i]
            result.append((0, x, y, z, bw, bh, bd))
        else:
            result.append((-1, 0, 0, 0, w, h, d))
    return result

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second margin before 120s limit
    
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Strategy 1: Best-Fit-Decreasing (sum-of-coords scoring)
    result1 = _best_fit_decreasing(boxes, container, deadline, max_candidates=300)
    score1 = _volume_filling(result1, container)
    
    # Strategy 2: Best-Fit-Decreasing (max-extent scoring) - tighter bounding box
    deadline2 = start_time + 110.0
    if time.time() < deadline2 - 5.0:
        result2 = _best_fit_decreasing(boxes, container, deadline2, max_candidates=300, score_func='max_extent')
        score2 = _volume_filling(result2, container)
    else:
        result2 = result1
        score2 = score1
    
    # Strategy 3: Shelf packing (fast, may be good for some cases)
    result3 = _shelf_packing(boxes, container)
    score3 = _volume_filling(result3, container)
    
    # Pick the best initial result
    best = max([(score1, result1), (score2, result2), (score3, result3)], key=lambda t: t[0])
    best_score = best[0]
    best_result = best[1]
    
    # Try to improve with local search if time allows
    if time.time() < deadline - 10.0:
        improved = _improve_placement(best_result, boxes, container, deadline, max_iters=30)
        improved_score = _volume_filling(improved, container)
        if improved_score > best_score:
            best_result = improved
            best_score = improved_score
    
    return best_result
# EVOLVE-BLOCK-END
