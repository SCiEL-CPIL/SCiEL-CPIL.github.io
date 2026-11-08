import time
import itertools
import random

def load(boxes, container):
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin
    
    W, H, D = container
    n = len(boxes)
    container_vol = W * H * D
    if container_vol == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    # Pre-compute all possible orientations for each box
    # We assume the box dimensions are (w, h, d)
    # Orientations are permutations of (w, h, d)
    box_orientations = []
    for b in boxes:
        ws, hs, ds = set(itertools.permutations(b))
        box_orientations.append(list(ws))

    def can_place(x, y, z, w, h, d, placed_boxes):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in placed_boxes:
            # Check overlap
            if x < px + pw and px < x + w and \
               y < py + ph and py < y + h and \
               z < pz + pd and pz < z + d:
                return False
        return True

    def score_position(x, y, z, w, h, d):
        # We want to pack tight.
        # Lower is better.
        # Distance to walls: (W - (x+w)) + (H - (y+h)) + (D - (z+d))
        # Also prefer lower z (gravity), then lower y, then lower x
        # Let's use a weighted sum of distances to the "origin" (0,0,0) and "back" walls
        # Actually, standard Best Fit Decreasing often minimizes the sum of distances to the walls
        # dist_x = W - (x + w)
        # dist_y = H - (y + h)
        # dist_z = D - (z + d)
        # However, we also want to fill from bottom-up.
        # So, minimize z first, then y, then x?
        # Or minimize the "bounding box" expansion?
        # Let's try: minimize (z, y, x) lexicographically? No, that might leave gaps.
        # Let's try: minimize (x + W/2, y + H/2, z + D/2) ? No.
        # Common heuristic: minimize the sum of distances to the nearest walls in each dimension.
        # But for container loading, "bottom-left-front" is usually good.
        # Let's score by: z * 1000 + y * 100 + x
        # This strongly prefers lower z, then lower y, then lower x.
        return z * 10000 + y * 100 + x

    def get_candidate_positions(placed_boxes):
        # Candidate positions are:
        # 1. (0, 0, 0)
        # 2. For each placed box, its "corners" that are free:
        #    (x + w, y, z), (x, y + h, z), (x, y, z + d)
        #    and combinations? (x+w, y+h, z), etc.
        # To be efficient, we just check the immediate neighbors.
        candidates = {(0, 0, 0)}
        for (px, py, pz, pw, ph, pd) in placed_boxes:
            candidates.add((px + pw, py, pz))
            candidates.add((px, py + ph, pz))
            candidates.add((px, py, pz + pd))
            candidates.add((px + pw, py + ph, pz))
            candidates.add((px + pw, py, pz + pd))
            candidates.add((px, py + ph, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
        return list(candidates)

    def try_packing(box_indices, orientations_map):
        placed_boxes = []
        result = [-1] * n
        # box_indices is a list of indices in original order
        # We need to map back to original indices for result
        
        # We will iterate through boxes in the order given by box_indices
        # But we need to store which original box was placed where.
        # Let's keep a list of (orig_idx, x, y, z, w, h, d)
        
        for orig_idx in box_indices:
            if time.time() - start_time > DEADLINE:
                return None, 0.0
            
            placed = False
            # Try all orientations for this box
            # Sort orientations by some metric? e.g., volume is same, but shape matters.
            # Let's just try all 6 permutations (or unique ones)
            best_pos = None
            best_score = float('inf')
            
            # Get candidate positions based on current placed boxes
            # Note: The set of candidates depends on placed_boxes, which changes as we place.
            # We generate candidates once per box iteration.
            candidates = get_candidate_positions(placed_boxes)
            
            # Also, it's possible that a position not in candidates is valid?
            # Yes, but usually in 3D packing, the optimal position is "touching" existing boxes or walls.
            # So checking these candidates is a standard heuristic.
            
            # To be safe, if no candidate works, we might miss a spot.
            # But for speed, we stick to candidates.
            
            for orient in orientations_map[orig_idx]:
                w, h, d = orient
                for (cx, cy, cz) in candidates:
                    if can_place(cx, cy, cz, w, h, d, placed_boxes):
                        s = score_position(cx, cy, cz, w, h, d)
                        if s < best_score:
                            best_score = s
                            best_pos = (cx, cy, cz, w, h, d)
            
            if best_pos is not None:
                x, y, z, w, h, d = best_pos
                placed_boxes.append((x, y, z, w, h, d))
                result[orig_idx] = (0, x, y, z, w, h, d)
            else:
                # Box cannot be placed
                result[orig_idx] = (-1, 0, 0, 0, boxes[orig_idx][0], boxes[orig_idx][1], boxes[orig_idx][2])
        
        # Calculate score
        filled_vol = 0
        for i in range(n):
            if result[i][0] == 0:
                _, x, y, z, w, h, d = result[i]
                filled_vol += w * h * d
        
        return result, filled_vol / container_vol

    # Strategy 1: Volume Descending
    indices_by_vol = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    # Strategy 2: Max Dimension Descending
    indices_by_max_dim = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    
    # Strategy 3: Aspect Ratio (Volume / Max Dim) or similar
    # Let's try sorting by the smallest dimension? Or largest?
    # Let's try a random shuffle with a few fixed seeds to break ties or explore different paths
    # But deterministic is better for reproducibility if possible, though random is allowed.
    
    best_result = None
    best_score = 0.0
    
    # Run Strategy 1
    res, sc = try_packing(indices_by_vol, box_orientations)
    if res is not None and sc > best_score:
        best_score = sc
        best_result = res
        
    if time.time() - start_time > DEADLINE:
        if best_result is None:
             # Fallback to simple row packing if we timed out before finishing even one
             # But try_packing returns None only on timeout.
             # If it returned a result, it's valid.
             pass
        return best_result if best_result is not None else [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    # Run Strategy 2
    res, sc = try_packing(indices_by_max_dim, box_orientations)
    if res is not None and sc > best_score:
        best_score = sc
        best_result = res
        
    if time.time() - start_time > DEADLINE:
        return best_result

    # Strategy 3: Randomized Restart with Volume Descending as base but with jitter?
    # Or just another sort: e.g., by (w+h+d)
    indices_by_sum = sorted(range(n), key=lambda i: sum(boxes[i]), reverse=True)
    res, sc = try_packing(indices_by_sum, box_orientations)
    if res is not None and sc > best_score:
        best_score = sc
        best_result = res

    if time.time() - start_time > DEADLINE:
        return best_result

    # If we still have time, try a few random shuffles
    # This can help escape local minima of the greedy approach
    for _ in range(10):
        if time.time() - start_time > DEADLINE:
            break
        random_indices = list(range(n))
        random.shuffle(random_indices)
        res, sc = try_packing(random_indices, box_orientations)
        if res is not None and sc > best_score:
            best_score = sc
            best_result = res

    if best_result is None:
        # Fallback: Return all -1
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    return best_result
