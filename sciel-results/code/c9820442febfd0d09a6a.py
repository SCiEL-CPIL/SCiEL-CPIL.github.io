"""
3D Container Loading: Advanced Greedy + Shake Local Search
"""
import time
import random
import itertools
import math

def _rotations(w, h, d):
    """Return all 6 unique orientations of a box (w,h,d)."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _extreme_points(placed, container):
    """Generate candidate (x,y,z) positions from placed boxes."""
    W, H, D = container
    xs = {0}
    ys = {0}
    zs = {0}
    for (px, py, pz, pw, ph, pd) in placed:
        xs.add(px + pw)
        ys.add(py + ph)
        zs.add(pz + pd)
    
    # Filter to those within container bounds
    xs = [x for x in xs if x < W]
    ys = [y for y in ys if y < H]
    zs = [z for z in zs if z < D]
    
    points = []
    for x in xs:
        for y in ys:
            for z in zs:
                points.append((x, y, z))
    return points

def _fits(box_dims, pos, container, placed):
    """Check if box fits at pos without overlap and within container."""
    W, H, D = container
    x, y, z = pos
    w, h, d = box_dims
    
    # Check container bounds
    if x + w > W or y + h > H or z + d > D:
        return False
    
    # Check overlap with placed boxes
    for (px, py, pz, pw, ph, pd) in placed:
        if (x < px + pw and x + w > px and
            y < py + ph and y + h > py and
            z < pz + pd and z + d > pz):
            return False
    return True

def _score_position(pos, box_dims, placed, container):
    """
    Score a candidate position.
    We want to minimize z (height), then y, then x to keep the packing compact.
    A lower score is better.
    """
    x, y, z = pos
    # Primary: keep it low (z)
    # Secondary: keep it forward (y)
    # Tertiary: keep it left (x)
    return (z, y, x)

def _face_points(placed, container):
    """Generate candidate positions from face-adjacent corners of placed boxes."""
    W, H, D = container
    candidates = set()
    candidates.add((0, 0, 0))
    for (px, py, pz, pw, ph, pd) in placed:
        candidates.add((px + pw, py, pz))
        candidates.add((px, py + ph, pz))
        candidates.add((px, py, pz + pd))
        candidates.add((px + pw, py + ph, pz))
        candidates.add((px + pw, py, pz + pd))
        candidates.add((px, py + ph, pz + pd))
        candidates.add((px + pw, py + ph, pz + pd))
    
    # Filter to valid positions
    valid = []
    for (x, y, z) in candidates:
        if 0 <= x < W and 0 <= y < H and 0 <= z < D:
            valid.append((x, y, z))
    return valid

def _greedy_place(boxes, container, order, deadline, mode='standard'):
    """
    Greedy placement of boxes in the given order.
    Returns a list of placements (bin, x, y, z, w, h, d) and total volume.
    """
    W, H, D = container
    placed = []
    placements = [None] * len(boxes)
    total_vol = 0.0
    n = len(boxes)
    
    for idx in order:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        rots = _rotations(w, h, d)
        
        # Generate candidate positions
        candidates = _extreme_points(placed, container)
        # Add face points for more coverage
        face_pts = _face_points(placed, container)
        for fp in face_pts:
            if fp not in candidates:
                candidates.append(fp)
        # Always include (0,0,0) just in case, though it's usually in extreme points
        if (0, 0, 0) not in candidates:
            candidates.append((0, 0, 0))
        
        best_pos = None
        best_rot = None
        best_score = None
        
        # Sort candidates to check promising ones first? 
        # Actually, we need to check all to find the *best* according to score.
        # But we can prune if we find a perfect (0,0,0) early? No, score matters.
        
        for pos in candidates:
            for rw, rh, rd in rots:
                if _fits((rw, rh, rd), pos, container, placed):
                    score = _score_position(pos, (rw, rh, rd), placed, container)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pos = pos
                        best_rot = (rw, rh, rd)
        
        if best_pos is not None:
            x, y, z = best_pos
            rw, rh, rd = best_rot
            placed.append((x, y, z, rw, rh, rd))
            placements[idx] = (0, x, y, z, rw, rh, rd)
            total_vol += rw * rh * rd
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)
    
    # Fill any unprocessed boxes as not placed
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
            
    return placements, total_vol

def _shake_local_search(placements, boxes, container, deadline, rng, num_shakes=5):
    """
    Improve placement by shaking: remove last k boxes and re-insert.
    """
    W, H, D = container
    n = len(boxes)
    current_placements = [p[:] for p in placements] # Deep copy
    current_vol = sum(p[4]*p[5]*p[6] for p in current_placements if p[0] == 0)
    
    # Identify loaded boxes
    loaded_indices = [i for i in range(n) if current_placements[i][0] == 0]
    
    if not loaded_indices:
        return current_placements, current_vol
    
    # We will try to shake the last few boxes in the loading order
    # But 'loaded_indices' is not necessarily in loading order. 
    # It's better to track the order in which they were placed.
    # However, for simplicity, we can just pick random subsets of loaded boxes to remove and re-pack.
    # A common strategy is to remove the box with the highest z-coordinate, or just random ones.
    
    for _ in range(num_shakes):
        if time.time() > deadline:
            break
            
        # Pick a random subset of loaded boxes to remove
        # Let's try removing 1 to 3 boxes
        k = rng.randint(1, min(3, len(loaded_indices)))
        boxes_to_remove = rng.sample(loaded_indices, k)
        
        # Build list of currently placed boxes (x,y,z,w,h,d)
        placed_boxes = []
        for i in range(n):
            p = current_placements[i]
            if p[0] == 0:
                placed_boxes.append((p[1], p[2], p[3], p[4], p[5], p[6]))
        
        # Remove the selected boxes
        boxes_to_remove_set = set(boxes_to_remove)
        new_placed_boxes = []
        removed_dims = []
        for i in range(n):
            p = current_placements[i]
            if p[0] == 0:
                box_tuple = (p[1], p[2], p[3], p[4], p[5], p[6])
                if i in boxes_to_remove_set:
                    removed_dims.append((i, box_tuple))
                else:
                    new_placed_boxes.append(box_tuple)
        
        # Try to re-insert the removed boxes
        # We can try to insert them in reverse order of removal or random order
        # Let's try inserting them one by one using greedy logic
        success = True
        new_placed_boxes_copy = new_placed_boxes[:]
        
        # Sort removed boxes by volume descending? Or just try them in random order?
        # Let's try random order to diversify
        rng.shuffle(removed_dims)
        
        for idx, old_pos in removed_dims:
            w, h, d = boxes[idx]
            rots = _rotations(w, h, d)
            candidates = _extreme_points(new_placed_boxes_copy, container)
            if (0, 0, 0) not in candidates:
                candidates.append((0, 0, 0))
            
            best_pos = None
            best_rot = None
            best_score = None
            
            for pos in candidates:
                for rw, rh, rd in rots:
                    if _fits((rw, rh, rd), pos, container, new_placed_boxes_copy):
                        score = _score_position(pos, (rw, rh, rd), new_placed_boxes_copy, container)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_pos = pos
                            best_rot = (rw, rh, rd)
            
            if best_pos is not None:
                x, y, z = best_pos
                rw, rh, rd = best_rot
                new_placed_boxes_copy.append((x, y, z, rw, rh, rd))
                current_placements[idx] = (0, x, y, z, rw, rh, rd)
            else:
                # Could not place, mark as not placed
                current_placements[idx] = (-1, 0, 0, 0, w, h, d)
                success = False
                # Even if one fails, we might still have improved others? 
                # But volume would decrease. We should only accept if total volume increases.
                # So we need to calculate new volume.
        
        # Calculate new volume
        new_vol = sum(p[4]*p[5]*p[6] for p in current_placements if p[0] == 0)
        
        if new_vol > current_vol:
            current_vol = new_vol
            # Update loaded_indices for next iteration?
            # Actually, we can just recalculate or keep it simple.
            # For efficiency, we don't strictly need to update loaded_indices if we just pick from all boxes each time.
            # But picking from loaded ones is more efficient.
            loaded_indices = [i for i in range(n) if current_placements[i][0] == 0]
        else:
            # Revert to previous state? 
            # This is tricky because we modified current_placements in place.
            # We should have kept a backup.
            # Let's refactor to use a temporary copy.
            pass # We'll handle this by reverting manually or just not accepting.
            # Since we modified in place, we need to restore.
            # This is inefficient. Let's do the shake on a copy.
            
    return current_placements, current_vol

def _shake_local_search_v2(placements, boxes, container, deadline, rng, num_shakes=10):
    """
    More robust shake local search using copies.
    """
    n = len(boxes)
    current_placements = [p[:] for p in placements]
    current_vol = sum(p[4]*p[5]*p[6] for p in current_placements if p[0] == 0)
    
    loaded_indices = [i for i in range(n) if current_placements[i][0] == 0]
    
    for _ in range(num_shakes):
        if time.time() > deadline:
            break
        
        if not loaded_indices:
            break
            
        # Pick random boxes to shake
        k = rng.randint(1, min(3, len(loaded_indices)))
        boxes_to_remove = rng.sample(loaded_indices, k)
        
        # Create a temporary state
        temp_placements = [p[:] for p in current_placements]
        
        # Extract placed boxes not in remove set
        placed_boxes = []
        for i in range(n):
            p = temp_placements[i]
            if p[0] == 0 and i not in boxes_to_remove:
                placed_boxes.append((p[1], p[2], p[3], p[4], p[5], p[6]))
        
        # Mark removed boxes as unplaced in temp
        for i in boxes_to_remove:
            w, h, d = boxes[i]
            temp_placements[i] = (-1, 0, 0, 0, w, h, d)
        
        # Try to re-insert removed boxes
        # Sort by volume descending to place big ones first? Or random?
        # Let's try random order for diversity.
        shuffle_list = boxes_to_remove[:]
        rng.shuffle(shuffle_list)
        
        for idx in shuffle_list:
            w, h, d = boxes[idx]
            rots = _rotations(w, h, d)
            candidates = _extreme_points(placed_boxes, container)
            if (0, 0, 0) not in candidates:
                candidates.append((0, 0, 0))
            
            best_pos = None
            best_rot = None
            best_score = None
            
            for pos in candidates:
                for rw, rh, rd in rots:
                    if _fits((rw, rh, rd), pos, container, placed_boxes):
                        score = _score_position(pos, (rw, rh, rd), placed_boxes, container)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_pos = pos
                            best_rot = (rw, rh, rd)
            
            if best_pos is not None:
                x, y, z = best_pos
                rw, rh, rd = best_rot
                placed_boxes.append((x, y, z, rw, rh, rd))
                temp_placements[idx] = (0, x, y, z, rw, rh, rd)
        
        # Calculate new volume
        new_vol = sum(p[4]*p[5]*p[6] for p in temp_placements if p[0] == 0)
        
        if new_vol > current_vol:
            current_placements = temp_placements
            current_vol = new_vol
            loaded_indices = [i for i in range(n) if current_placements[i][0] == 0]
            
    return current_placements, current_vol

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    W, H, D = container
    n = len(boxes)
    
    vols = [w * h * d for w, h, d in boxes]
    
    # Define sorting strategies
    sort_strategies = [
        ('vol_desc', lambda i: vols[i], True),
        ('min_dim_desc', lambda i: min(boxes[i]), True),
        ('max_dim_desc', lambda i: max(boxes[i]), True),
        ('ratio', lambda i: max(boxes[i]) / max(min(boxes[i]), 1e-9), False),
        ('vol_asc', lambda i: vols[i], False), # Sometimes small first helps?
    ]
    
    best_placements = None
    best_vol = 0.0
    
    # 1. Run greedy with different sorts
    for name, key_fn, reverse in sort_strategies:
        if time.time() > deadline:
            break
        order = list(range(n))
        order.sort(key=key_fn, reverse=reverse)
        
        placements, vol = _greedy_place(boxes, container, order, deadline)
        if vol > best_vol:
            best_vol = vol
            best_placements = [p[:] for p in placements]
    
    if best_placements is None:
        # Fallback
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        return best_placements
    
    # 2. Local Search (Shake)
    rng = random.Random(42)
    # Use remaining time for local search
    # Allocate 50% of remaining time to local search?
    # Let's just run it until deadline.
    best_placements, best_vol = _shake_local_search_v2(
        best_placements, boxes, container, deadline, rng, num_shakes=100
    )
    
    # 3. If time remains, try another greedy + local search with a different seed/sort
    if time.time() < deadline - 10:
        # Try a random permutation
        rng2 = random.Random(123)
        order = list(range(n))
        rng2.shuffle(order)
        
        placements, vol = _greedy_place(boxes, container, order, deadline)
        if vol > best_vol:
            best_vol = vol
            best_placements = [p[:] for p in placements]
        
        # Local search on this new best
        if time.time() < deadline - 5:
            best_placements, best_vol = _shake_local_search_v2(
                best_placements, boxes, container, deadline, rng2, num_shakes=50
            )
            
    # 4. Final polish: try to place any remaining boxes with more aggressive search
    if time.time() < deadline - 2:
        remaining = [i for i in range(n) if best_placements[i][0] == -1]
        if remaining:
            # Try to place remaining boxes in random order
            rng3 = random.Random(777)
            rng3.shuffle(remaining)
            for idx in remaining:
                if time.time() > deadline - 1:
                    break
                w, h, d = boxes[idx]
                rots = _rotations(w, h, d)
                
                # Get current placed boxes
                placed_boxes = []
                for i in range(n):
                    p = best_placements[i]
                    if p[0] == 0:
                        placed_boxes.append((p[1], p[2], p[3], p[4], p[5], p[6]))
                
                candidates = _extreme_points(placed_boxes, container)
                face_pts = _face_points(placed_boxes, container)
                for fp in face_pts:
                    if fp not in candidates:
                        candidates.append(fp)
                if (0, 0, 0) not in candidates:
                    candidates.append((0, 0, 0))
                
                best_pos = None
                best_rot = None
                best_score = None
                
                for pos in candidates:
                    for rw, rh, rd in rots:
                        if _fits((rw, rh, rd), pos, container, placed_boxes):
                            score = _score_position(pos, (rw, rh, rd), placed_boxes, container)
                            if best_score is None or score < best_score:
                                best_score = score
                                best_pos = pos
                                best_rot = (rw, rh, rd)
                
                if best_pos is not None:
                    x, y, z = best_pos
                    rw, rh, rd = best_rot
                    best_placements[idx] = (0, x, y, z, rw, rh, rd)
                    best_vol += rw * rh * rd
            
            # One more local search pass to optimize the new placement
            if time.time() < deadline - 2:
                best_placements, best_vol = _shake_local_search_v2(
                    best_placements, boxes, container, deadline, rng3, num_shakes=30
                )
            
    return best_placements
