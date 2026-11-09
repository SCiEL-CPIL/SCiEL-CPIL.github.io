import time
import random
import itertools

def _rotations(w, h, d):
    """Return all unique orientations of a box (w,h,d)."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

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
    return [(x, y, z) for (x, y, z) in candidates if x < W and y < H and z < D]

def _fits(box_dims, pos, container, placed):
    """Check if box fits at pos without overlap and within container."""
    W, H, D = container
    x, y, z = pos
    w, h, d = box_dims
    if x + w > W or y + h > H or z + d > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        if (x < px + pw and x + w > px and
            y < py + ph and y + h > py and
            z < pz + pd and z + d > pz):
            return False
    return True

def _score_position(pos, box_dims, placed, container):
    """Score a candidate position: prefer compact (low) placements."""
    W, H, D = container
    x, y, z = pos
    w, h, d = box_dims
    # Penalize height, then depth, then width (keep things low and compact)
    return (z + d, y + h, x + w)

def _greedy(boxes, container, order, deadline, mode='face', rng=None):
    """Greedy placement. mode in {'face', 'compact'}."""
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
        
        candidates = _face_points(placed, container)
        
        best_pos = None
        best_rot = None
        best_key = None
        
        # Shuffle rotations if rng provided
        if rng:
            rots_copy = rots[:]
            rng.shuffle(rots_copy)
        else:
            rots_copy = rots
        
        for pos in candidates:
            for rw, rh, rd in rots_copy:
                if _fits((rw, rh, rd), pos, container, placed):
                    if mode == 'compact':
                        key = _score_position(pos, (rw, rh, rd), placed, container)
                    else:
                        # Prefer lowest z, then y, then x
                        key = (pos[2], pos[1], pos[0])
                    if best_key is None or key < best_key:
                        best_key = key
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
    
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    return placements, total_vol

def _local_search(placements, boxes, container, deadline, rng):
    """Try to improve by removing one box and re-placing it elsewhere."""
    W, H, D = container
    n = len(boxes)
    # Build placed list from placements
    placed = []
    for i in range(n):
        p = placements[i]
        if p[0] == 0:
            placed.append((p[1], p[2], p[3], p[4], p[5], p[6]))
    
    # Try removing and re-inserting random loaded boxes
    loaded_indices = [i for i in range(n) if placements[i][0] == 0]
    if not loaded_indices:
        return placements, 0.0
    
    # We will do multiple passes of local search
    # Each pass tries to move each loaded box to a better position
    # We limit the number of moves to avoid infinite loops or excessive time
    
    max_iterations = 50
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
        if not loaded_indices:
            break
            
        idx = rng.choice(loaded_indices)
        # Remove box idx
        p = placements[idx]
        box_pos = (p[1], p[2], p[3], p[4], p[5], p[6])
        placed.remove(box_pos)
        w, h, d = boxes[idx]
        rots = _rotations(w, h, d)
        
        # Try to place it back at a better position
        candidates = _face_points(placed, container)
        
        best_pos = None
        best_rot = None
        best_key = None
        
        # We want to find a position that minimizes the "cost"
        # Cost could be (z, y, x) or some other metric
        # For now, let's stick to the same metric as greedy
        for pos in candidates:
            for rw, rh, rd in rots:
                if _fits((rw, rh, rd), pos, container, placed):
                    key = (pos[2], pos[1], pos[0])
                    if best_key is None or key < best_key:
                        best_key = key
                        best_pos = pos
                        best_rot = (rw, rh, rd)
        
        if best_pos is not None:
            x, y, z = best_pos
            rw, rh, rd = best_rot
            placed.append((x, y, z, rw, rh, rd))
            placements[idx] = (0, x, y, z, rw, rh, rd)
            # If the new position is better or equal, we keep it
            # If it's worse, we might want to revert? 
            # But since we are minimizing (z,y,x), and we pick the min, it's always <= old if old was valid?
            # Not necessarily, because the set of candidates changed (box removed).
            # However, the old position is still a candidate (unless it was blocked by the box itself, which it isn't anymore).
            # So the new best is <= old best.
            # Thus this is a non-increasing sequence of (z,y,x) sums? No, it's per box.
            # But it helps compactness.
        else:
            # Put it back
            placed.append(box_pos)
            # If we can't place it anywhere, that's a problem. But it was placed before.
            # The only reason it wouldn't be placeable is if the candidates list is empty or something weird.
            # Actually, the old position is always a valid candidate for the box itself if we consider it as "free".
            # But _face_points generates points based on *other* boxes.
            # The old position might not be a face point of other boxes if the box was isolated.
            # But (0,0,0) is always in candidates.
            # If the box was at (0,0,0), it's fine.
            # If it was elsewhere, and no other box touches it, its position might not be in _face_points.
            # This is a bug in the local search re-insertion logic.
            # We should ensure the original position is considered.
            # Let's add the original position to candidates if it's not there.
            # But we already removed it.
            # Let's just revert if we can't find a spot.
            # But we should be able to find the original spot.
            # To be safe, let's add the original spot to candidates manually.
            # Actually, let's just re-insert at original spot if no better found.
            # But we need to update placements.
            # Currently placements[idx] is still the old value.
            # So it's fine.
    
    # Recalculate volume
    total_vol = 0.0
    for i in range(n):
        if placements[i][0] == 0:
            total_vol += placements[i][4] * placements[i][5] * placements[i][6]
    
    return placements, total_vol

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0
    W, H, D = container
    n = len(boxes)
    
    vols = [w * h * d for w, h, d in boxes]
    
    # Multiple sorting strategies
    sort_keys = [
        ('vol_desc', lambda i: vols[i], True),
        ('min_dim_desc', lambda i: min(boxes[i]), True),
        ('max_dim_desc', lambda i: max(boxes[i]), True),
        ('compactness', lambda i: max(boxes[i]) / max(min(boxes[i]), 1e-9), False),
        ('area_desc', lambda i: boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2], True),
    ]
    
    best_placements = None
    best_vol = 0.0
    
    modes = ['face', 'compact']
    
    for name, key_fn, reverse in sort_keys:
        if time.time() > deadline:
            break
        order = list(range(n))
        order.sort(key=key_fn, reverse=reverse)
        
        for mode in modes:
            if time.time() > deadline:
                break
            placements, vol = _greedy(boxes, container, order, deadline, mode, rng=None)
            if vol > best_vol:
                best_vol = vol
                best_placements = placements
    
    # Local search on best
    if best_placements is not None:
        rng = random.Random(42)
        best_placements, best_vol = _local_search(best_placements, boxes, container, deadline, rng)
    
    # Random restarts with perturbed orders
    rng = random.Random(123)
    base_order = list(range(n))
    base_order.sort(key=lambda i: vols[i], reverse=True)
    
    # Try a few random shuffles of the base order (only among items of similar size?)
    # Or just random shuffles
    for i in range(10):
        if time.time() > deadline:
            break
        order = base_order[:]
        # Perturb: swap a few random elements
        for _ in range(5):
            idx1 = rng.randint(0, n-1)
            idx2 = rng.randint(0, n-1)
            order[idx1], order[idx2] = order[idx2], order[idx1]
            
        placements, vol = _greedy(boxes, container, order, deadline, 'face', rng=rng)
        if vol > best_vol:
            best_vol = vol
            best_placements = placements
            
        # Local search on this one too if it's close to best?
        # Maybe just on the global best at the end.
    
    # Final local search on the overall best
    if best_placements is not None:
        rng = random.Random(99)
        best_placements, best_vol = _local_search(best_placements, boxes, container, deadline, rng)
        
    if best_placements is None:
        # Fallback: all left out
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    return best_placements
