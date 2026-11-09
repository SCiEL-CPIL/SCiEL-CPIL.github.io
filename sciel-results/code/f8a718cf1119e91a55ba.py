"""3D Container Loading: Fast search-based approach with diverse heuristics and local improvement."""
import time
import random
import itertools
from collections import defaultdict

def _rotations(w, h, d):
    """Return all 6 unique orientations of a box (w,h,d)."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _fits(box_dims, pos, container, placed):
    """Check if box fits at pos without overlap and within container."""
    W, H, D = container
    x, y, z = pos
    w, h, d = box_dims
    if x + w > W or y + h > H or z + d > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        if x >= px + pw or x + w <= px:
            continue
        if y >= py + ph or y + h <= py:
            continue
        if z >= pz + pd or z + d <= pz:
            continue
        return False
    return True

def _candidate_positions(placed, container, mode='faces'):
    """Generate candidate (x,y,z) positions from placed boxes."""
    W, H, D = container
    
    if not placed:
        return [(0, 0, 0)]
    
    candidates = set()
    candidates.add((0, 0, 0))
    
    # Generate points based on faces of existing boxes
    # This is O(N) in the number of placed boxes
    for (px, py, pz, pw, ph, pd) in placed:
        candidates.add((px + pw, py, pz))
        candidates.add((px, py + ph, pz))
        candidates.add((px, py, pz + pd))
        
        # Add some corner-like combinations to improve packing
        candidates.add((px + pw, py + ph, pz))
        candidates.add((px + pw, py, pz + pd))
        candidates.add((px, py + ph, pz + pd))
        candidates.add((px + pw, py + ph, pz + pd))
        
    # Filter valid points
    points = []
    for (x, y, z) in candidates:
        if 0 <= x < W and 0 <= y < H and 0 <= z < D:
            points.append((x, y, z))
            
    # If too many points, limit to a random subset to prevent timeout
    # 200 points is usually sufficient for 60 boxes
    if len(points) > 200:
        import random
        # Use a fixed seed for determinism in the subset if needed, 
        # but random is fine for exploration
        points = random.sample(points, 200)
        
    return points

def _position_score(pos, box_dims, placed, container, strategy='compact'):
    """Score a candidate position based on the strategy."""
    W, H, D = container
    x, y, z = pos
    w, h, d = box_dims
    
    if strategy == 'compact':
        # Prefer lowest z, then lowest y, then lowest x
        return (z, y, x)
    elif strategy == 'back':
        # Prefer furthest back (high x, high y, high z)
        return (-x, -y, -z)
    elif strategy == 'corner':
        # Prefer corners: maximize distance from origin
        return (-(x + y + z),)
    elif strategy == 'fill':
        # Prefer positions that minimize remaining gaps
        # Approximate by preferring lower positions
        return (z, y, x)
    else:
        return (z, y, x)

def _greedy_place(boxes, container, order, deadline, mode='extreme', strategy='compact'):
    """Greedy placement with given order and mode."""
    W, H, D = container
    placed = []
    placements = [None] * len(boxes)
    total_vol = 0.0
    n = len(boxes)
    
    # Build sorted candidate coordinates for faster lookup
    # Pre-sort placed boxes by their max extents for early termination
    
    for idx in order:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        rots = _rotations(w, h, d)
        
        candidates = _candidate_positions(placed, container, mode)
        
        best_pos = None
        best_rot = None
        best_key = None
        
        # Sort candidates by position score first to enable early termination
        scored_candidates = []
        for pos in candidates:
            for rw, rh, rd in rots:
                key = _position_score(pos, (rw, rh, rd), placed, container, strategy)
                scored_candidates.append((key, pos, (rw, rh, rd)))
        
        scored_candidates.sort()
        
        for key, pos, (rw, rh, rd) in scored_candidates:
            if _fits((rw, rh, rd), pos, container, placed):
                best_pos = pos
                best_rot = (rw, rh, rd)
                best_key = key
                break
        
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

def _local_search(placements, boxes, container, deadline, rng, max_iters=50):
    """Try to improve by removing and re-placing boxes."""
    W, H, D = container
    n = len(boxes)
    
    placed = []
    for i in range(n):
        p = placements[i]
        if p[0] == 0:
            placed.append((p[1], p[2], p[3], p[4], p[5], p[6]))
    
    loaded_indices = [i for i in range(n) if placements[i][0] == 0]
    if not loaded_indices:
        return placements, 0.0
    
    total_vol = sum(boxes[i][0] * boxes[i][1] * boxes[i][2] for i in loaded_indices)
    
    improved = True
    iters = 0
    while improved and iters < max_iters and time.time() < deadline:
        improved = False
        iters += 1
        
        for idx in loaded_indices:
            if time.time() > deadline:
                break
            
            p = placements[idx]
            box_pos = (p[1], p[2], p[3], p[4], p[5], p[6])
            
            # Try removing this box
            try:
                placed.remove(box_pos)
            except ValueError:
                continue
            
            w, h, d = boxes[idx]
            rots = _rotations(w, h, d)
            candidates = _candidate_positions(placed, container, 'extreme')
            
            best_pos = None
            best_rot = None
            best_key = None
            
            # Sort candidates for early termination
            scored_candidates = []
            for pos in candidates:
                for rw, rh, rd in rots:
                    key = (pos[2], pos[1], pos[0])
                    scored_candidates.append((key, pos, (rw, rh, rd)))
            
            scored_candidates.sort()
            
            for key, pos, (rw, rh, rd) in scored_candidates:
                if _fits((rw, rh, rd), pos, container, placed):
                    best_pos = pos
                    best_rot = (rw, rh, rd)
                    best_key = key
                    break
            
            if best_pos is not None:
                x, y, z = best_pos
                rw, rh, rd = best_rot
                new_pos = (x, y, z, rw, rh, rd)
                
                # Check if this is an improvement (lower position)
                old_score = (box_pos[2], box_pos[1], box_pos[0])
                new_score = (z, y, x)
                
                if new_score < old_score:
                    placed.append(new_pos)
                    placements[idx] = (0, x, y, z, rw, rh, rd)
                    improved = True
                else:
                    # Put it back
                    placed.append(box_pos)
            else:
                # Can't re-place, put it back
                placed.append(box_pos)
    
    # Recalculate volume
    total_vol = 0.0
    for i in range(n):
        if placements[i][0] == 0:
            total_vol += placements[i][4] * placements[i][5] * placements[i][6]
    
    return placements, total_vol

def _shake_local_search(placements, boxes, container, deadline, rng, max_shakes=20):
    """Shake local search: randomly remove a few boxes and re-place them."""
    W, H, D = container
    n = len(boxes)
    
    placed = []
    for i in range(n):
        p = placements[i]
        if p[0] == 0:
            placed.append((p[1], p[2], p[3], p[4], p[5], p[6]))
    
    loaded_indices = [i for i in range(n) if placements[i][0] == 0]
    if not loaded_indices:
        return placements, 0.0
    
    total_vol = sum(boxes[i][0] * boxes[i][1] * boxes[i][2] for i in loaded_indices)
    
    best_placements = [p[:] for p in placements]
    best_vol = total_vol
    
    for shake in range(max_shakes):
        if time.time() > deadline:
            break
        
        # Remove 1-3 random boxes
        num_remove = min(rng.randint(1, 3), len(loaded_indices))
        to_remove = rng.sample(loaded_indices, num_remove)
        
        removed_boxes = []
        for idx in to_remove:
            p = placements[idx]
            box_pos = (p[1], p[2], p[3], p[4], p[5], p[6])
            try:
                placed.remove(box_pos)
            except ValueError:
                continue
            removed_boxes.append((idx, box_pos))
        
        # Try to re-place them
        success = True
        new_placements = [p[:] for p in placements]
        
        for idx, old_pos in removed_boxes:
            w, h, d = boxes[idx]
            rots = _rotations(w, h, d)
            candidates = _candidate_positions(placed, container, 'extreme')
            
            best_pos = None
            best_rot = None
            best_key = None
            
            # Sort candidates for early termination
            scored_candidates = []
            for pos in candidates:
                for rw, rh, rd in rots:
                    key = (pos[2], pos[1], pos[0])
                    scored_candidates.append((key, pos, (rw, rh, rd)))
            
            scored_candidates.sort()
            
            for key, pos, (rw, rh, rd) in scored_candidates:
                if _fits((rw, rh, rd), pos, container, placed):
                    best_pos = pos
                    best_rot = (rw, rh, rd)
                    best_key = key
                    break
            
            if best_pos is not None:
                x, y, z = best_pos
                rw, rh, rd = best_rot
                placed.append((x, y, z, rw, rh, rd))
                new_placements[idx] = (0, x, y, z, rw, rh, rd)
            else:
                # Can't place, revert
                success = False
                break
        
        if success:
            # Calculate new volume
            new_vol = 0.0
            for i in range(n):
                if new_placements[i][0] == 0:
                    new_vol += new_placements[i][4] * new_placements[i][5] * new_placements[i][6]
            
            if new_vol > best_vol:
                best_vol = new_vol
                best_placements = [p[:] for p in new_placements]
                # Update placed to match best
                placed = []
                for i in range(n):
                    p = best_placements[i]
                    if p[0] == 0:
                        placed.append((p[1], p[2], p[3], p[4], p[5], p[6]))
        else:
            # Revert all changes
            for idx, old_pos in removed_boxes:
                try:
                    placed.remove(old_pos)
                except ValueError:
                    pass
                placed.append(old_pos)
    
    return best_placements, best_vol

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s/150s kill
    W, H, D = container
    n = len(boxes)
    
    vols = [w * h * d for w, h, d in boxes]
    mins = [min(w, h, d) for w, h, d in boxes]
    maxs = [max(w, h, d) for w, h, d in boxes]
    ratios = [max(w, h, d) / max(min(w, h, d), 1e-9) for w, h, d in boxes]
    
    best_placements = None
    best_vol = 0.0
    
    # Generate diverse sorting orders
    sort_strategies = [
        ('vol_desc', lambda i: vols[i], True),
        ('vol_asc', lambda i: vols[i], False),
        ('min_dim_desc', lambda i: mins[i], True),
        ('min_dim_asc', lambda i: mins[i], False),
        ('max_dim_desc', lambda i: maxs[i], True),
        ('max_dim_asc', lambda i: maxs[i], False),
        ('ratio_desc', lambda i: ratios[i], True),
        ('ratio_asc', lambda i: ratios[i], False),
    ]
    
    modes = ['extreme', 'faces']
    strategies = ['compact', 'back', 'corner']
    
    # First pass: systematic exploration of different combinations
    # Reduced number of strategies to save time
    for sort_name, key_fn, reverse in sort_strategies[:4]: # Only top 4
        if time.time() > deadline:
            break
        
        order = list(range(n))
        order.sort(key=key_fn, reverse=reverse)
        
        # Only use 'faces' mode which is faster
        placements, vol = _greedy_place(boxes, container, order, deadline, 'faces', 'compact')
        if vol > best_vol:
            best_vol = vol
            best_placements = [p[:] for p in placements]
    
    # Local search on best
    if best_placements is not None and time.time() < deadline:
        rng = random.Random(42)
        # Reduced iterations for local search
        best_placements, best_vol = _local_search(best_placements, boxes, container, deadline, rng, max_iters=10)
    
    # Random restarts with perturbed orders
    rng = random.Random(123)
    base_order = list(range(n))
    base_order.sort(key=lambda i: vols[i], reverse=True)
    
    # Use remaining time for random restarts
    remaining_time = deadline - time.time()
    # Estimate restarts more conservatively
    num_restarts = int(min(50, max(10, remaining_time / 2.0)))
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Perturb the base order
        order = base_order[:]
        num_swaps = rng.randint(2, min(12, n))
        for _ in range(num_swaps):
            a, b = rng.randrange(n), rng.randrange(n)
            if a != b:
                order[a], order[b] = order[b], order[a]
        
        strat = rng.choice(strategies)
        
        placements, vol = _greedy_place(boxes, container, order, deadline, 'faces', strat)
        if vol > best_vol:
            best_vol = vol
            best_placements = [p[:] for p in placements]
    
    # Final shake local search
    if best_placements is not None and time.time() < deadline:
        rng2 = random.Random(999)
        best_placements, best_vol = _shake_local_search(best_placements, boxes, container, deadline, rng2, max_shakes=10)
    
    if best_placements is None:
        # Fallback: try to place boxes in order of volume
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        # Try greedy as fallback
        order = list(range(n))
        order.sort(key=lambda i: vols[i], reverse=True)
        placements, vol = _greedy_place(boxes, container, order, deadline, 'extreme', 'compact')
        if vol > 0:
            best_placements = placements
    
    return best_placements
