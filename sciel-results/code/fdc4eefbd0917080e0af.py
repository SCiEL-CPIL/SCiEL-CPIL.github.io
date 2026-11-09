import time
import itertools
import random

def _rotations(w, h, d):
    """Return all 6 unique orientations of a box."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _fits(box_dims, pos, container, placed):
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
    xs = [x for x in xs if x < W]
    ys = [y for y in ys if y < H]
    zs = [z for z in zs if z < D]
    points = []
    for x in xs:
        for y in ys:
            for z in zs:
                points.append((x, y, z))
    return points

def _face_points(placed, container):
    """Generate candidate points on faces of placed boxes."""
    W, H, D = container
    points = set([(0, 0, 0)])
    for (px, py, pz, pw, ph, pd) in placed:
        # Points on the three faces opposite to the origin
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

def _score_position(pos, box_dims, container, placed, mode='blb'):
    """Score a position for a box. Lower is better."""
    x, y, z = pos
    w, h, d = box_dims
    W, H, D = container
    
    if mode == 'blb':
        # Bottom-Left-Back: prefer low x, y, z
        return (x, y, z)
    elif mode == 'compact':
        # Prefer positions that minimize the bounding box extent
        max_x = max([p[0] + p[3] for p in placed] + [x + w])
        max_y = max([p[1] + p[4] for p in placed] + [y + h])
        max_z = max([p[2] + p[5] for p in placed] + [z + d])
        return (max_x * max_y * max_z, x, y, z)
    elif mode == 'gap':
        # Prefer positions that fill gaps (closer to existing boxes)
        min_dist = float('inf')
        for (px, py, pz, pw, ph, pd) in placed:
            dx = max(0, max(x - (px + pw), px - (x + w)))
            dy = max(0, max(y - (py + ph), py - (y + h)))
            dz = max(0, max(z - (pz + pd), pz - (z + d)))
            dist = dx + dy + dz
            if dist < min_dist:
                min_dist = dist
        return (min_dist, x, y, z)
    else:
        return (x, y, z)

def _greedy(boxes, container, sort_key, score_mode='blb', use_face_points=False, time_limit=30):
    """Run a single greedy packing strategy."""
    W, H, D = container
    n = len(boxes)
    start_time = time.time()
    
    indexed_boxes = [(boxes[j], j) for j in range(n)]
    indexed_boxes.sort(key=lambda x: sort_key(x[0]), reverse=True)
    
    placed_indices = {}
    placed_boxes = []
    total_vol = 0
    
    for box_dims, orig_idx in indexed_boxes:
        if time.time() - start_time > time_limit:
            break
        
        w, h, d = box_dims
        rots = _rotations(w, h, d)
        
        if use_face_points:
            candidates = _face_points(placed_boxes, container)
        else:
            candidates = _extreme_points(placed_boxes, container)
        
        valid_options = []
        for cx, cy, cz in candidates:
            for rw, rh, rd in rots:
                if _fits((rw, rh, rd), (cx, cy, cz), container, placed_boxes):
                    score = _score_position((cx, cy, cz), (rw, rh, rd), container, placed_boxes, score_mode)
                    valid_options.append((score, cx, cy, cz, rw, rh, rd))
        
        if not valid_options:
            continue
        
        valid_options.sort(key=lambda opt: opt[0])
        _, bx, by, bz, bw, bh, bd = valid_options[0]
        
        placed_indices[orig_idx] = (bx, by, bz, bw, bh, bd)
        placed_boxes.append((bx, by, bz, bw, bh, bd))
        total_vol += bw * bh * bd
    
    return placed_indices, total_vol

def _local_search(placed_indices, boxes, container, max_iterations=50, time_limit=30):
    """Try to improve the placement by moving boxes to better positions."""
    W, H, D = container
    n = len(boxes)
    start_time = time.time()
    
    # Create a mutable copy
    current_placement = dict(placed_indices)
    current_volume = sum(bw * bh * bd for bw, bh, bd in 
                        [(p[3], p[4], p[5]) for p in current_placement.values()])
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        placed_boxes = list(current_placement.values())
        
        # Try to improve each placed box
        for idx in list(current_placement.keys()):
            if time.time() - start_time > time_limit:
                break
            
            x, y, z, w, h, d = current_placement[idx]
            box_dims = boxes[idx]
            rots = _rotations(box_dims[0], box_dims[1], box_dims[2])
            
            # Remove this box temporarily
            placed_boxes = [p for p in placed_boxes if p != (x, y, z, w, h, d)]
            
            best_score = _score_position((x, y, z), (w, h, d), container, placed_boxes, 'compact')
            best_pos = (x, y, z, w, h, d)
            
            candidates = _face_points(placed_boxes, container)
            for cx, cy, cz in candidates:
                for rw, rh, rd in rots:
                    if _fits((rw, rh, rd), (cx, cy, cz), container, placed_boxes):
                        score = _score_position((cx, cy, cz), (rw, rh, rd), container, placed_boxes, 'compact')
                        if score < best_score:
                            best_score = score
                            best_pos = (cx, cy, cz, rw, rh, rd)
            
            if best_pos != (x, y, z, w, h, d):
                current_placement[idx] = best_pos
                placed_boxes.append(best_pos)
                improved = True
            else:
                placed_boxes.append((x, y, z, w, h, d))
        
        if not improved:
            break
    
    return current_placement, current_volume

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # Time guard
    start_time = time.time()
    total_time_limit = 140  # Leave 10s margin
    
    # Define sorting strategies
    strategies = [
        ('volume', lambda b: b[0]*b[1]*b[2]),
        ('max_dim', lambda b: max(b)),
        ('surface', lambda b: 2*(b[0]*b[1] + b[1]*b[2] + b[0]*b[2])),
        ('min_dim', lambda b: -min(b)),  # Negative for ascending sort
    ]
    
    score_modes = ['blb', 'compact', 'gap']
    
    best_placements = None
    best_volume = 0
    
    # Run multiple greedy strategies with different scoring modes
    for strategy_name, sort_key in strategies:
        if time.time() - start_time > total_time_limit:
            break
        
        for score_mode in score_modes:
            if time.time() - start_time > total_time_limit:
                break
            
            # Calculate time remaining for this greedy run
            elapsed = time.time() - start_time
            time_left = total_time_limit - elapsed
            per_greedy_time = max(5, min(30, time_left / 12))  # 12 total greedy runs
            
            placements, vol = _greedy(boxes, container, sort_key, score_mode, 
                                       use_face_points=True, time_limit=per_greedy_time)
            
            if vol > best_volume:
                best_volume = vol
                best_placements = placements.copy()
    
    # Run local search on the best placement
    if best_placements is not None and time.time() - start_time < total_time_limit:
        elapsed = time.time() - start_time
        local_search_time = max(10, min(40, total_time_limit - elapsed - 5))
        
        improved_placements, improved_vol = _local_search(
            best_placements, boxes, container, 
            max_iterations=100, time_limit=local_search_time
        )
        
        if improved_vol > best_volume:
            best_volume = improved_vol
            best_placements = improved_placements
    
    # Construct final output
    if best_placements is None:
        return [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    
    result = []
    for i in range(n):
        if i in best_placements:
            x, y, z, w, h, d = best_placements[i]
            result.append((0, x, y, z, w, h, d))
        else:
            w, h, d = boxes[i]
            result.append((-1, 0, 0, 0, w, h, d))
    
    return result
