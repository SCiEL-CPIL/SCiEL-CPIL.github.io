import time
import random
import itertools

def _rotations(w, h, d):
    """Return all 6 unique orientations of a box."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _fits(box_dims, pos, container):
    W, H, D = container
    x, y, z = pos
    w, h, d = box_dims
    return x + w <= W and y + h <= H and z + d <= D

def _no_overlap(box_dims, pos, placed):
    x, y, z = pos
    w, h, d = box_dims
    for (px, py, pz, pw, ph, pd) in placed:
        if (x < px + pw and x + w > px and
            y < py + ph and y + h > py and
            z < pz + pd and z + d > pz):
            return False
    return True

def _extreme_points(placed, container):
    W, H, D = container
    candidates = set()
    candidates.add((0, 0, 0))
    for (px, py, pz, pw, ph, pd) in placed:
        points = [
            (px + pw, py, pz),
            (px, py + ph, pz),
            (px, py, pz + pd),
            (px + pw, py + ph, pz),
            (px, py + ph, pz + pd),
            (px + pw, py, pz + pd),
            (px + pw, py + ph, pz + pd),
        ]
        for (cx, cy, cz) in points:
            if cx <= W and cy <= H and cz <= D:
                candidates.add((cx, cy, cz))
    return list(candidates)

def _try_place_eap(boxes, container, order_idx, deadline, scoring_func=None, rng=None):
    """Extreme Points Approach with optional custom scoring and randomization."""
    W, H, D = container
    placed = []
    placements = [None] * len(boxes)
    
    if scoring_func is None:
        def scoring_func(pos, dim):
            return (pos[2], pos[1], pos[0])
    
    for orig_idx in order_idx:
        if time.time() > deadline:
            break
        w, h, d = boxes[orig_idx]
        rots = _rotations(w, h, d)
        candidates = _extreme_points(placed, container)
        
        best_pos = None
        best_dim = None
        best_score = None
        best_candidates = []
        
        for (cx, cy, cz) in candidates:
            for (rw, rh, rd) in rots:
                if not _fits((rw, rh, rd), (cx, cy, cz), container):
                    continue
                if not _no_overlap((rw, rh, rd), (cx, cy, cz), placed):
                    continue
                
                score = scoring_func((cx, cy, cz), (rw, rh, rd))
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
                    best_dim = (rw, rh, rd)
                    best_candidates = [(best_pos, best_dim)]
                elif score == best_score:
                    best_candidates.append(((cx, cy, cz), (rw, rh, rd)))
        
        if best_pos is not None:
            if rng and len(best_candidates) > 1:
                best_pos, best_dim = rng.choice(best_candidates)
            
            placed.append((best_pos[0], best_pos[1], best_pos[2],
                          best_dim[0], best_dim[1], best_dim[2]))
            placements[orig_idx] = (0, best_pos[0], best_pos[1], best_pos[2],
                                   best_dim[0], best_dim[1], best_dim[2])
        else:
            placements[orig_idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(len(boxes)):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    volume_filled = sum(p[4] * p[5] * p[6] for p in placements if p[0] == 0)
    return placements, volume_filled

def _shelf_place(boxes, container, order_idx, deadline):
    """Simple shelf-based placement for diversity."""
    W, H, D = container
    placements = [None] * len(boxes)
    x, y, z = 0, 0, 0
    shelf_h = 0
    shelf_d = 0
    
    for orig_idx in order_idx:
        if time.time() > deadline:
            break
        w, h, d = boxes[orig_idx]
        rots = _rotations(w, h, d)
        
        best = None
        for rw, rh, rd in rots:
            # Try current position
            if x + rw <= W and y + rh <= H and z + rd <= D:
                best = (x, y, z, rw, rh, rd)
                break
            
            # Move to next shelf
            if x + rw > W:
                x = 0
                shelf_h = 0
            if y + rh > H:
                y = 0
                z = max(z, shelf_d) + (shelf_h if shelf_h > 0 else 0)
                shelf_h = 0
                shelf_d = 0
            
            if z + rd <= D and x + rw <= W and y + rh <= H:
                best = (x, y, z, rw, rh, rd)
                break
        
        if best:
            bx, by, bz, rw, rh, rd = best
            placements[orig_idx] = (0, bx, by, bz, rw, rh, rd)
            x += rw
            shelf_h = max(shelf_h, rh)
            shelf_d = max(shelf_d, rd)
        else:
            placements[orig_idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(len(boxes)):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    volume_filled = sum(p[4] * p[5] * p[6] for p in placements if p[0] == 0)
    return placements, volume_filled

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 145
    
    W, H, D = container
    container_volume = W * H * D
    n = len(boxes)
    
    volumes = [b[0]*b[1]*b[2] for b in boxes]
    max_dims = [max(b) for b in boxes]
    min_dims = [min(b) for b in boxes]
    sum_dims = [sum(b) for b in boxes]
    surface_areas = [2*(b[0]*b[1] + b[1]*b[2] + b[0]*b[2]) for b in boxes]
    
    # Generate diverse initial solutions
    strategies = []
    
    vol_order = sorted(range(n), key=lambda i: volumes[i], reverse=True)
    strategies.append(('eap', vol_order, None, None))
    
    maxdim_order = sorted(range(n), key=lambda i: max_dims[i], reverse=True)
    strategies.append(('eap', maxdim_order, None, None))
    
    mindim_order = sorted(range(n), key=lambda i: min_dims[i])
    strategies.append(('eap', mindim_order, None, None))
    
    sumdim_order = sorted(range(n), key=lambda i: sum_dims[i], reverse=True)
    strategies.append(('eap', sumdim_order, None, None))
    
    sa_order = sorted(range(n), key=lambda i: surface_areas[i], reverse=True)
    strategies.append(('eap', sa_order, None, None))
    
    mixed1 = sorted(range(n), key=lambda i: (volumes[i], max_dims[i]), reverse=True)
    strategies.append(('eap', mixed1, None, None))
    
    def score_sum(pos, dim):
        return pos[0] + pos[1] + pos[2]
        
    def score_max(pos, dim):
        return max(pos)
        
    def score_z_sum(pos, dim):
        return (pos[2], pos[0] + pos[1])
        
    def score_z_max(pos, dim):
        return (pos[2], max(pos[0], pos[1]))
        
    def score_compact(pos, dim):
        return (pos[0]*pos[1]*pos[2])
        
    strategies.append(('eap', vol_order, score_sum, None))
    strategies.append(('eap', vol_order, score_max, None))
    strategies.append(('eap', vol_order, score_z_sum, None))
    strategies.append(('eap', vol_order, score_z_max, None))
    strategies.append(('eap', vol_order, score_compact, None))
    
    # Random tie-breaking variants
    for seed in range(8):
        strategies.append(('eap', vol_order, None, random.Random(seed)))
    
    # Shelf-based for diversity
    strategies.append(('shelf', vol_order, None, None))
    strategies.append(('shelf', maxdim_order, None, None))
    
    best_placements = None
    best_volume = -1
    
    # Run heuristics
    for algo, order, scorer, rng in strategies:
        if time.time() > deadline - 70:
            break
        
        if algo == 'eap':
            placements, vol = _try_place_eap(boxes, container, order, deadline - 70, scorer, rng)
        else:
            placements, vol = _shelf_place(boxes, container, order, deadline - 70)
        
        if vol > best_volume:
            best_volume = vol
            best_placements = placements
            if best_volume >= container_volume * 0.99:
                break
    
    if best_placements is None:
        placements = []
        for w, h, d in boxes:
            placements.append((-1, 0, 0, 0, w, h, d))
        best_placements = placements
        best_volume = 0

    # Local search / perturbation phase
    current_placements = [tuple(p) for p in best_placements]
    current_volume = best_volume
    
    def get_placed_list(placements):
        placed = []
        for i, p in enumerate(placements):
            if p[0] == 0:
                placed.append((p[1], p[2], p[3], p[4], p[5], p[6]))
        return placed
    
    def try_insert_box(box_idx, placements, container, rng=None):
        w, h, d = boxes[box_idx]
        rots = _rotations(w, h, d)
        placed = get_placed_list(placements)
        candidates = _extreme_points(placed, container)
        
        if rng and len(candidates) > 1:
            rng.shuffle(candidates)
        
        for (cx, cy, cz) in candidates:
            for (rw, rh, rd) in rots:
                if not _fits((rw, rh, rd), (cx, cy, cz), container):
                    continue
                if not _no_overlap((rw, rh, rd), (cx, cy, cz), placed):
                    continue
                return (0, cx, cy, cz, rw, rh, rd)
        return None
    
    # Iterated local search
    rng_ls = random.Random(42)
    ls_rounds = 20
    
    for round_num in range(ls_rounds):
        if time.time() > deadline - 5:
            break
        
        # Perturbation: remove k random placed boxes
        k = random.Random(round_num).randint(2, 5)
        placed_indices = [i for i, p in enumerate(current_placements) if p[0] == 0]
        
        if not placed_indices:
            break
        
        removed = set()
        for _ in range(min(k, len(placed_indices))):
            idx = rng_ls.choice(placed_indices)
            if idx not in removed:
                removed.add(idx)
                current_placements[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
        
        # Try to re-insert removed boxes
        for idx in sorted(removed):
            new_pl = try_insert_box(idx, current_placements, container, rng_ls)
            if new_pl:
                current_placements[idx] = new_pl
        
        # Try to insert any remaining unplaced boxes
        unplaced = [i for i, p in enumerate(current_placements) if p[0] == -1]
        for idx in unplaced:
            new_pl = try_insert_box(idx, current_placements, container, rng_ls)
            if new_pl:
                current_placements[idx] = new_pl
        
        # Compute current volume
        current_volume = sum(p[4] * p[5] * p[6] for p in current_placements if p[0] == 0)
        
        # Accept if better
        if current_volume > best_volume:
            best_volume = current_volume
            best_placements = [tuple(p) for p in current_placements]
        
        # If not better, revert perturbation (keep best)
        # But we already modified current_placements, so we need to track state
        # For simplicity, just keep going with the modified state
    
    return best_placements
