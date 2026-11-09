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
            return (pos[2], pos[1], pos[0])  # Bottom, Left, Far
    
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

def _adaptive_shelf_pack(boxes, container):
    """
    Adaptive shelving: treat container as a stack of horizontal shelves.
    Each shelf has a fixed height (tallest box placed in it).
    Within a shelf, boxes are placed left-to-right, front-to-back.
    Boxes that don't fit in the current shelf start a new shelf (if height allows).
    This is NOT the same as uniform layers - shelf heights adapt to box sizes.
    """
    W, H, D = container
    placements = [None] * len(boxes)
    
    # Sort boxes: tallest first (by max dimension), then by volume
    def sort_key(i):
        w, h, d = boxes[i]
        return (max(w, h, d), -(w * h * d))
    
    order = sorted(range(len(boxes)), key=sort_key)
    
    # State: current shelf info
    shelf_height = 0       # height of current shelf
    z_offset = 0           # z-position of current shelf bottom
    # Within shelf: track occupancy in the W x D plane
    # We'll use a simple list of (x, y, w, d) placed in current shelf
    shelf_boxes = []       # list of (x, y, w, d, box_idx)
    x_cursor = 0           # current x position for new box in row
    y_cursor = 0           # current y position for new box in row
    row_max_d = 0          # max depth in current row
    
    for idx in order:
        w, h, d = boxes[idx]
        rots = _rotations(w, h, d)
        placed_this = False
        
        for rw, rh, rd in rots:
            # Check if this orientation fits in remaining vertical space
            if z_offset + rh > H:
                continue
            
            # Try to place in current shelf
            # The shelf has height = max(shelf_height, rh)
            # We need to find a position (x, y) in the W x D plane
            # that doesn't overlap with shelf_boxes
            
            # Try at current cursor position
            can_place = False
            test_x = x_cursor
            test_y = y_cursor
            
            if test_x + rw <= W and test_y + rd <= D:
                # Check overlap with existing shelf boxes
                no_ovl = True
                for (sx, sy, sw, sd, _) in shelf_boxes:
                    if (test_x < sx + sw and test_x + rw > sx and
                        test_y < sy + sd and test_y + rd > sy):
                        no_ovl = False
                        break
                if no_ovl:
                    can_place = True
            
            if not can_place:
                # Try next y position (new row)
                test_x = 0
                test_y = y_cursor + row_max_d
                if test_y + rd <= D and test_x + rw <= W:
                    no_ovl = True
                    for (sx, sy, sw, sd, _) in shelf_boxes:
                        if (test_x < sx + sw and test_x + rw > sx and
                            test_y < sy + sd and test_y + rd > sy):
                            no_ovl = False
                            break
                    if no_ovl:
                        can_place = True
            
            if not can_place:
                # Try a few more y positions
                for ty in range(0, D - rd + 1, max(1, rd // 3)):
                    for tx in range(0, W - rw + 1, max(1, rw // 3)):
                        no_ovl = True
                        for (sx, sy, sw, sd, _) in shelf_boxes:
                            if (tx < sx + sw and tx + rw > sx and
                                ty < sy + sd and ty + rd > sy):
                                no_ovl = False
                                break
                        if no_ovl:
                            test_x, test_y = tx, ty
                            can_place = True
                            break
                    if can_place:
                        break
            
            if can_place:
                new_shelf_height = max(shelf_height, rh)
                shelf_boxes.append((test_x, test_y, rw, rd, idx))
                placements[idx] = (0, test_x, test_y, z_offset, rw, rh, rd)
                
                # Update cursor
                if test_x == x_cursor:
                    x_cursor = test_x + rw
                    row_max_d = max(row_max_d, rd)
                else:
                    # New row
                    y_cursor = test_y
                    x_cursor = test_x + rw
                    row_max_d = rd
                
                shelf_height = new_shelf_height
                placed_this = True
                break
        
        if not placed_this:
            # Start a new shelf if possible
            new_z = z_offset + shelf_height if shelf_height > 0 else 0
            # Find best orientation that fits vertically
            best_rot = None
            for rw, rh, rd in rots:
                if new_z + rh <= H:
                    if best_rot is None or rh < best_rot[1]:
                        best_rot = (rw, rh, rd)
            
            if best_rot is not None:
                rw, rh, rd = best_rot
                # Try to place at origin of new shelf
                test_x, test_y = 0, 0
                if test_x + rw <= W and test_y + rd <= D:
                    placements[idx] = (0, test_x, test_y, new_z, rw, rh, rd)
                    z_offset = new_z
                    shelf_height = rh
                    shelf_boxes = [(test_x, test_y, rw, rd, idx)]
                    x_cursor = rw
                    y_cursor = 0
                    row_max_d = rd
                    placed_this = True
            
            if not placed_this:
                placements[idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(len(boxes)):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    volume_filled = sum(p[4] * p[5] * p[6] for p in placements if p[0] == 0)
    return placements, volume_filled

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    W, H, D = container
    container_volume = W * H * D
    n = len(boxes)
    
    # Pre-compute box properties
    volumes = [b[0] * b[1] * b[2] for b in boxes]
    max_dims = [max(b) for b in boxes]
    min_dims = [min(b) for b in boxes]
    sum_dims = [sum(b) for b in boxes]
    
    best_placements = None
    best_volume = -1
    
    # === Phase 1: Adaptive Shelving (new) ===
    if time.time() < deadline - 90:
        ash_placements, ash_vol = _adaptive_shelf_pack(boxes, container)
        if ash_vol > best_volume:
            best_volume = ash_vol
            best_placements = ash_placements
    
    # === Phase 2: EAP with diverse heuristics ===
    strategies = []
    
    vol_order = sorted(range(n), key=lambda i: volumes[i], reverse=True)
    strategies.append(('eap', vol_order, None, None))
    
    maxdim_order = sorted(range(n), key=lambda i: max_dims[i], reverse=True)
    strategies.append(('eap', maxdim_order, None, None))
    
    mindim_order = sorted(range(n), key=lambda i: min_dims[i])
    strategies.append(('eap', mindim_order, None, None))
    
    sumdim_order = sorted(range(n), key=lambda i: sum_dims[i], reverse=True)
    strategies.append(('eap', sumdim_order, None, None))
    
    def score_sum(pos, dim):
        return pos[0] + pos[1] + pos[2]
    
    def score_max(pos, dim):
        return max(pos)
    
    def score_z_sum(pos, dim):
        return (pos[2], pos[0] + pos[1])
    
    def score_weighted(pos, dim):
        # Weight towards filling corners
        w, h, d = dim
        return (pos[2] * 2 + pos[1] + pos[0])
    
    strategies.append(('eap', vol_order, score_sum, None))
    strategies.append(('eap', vol_order, score_max, None))
    strategies.append(('eap', vol_order, score_z_sum, None))
    strategies.append(('eap', vol_order, score_weighted, None))
    
    # Randomized EAP variants
    rngs = [random.Random(i) for i in range(5)]
    for r in rngs:
        strategies.append(('eap', vol_order, None, r))
    
    rng_main = random.Random(42)
    for _ in range(5):
        order = list(range(n))
        rng_main.shuffle(order)
        strategies.append(('eap', order, None, random.Random(rng_main.randint(1, 1000))))
    
    for algo, order, scorer, rng in strategies:
        if time.time() > deadline - 40:
            break
        placements, vol = _try_place_eap(boxes, container, order, deadline - 40, scorer, rng)
        
        if vol > best_volume:
            best_volume = vol
            best_placements = placements
            if best_volume >= container_volume:
                break
    
    if best_placements is None:
        # Fallback
        placements = []
        x, y, z, row_h, layer_d = 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d, x, row_h = 0, z + layer_d, 0, 0, 0
            if z + d > D:
                placements.append((-1, 0, 0, 0, w, h, d))
                continue
            placements.append((0, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        best_placements = placements
        best_volume = sum(p[4] * p[5] * p[6] for p in placements if p[0] == 0)
    
    # === Phase 3: Local Search Polish ===
    # Try to improve by re-placing unplaced boxes and moving placed boxes
    if time.time() < deadline:
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
        
        rng_ls = random.Random(999)
        ls_deadline = min(deadline, start_time + 148)
        
        # First: try to insert all unplaced boxes
        for _ in range(3):  # multiple passes
            if time.time() > ls_deadline:
                break
            improved = False
            unplaced = [i for i, p in enumerate(current_placements) if p[0] == -1]
            for idx in unplaced:
                if time.time() > ls_deadline:
                    break
                new_pl = try_insert_box(idx, current_placements, container, rng_ls)
                if new_pl:
                    current_placements[idx] = new_pl
                    current_volume += new_pl[4] * new_pl[5] * new_pl[6]
                    improved = True
            
            if not improved:
                break
        
        if current_volume > best_volume:
            best_volume = current_volume
            best_placements = [tuple(p) for p in current_placements]
        
        # Then: try removing and re-inserting placed boxes to free space
        if time.time() < ls_deadline:
            placed_indices = [i for i, p in enumerate(current_placements) if p[0] == 0]
            # Try removing smaller boxes that might be blocking
            placed_indices.sort(key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2])
            
            for _ in range(2):
                if time.time() > ls_deadline:
                    break
                for idx in placed_indices[:10]:  # try up to 10 smallest placed boxes
                    if time.time() > ls_deadline:
                        break
                    old_pl = current_placements[idx]
                    old_vol = old_pl[4] * old_pl[5] * old_pl[6]
                    
                    # Temporarily remove
                    current_placements[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
                    current_volume -= old_vol
                    
                    # Try to insert unplaced boxes first
                    inserted_any = False
                    unplaced = [i for i, p in enumerate(current_placements) if p[0] == -1 and i != idx]
                    for uidx in unplaced:
                        if time.time() > ls_deadline:
                            break
                        new_pl = try_insert_box(uidx, current_placements, container, rng_ls)
                        if new_pl:
                            current_placements[uidx] = new_pl
                            current_volume += new_pl[4] * new_pl[5] * new_pl[6]
                            inserted_any = True
                    
                    # Try to re-insert the removed box
                    new_pl = try_insert_box(idx, current_placements, container, rng_ls)
                    if new_pl:
                        current_placements[idx] = new_pl
                        current_volume += new_pl[4] * new_pl[5] * new_pl[6]
                    else:
                        # Keep it unplaced if it doesn't fit back
                        pass
                    
                    if current_volume < old_vol and not inserted_any:
                        # Revert if nothing improved
                        current_placements[idx] = old_pl
                        current_volume = old_vol + old_vol  # oops, need to recalculate
        
        # Recalculate volume properly
        current_volume = sum(p[4] * p[5] * p[6] for p in current_placements if p[0] == 0)
        if current_volume > best_volume:
            best_volume = current_volume
            best_placements = [tuple(p) for p in current_placements]
    
    return best_placements
