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

def _no_overlap_grid(box_dims, pos, grid, container):
    """Check overlap using spatial hash grid."""
    W, H, D = container
    x, y, z = pos
    w, h, d = box_dims
    
    # Check cells that the new box would occupy
    min_cx = max(0, int(x // 1))
    max_cx = int((x + w) // 1)
    min_cy = max(0, int(y // 1))
    max_cy = int((y + h) // 1)
    min_cz = max(0, int(z // 1))
    max_cz = int((z + d) // 1)
    
    for cx in range(min_cx, max_cx + 1):
        for cy in range(min_cy, max_cy + 1):
            for cz in range(min_cz, max_cz + 1):
                key = (cx, cy, cz)
                if key in grid:
                    for (px, py, pz, pw, ph, pd) in grid[key]:
                        if (x < px + pw and x + w > px and
                            y < py + ph and y + h > py and
                            z < pz + pd and z + d > pz):
                            return False
    return True

def _add_to_grid(placed, grid, box):
    """Add a placed box to the spatial hash grid."""
    px, py, pz, pw, ph, pd = box
    min_cx = max(0, int(px // 1))
    max_cx = int((px + pw) // 1)
    min_cy = max(0, int(py // 1))
    max_cy = int((py + ph) // 1)
    min_cz = max(0, int(pz // 1))
    max_cz = int((pz + pd) // 1)
    
    for cx in range(min_cx, max_cx + 1):
        for cy in range(min_cy, max_cy + 1):
            for cz in range(min_cz, max_cz + 1):
                key = (cx, cy, cz)
                if key not in grid:
                    grid[key] = []
                grid[key].append(box)

def _remove_from_grid(grid, box):
    """Remove a box from the spatial hash grid."""
    px, py, pz, pw, ph, pd = box
    min_cx = max(0, int(px // 1))
    max_cx = int((px + pw) // 1)
    min_cy = max(0, int(py // 1))
    max_cy = int((py + ph) // 1)
    min_cz = max(0, int(pz // 1))
    max_cz = int((pz + pd) // 1)
    
    for cx in range(min_cx, max_cx + 1):
        for cy in range(min_cy, max_cy + 1):
            for cz in range(min_cz, max_cz + 1):
                key = (cx, cy, cz)
                if key in grid:
                    grid[key] = [b for b in grid[key] if b != box]
                    if not grid[key]:
                        del grid[key]

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

def _candidate_positions(placed, container, grid):
    """Generate candidate positions: extreme points plus grid-aligned positions."""
    W, H, D = container
    candidates = set()
    
    # Add extreme points
    for (cx, cy, cz) in _extreme_points(placed, container):
        candidates.add((cx, cy, cz))
    
    # Add positions at each x, y, z boundary of placed boxes
    xs = {0, W}
    ys = {0, H}
    zs = {0, D}
    for (px, py, pz, pw, ph, pd) in placed:
        xs.add(px)
        xs.add(min(W, px + pw))
        ys.add(py)
        ys.add(min(H, py + ph))
        zs.add(pz)
        zs.add(min(D, pz + pd))
    
    # Sample some combinations to keep it manageable
    if len(xs) * len(ys) * len(zs) <= 2000:
        for x in xs:
            for y in ys:
                for z in zs:
                    candidates.add((x, y, z))
    else:
        # Fallback: just use extreme points and a few sampled positions
        rng = random.Random(12345)
        for _ in range(100):
            x = rng.choice(list(xs))
            y = rng.choice(list(ys))
            z = rng.choice(list(zs))
            candidates.add((x, y, z))
    
    return list(candidates)

def _try_place_eap_fast(boxes, container, order_idx, deadline):
    """Extreme Points Approach with spatial hash grid for fast overlap checking."""
    W, H, D = container
    placed = []
    grid = {}
    placements = [None] * len(boxes)
    
    for orig_idx in order_idx:
        if time.time() > deadline:
            break
        w, h, d = boxes[orig_idx]
        rots = _rotations(w, h, d)
        candidates = _candidate_positions(placed, container, grid)
        
        best_pos = None
        best_dim = None
        best_score = None
        
        for (cx, cy, cz) in candidates:
            for (rw, rh, rd) in rots:
                if not _fits((rw, rh, rd), (cx, cy, cz), container):
                    continue
                if not _no_overlap_grid((rw, rh, rd), (cx, cy, cz), grid, container):
                    continue
                score = (cz, cy, cx)
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
                    best_dim = (rw, rh, rd)
        
        if best_pos is not None:
            box = (best_pos[0], best_pos[1], best_pos[2],
                   best_dim[0], best_dim[1], best_dim[2])
            placed.append(box)
            _add_to_grid(placed, grid, box)
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

def _try_place_shelf_fast(boxes, container, order_idx, deadline):
    """Shelf packing: pack boxes into horizontal shelves."""
    W, H, D = container
    placements = [None] * len(boxes)
    
    # Track current shelf state
    shelf_x = 0
    shelf_y = 0
    shelf_z = 0
    shelf_h = 0  # height of current shelf
    shelf_d = 0  # depth of current shelf
    
    for orig_idx in order_idx:
        if time.time() > deadline:
            break
        w, h, d = boxes[orig_idx]
        rots = _rotations(w, h, d)
        
        best_pos = None
        best_dim = None
        best_score = None
        
        for (rw, rh, rd) in rots:
            # Try placing in current shelf
            if shelf_x + rw <= W and shelf_h >= rh and shelf_d >= rd:
                pos = (shelf_x, 0, shelf_z)
                if best_score is None or (rh, rd) < (best_dim[1], best_dim[2]):
                    best_score = (rh, rd)
                    best_pos = pos
                    best_dim = (rw, rh, rd)
            
            # Try starting a new shelf
            if shelf_x + rw <= W:
                new_shelf_y = shelf_y + shelf_h
                if new_shelf_y + rh <= H and shelf_d >= rd:
                    pos = (shelf_x, new_shelf_y, shelf_z)
                    if best_score is None or (rh, rd) < (best_dim[1], best_dim[2]):
                        best_score = (rh, rd)
                        best_pos = pos
                        best_dim = (rw, rh, rd)
            
            # Try starting a new layer
            if rw <= W and rh <= H and rd <= D:
                new_shelf_z = shelf_z + shelf_d
                if new_shelf_z + rd <= D:
                    pos = (0, 0, new_shelf_z)
                    if best_score is None or (rh, rd) < (best_dim[1], best_dim[2]):
                        best_score = (rh, rd)
                        best_pos = pos
                        best_dim = (rw, rh, rd)
        
        if best_pos is not None:
            placements[orig_idx] = (0, best_pos[0], best_pos[1], best_pos[2],
                                   best_dim[0], best_dim[1], best_dim[2])
            # Update shelf state
            if best_pos[2] > shelf_z:
                # New layer
                shelf_z = best_pos[2]
                shelf_x = best_dim[0]
                shelf_y = 0
                shelf_h = best_dim[1]
                shelf_d = best_dim[2]
            elif best_pos[1] > shelf_y:
                # New shelf in same layer
                shelf_y = best_pos[1]
                shelf_x = best_dim[0]
                shelf_h = best_dim[1]
                shelf_d = max(shelf_d, best_dim[2])
            else:
                # Same shelf
                shelf_x += best_dim[0]
                shelf_h = max(shelf_h, best_dim[1])
                shelf_d = max(shelf_d, best_dim[2])
        else:
            placements[orig_idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(len(boxes)):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    volume_filled = sum(p[4] * p[5] * p[6] for p in placements if p[0] == 0)
    return placements, volume_filled

def _local_improve(placements, boxes, container, deadline):
    """Try to improve a solution by moving boxes to better positions."""
    W, H, D = container
    n = len(boxes)
    placed_indices = [i for i in range(n) if placements[i][0] == 0]
    
    # Build list of placed boxes
    placed_boxes = []
    for i in placed_indices:
        p = placements[i]
        placed_boxes.append((i, p[1], p[2], p[3], p[4], p[5], p[6]))
    
    # Try to move each box to a better position
    improved = False
    for iter_num in range(3):  # Limit iterations
        if time.time() > deadline:
            break
        for i in placed_indices:
            if time.time() > deadline:
                break
            p = placements[i]
            w, h, d = p[4], p[5], p[6]
            rots = _rotations(w, h, d)
            
            # Get other placed boxes
            other_boxes = [b for b in placed_boxes if b[0] != i]
            
            # Generate candidate positions
            candidates = {(0, 0, 0)}
            for (j, px, py, pz, pw, ph, pd) in other_boxes:
                for (cx, cy, cz) in [
                    (px + pw, py, pz),
                    (px, py + ph, pz),
                    (px, py, pz + pd),
                    (px + pw, py + ph, pz),
                    (px, py + ph, pz + pd),
                    (px + pw, py, pz + pd),
                    (px + pw, py + ph, pz + pd),
                ]:
                    if cx <= W and cy <= H and cz <= D:
                        candidates.add((cx, cy, cz))
            
            best_pos = None
            best_dim = None
            best_score = None
            
            for (cx, cy, cz) in candidates:
                for (rw, rh, rd) in rots:
                    if not _fits((rw, rh, rd), (cx, cy, cz), container):
                        continue
                    # Check overlap with other boxes
                    overlap = False
                    for (j, px, py, pz, pw, ph, pd) in other_boxes:
                        if (cx < px + pw and cx + rw > px and
                            cy < py + ph and cy + rh > py and
                            cz < pz + pd and cz + rd > pz):
                            overlap = True
                            break
                    if overlap:
                        continue
                    score = (cz, cy, cx)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pos = (cx, cy, cz)
                        best_dim = (rw, rh, rd)
            
            if best_pos is not None and best_dim is not None:
                if (best_pos[0], best_pos[1], best_pos[2], best_dim[0], best_dim[1], best_dim[2]) != (p[1], p[2], p[3], p[4], p[5], p[6]):
                    placements[i] = (0, best_pos[0], best_pos[1], best_pos[2],
                                    best_dim[0], best_dim[1], best_dim[2])
                    # Update placed_boxes
                    placed_boxes = [(j, pl[1], pl[2], pl[3], pl[4], pl[5], pl[6]) 
                                   for j, pl in enumerate(placements) if pl[0] == 0]
                    improved = True
    
    return placements, improved

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    W, H, D = container
    container_volume = W * H * D
    n = len(boxes)
    
    strategies = []
    
    # Volume-based ordering
    vol_order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    strategies.append(('eap_fast', vol_order))
    
    # Max dimension ordering
    maxdim_order = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    strategies.append(('eap_fast', maxdim_order))
    
    # Min dimension ordering
    mindim_order = sorted(range(n), key=lambda i: min(boxes[i]))
    strategies.append(('eap_fast', mindim_order))
    
    # Sum dimension ordering
    sumdim_order = sorted(range(n), key=lambda i: sum(boxes[i]), reverse=True)
    strategies.append(('eap_fast', sumdim_order))
    
    # Surface area ordering
    sa_order = sorted(range(n), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)
    strategies.append(('eap_fast', sa_order))
    
    # Shelf-based strategies
    strategies.append(('shelf_fast', vol_order))
    strategies.append(('shelf_fast', maxdim_order))
    
    # Random orderings
    rng = random.Random(42)
    for _ in range(6):
        order = list(range(n))
        rng.shuffle(order)
        strategies.append(('eap_fast', order))
    
    best_placements = None
    best_volume = -1
    
    for algo, order in strategies:
        if time.time() > deadline:
            break
        if algo == 'eap_fast':
            placements, vol = _try_place_eap_fast(boxes, container, order, deadline)
        elif algo == 'shelf_fast':
            placements, vol = _try_place_shelf_fast(boxes, container, order, deadline)
        else:
            continue
        
        if vol > best_volume:
            best_volume = vol
            best_placements = placements
            if best_volume >= container_volume:
                break
    
    if best_placements is None:
        # Fallback: simple sequential packing
        placements = []
        x, y, z = 0, 0, 0
        row_h = 0
        layer_d = 0
        for w, h, d in boxes:
            if x + w > W:
                x = 0
                y += row_h
                row_h = 0
            if y + h > H:
                y = 0
                z += layer_d
                layer_d = 0
                x = 0
                row_h = 0
            if z + d > D:
                placements.append((-1, 0, 0, 0, w, h, d))
                continue
            placements.append((0, x, y, z, w, h, d))
            x += w
            row_h = max(row_h, h)
            layer_d = max(layer_d, d)
        best_placements = placements
    
    # Try local improvement if we have time
    if time.time() < deadline - 10 and best_placements is not None:
        best_placements, _ = _local_improve(best_placements, boxes, container, deadline)
        best_volume = sum(p[4] * p[5] * p[6] for p in best_placements if p[0] == 0)
    
    return best_placements
