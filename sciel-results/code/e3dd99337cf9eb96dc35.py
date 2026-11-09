import time
import random
import itertools

def _rotations(w, h, d):
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

def _greedy(boxes, container, order, deadline, mode='extreme', scoring_func=None, rng=None):
    W, H, D = container
    placed = []
    placements = [None] * len(boxes)
    total_vol = 0.0
    n = len(boxes)
    
    if scoring_func is None:
        def scoring_func(pos, dim):
            return (pos[2], pos[1], pos[0])
    
    for idx in order:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        rots = _rotations(w, h, d)
        
        if mode == 'extreme':
            candidates = _extreme_points(placed, container)
            if (0, 0, 0) not in candidates:
                candidates.append((0, 0, 0))
        elif mode == 'faces':
            candidates = _face_points(placed, container)
        else:  # compact
            candidates = _extreme_points(placed, container)
            if (0, 0, 0) not in candidates:
                candidates.append((0, 0, 0))
        
        best_pos = None
        best_rot = None
        best_key = None
        best_candidates = []
        
        for pos in candidates:
            for rw, rh, rd in rots:
                if _fits((rw, rh, rd), pos, container, placed):
                    key = scoring_func(pos, (rw, rh, rd))
                    if best_key is None or key < best_key:
                        best_key = key
                        best_pos = pos
                        best_rot = (rw, rh, rd)
                        best_candidates = [(best_pos, best_rot)]
                    elif key == best_key:
                        best_candidates.append((pos, (rw, rh, rd)))
        
        if best_pos is not None:
            if rng and len(best_candidates) > 1:
                best_pos, best_rot = rng.choice(best_candidates)
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

def _try_insert_box(box_idx, boxes, placements, container, rng=None, shuffle_candidates=False, mode='extreme'):
    w, h, d = boxes[box_idx]
    rots = _rotations(w, h, d)
    placed = []
    for i, p in enumerate(placements):
        if p[0] == 0:
            placed.append((p[1], p[2], p[3], p[4], p[5], p[6]))
    
    if mode == 'extreme':
        candidates = _extreme_points(placed, container)
        if (0, 0, 0) not in candidates:
            candidates.append((0, 0, 0))
    else:
        candidates = _face_points(placed, container)
    
    if rng and shuffle_candidates and len(candidates) > 1:
        rng.shuffle(candidates)
    
    for pos in candidates:
        for rw, rh, rd in rots:
            if _fits((rw, rh, rd), pos, container, placed):
                return (0, pos[0], pos[1], pos[2], rw, rh, rd)
    return None

def _local_search(placements, boxes, container, deadline, rng, max_rounds=4):
    W, H, D = container
    n = len(boxes)
    current_placements = [tuple(p) for p in placements]
    current_volume = sum(p[4]*p[5]*p[6] for p in current_placements if p[0]==0)
    ls_best_placements = [tuple(p) for p in current_placements]
    ls_best_volume = current_volume
    
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        
        # Try to insert unplaced boxes
        unplaced = [i for i, p in enumerate(current_placements) if p[0] == -1]
        for idx in unplaced:
            if time.time() > deadline:
                break
            new_pl = _try_insert_box(idx, boxes, current_placements, container, rng)
            if new_pl:
                vol = new_pl[4]*new_pl[5]*new_pl[6]
                current_placements[idx] = new_pl
                current_volume += vol
                improved = True
        
        if current_volume > ls_best_volume:
            ls_best_volume = current_volume
            ls_best_placements = [tuple(p) for p in current_placements]
        
        # Try relocating placed boxes
        placed_idx = [i for i, p in enumerate(current_placements) if p[0] == 0]
        rng.shuffle(placed_idx)
        for idx in placed_idx:
            if time.time() > deadline:
                break
            
            old_pl = current_placements[idx]
            old_vol = old_pl[4]*old_pl[5]*old_pl[6]
            
            current_placements[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
            current_volume -= old_vol
            
            new_pl = _try_insert_box(idx, boxes, current_placements, container, rng, shuffle_candidates=True)
            
            if new_pl:
                new_vol = new_pl[4]*new_pl[5]*new_pl[6]
                current_placements[idx] = new_pl
                current_volume += new_vol
                improved = True
                
                if current_volume > ls_best_volume:
                    ls_best_volume = current_volume
                    ls_best_placements = [tuple(p) for p in current_placements]
            else:
                current_placements[idx] = old_pl
                current_volume += old_vol
        
        # Shake: randomly relocate a few boxes
        if not improved and len(placed_idx) > 0:
            shake_count = min(3, len(placed_idx))
            shake_indices = rng.sample(placed_idx, shake_count)
            for idx in shake_indices:
                if time.time() > deadline:
                    break
                old_pl = current_placements[idx]
                old_vol = old_pl[4]*old_pl[5]*old_pl[6]
                
                current_placements[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
                current_volume -= old_vol
                
                new_pl = _try_insert_box(idx, boxes, current_placements, container, rng, shuffle_candidates=True)
                
                if new_pl:
                    current_placements[idx] = new_pl
                    current_volume += new_pl[4]*new_pl[5]*new_pl[6]
                    if current_volume > ls_best_volume:
                        ls_best_volume = current_volume
                        ls_best_placements = [tuple(p) for p in current_placements]
                else:
                    current_placements[idx] = old_pl
                    current_volume += old_vol
        
        if time.time() > deadline:
            break
    
    return ls_best_placements, ls_best_volume

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 145.0
    W, H, D = container
    n = len(boxes)
    container_volume = W * H * D
    
    volumes = [b[0]*b[1]*b[2] for b in boxes]
    max_dims = [max(b) for b in boxes]
    min_dims = [min(b) for b in boxes]
    sum_dims = [sum(b) for b in boxes]
    surface_areas = [2*(b[0]*b[1] + b[1]*b[2] + b[0]*b[2]) for b in boxes]
    
    # Generate diverse initial solutions
    strategies = []
    
    vol_order = sorted(range(n), key=lambda i: volumes[i], reverse=True)
    strategies.append(('extreme', vol_order, None, None))
    strategies.append(('faces', vol_order, None, None))
    strategies.append(('compact', vol_order, None, None))
    
    maxdim_order = sorted(range(n), key=lambda i: max_dims[i], reverse=True)
    strategies.append(('extreme', maxdim_order, None, None))
    strategies.append(('faces', maxdim_order, None, None))
    
    mindim_order = sorted(range(n), key=lambda i: min_dims[i])
    strategies.append(('extreme', mindim_order, None, None))
    strategies.append(('faces', mindim_order, None, None))
    
    sumdim_order = sorted(range(n), key=lambda i: sum_dims[i], reverse=True)
    strategies.append(('extreme', sumdim_order, None, None))
    strategies.append(('faces', sumdim_order, None, None))
    
    sa_order = sorted(range(n), key=lambda i: surface_areas[i], reverse=True)
    strategies.append(('extreme', sa_order, None, None))
    strategies.append(('faces', sa_order, None, None))
    
    mixed1 = sorted(range(n), key=lambda i: (volumes[i], max_dims[i]), reverse=True)
    strategies.append(('extreme', mixed1, None, None))
    strategies.append(('faces', mixed1, None, None))
    
    def score_sum(pos, dim):
        return pos[0] + pos[1] + pos[2]
        
    def score_max(pos, dim):
        return max(pos)
        
    def score_z_sum(pos, dim):
        return (pos[2], pos[0] + pos[1])
        
    def score_compact(pos, dim):
        return (pos[2] + dim[2], pos[1] + dim[1], pos[0] + dim[0])
        
    strategies.append(('extreme', vol_order, score_sum, None))
    strategies.append(('faces', vol_order, score_sum, None))
    strategies.append(('compact', vol_order, score_compact, None))
    strategies.append(('extreme', vol_order, score_max, None))
    strategies.append(('faces', vol_order, score_max, None))
    strategies.append(('extreme', vol_order, score_z_sum, None))
    strategies.append(('faces', vol_order, score_z_sum, None))
    
    rngs = [random.Random(i) for i in range(8)]
    for r in rngs:
        strategies.append(('extreme', vol_order, None, r))
        strategies.append(('faces', vol_order, None, r))
        
    rng_main = random.Random(42)
    for _ in range(12):
        order = list(range(n))
        rng_main.shuffle(order)
        strategies.append(('extreme', order, None, random.Random(rng_main.randint(1, 1000))))
        strategies.append(('faces', order, None, random.Random(rng_main.randint(1, 1000))))

    best_placements = None
    best_volume = -1
    
    for algo, order, scorer, rng in strategies:
        if time.time() > deadline - 50:
            break
        placements, vol = _greedy(boxes, container, order, deadline - 50, algo, scorer, rng)
        
        if vol > best_volume:
            best_volume = vol
            best_placements = placements
            if best_volume >= container_volume:
                break
    
    if best_placements is None:
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
        best_volume = sum(p[4]*p[5]*p[6] for p in placements if p[0]==0)

    # Run multiple local searches with different random seeds
    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        if time.time() > deadline - 3:
            break
        rng_ls = random.Random(seed)
        ls_placements, ls_volume = _local_search(best_placements, boxes, container, deadline, rng_ls, max_rounds=3)
        
        if ls_volume > best_volume:
            best_volume = ls_volume
            best_placements = ls_placements
            if best_volume >= container_volume:
                break
    
    # Final polish: try inserting any remaining unplaced boxes one more time
    unplaced_final = [i for i, p in enumerate(best_placements) if p[0] == -1]
    for idx in unplaced_final:
        if time.time() > deadline - 1:
            break
        new_pl = _try_insert_box(idx, boxes, best_placements, container)
        if new_pl:
            best_placements[idx] = new_pl
            best_volume += new_pl[4]*new_pl[5]*new_pl[6]
    
    return best_placements
