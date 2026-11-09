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
    """Check if box_dims at pos overlaps with any box in placed list."""
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

def _greedy_seed(boxes, container, order_idx, deadline):
    """Generate a greedy placement as starting point for SA."""
    W, H, D = container
    placed = []
    placements = [None] * len(boxes)
    
    for orig_idx in order_idx:
        if time.time() > deadline:
            break
        w, h, d = boxes[orig_idx]
        rots = _rotations(w, h, d)
        candidates = _extreme_points(placed, container)
        
        best_pos = None
        best_dim = None
        best_score = None
        
        for (cx, cy, cz) in candidates:
            for (rw, rh, rd) in rots:
                if not _fits((rw, rh, rd), (cx, cy, cz), container):
                    continue
                if not _no_overlap((rw, rh, rd), (cx, cy, cz), placed):
                    continue
                
                # Score: prefer lower z, then lower y, then lower x
                score = (cz, cy, cx)
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
                    best_dim = (rw, rh, rd)
        
        if best_pos is not None:
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

def _simulated_annealing(boxes, container, initial_placements, deadline, rng):
    """
    Simulated Annealing to improve a greedy placement.
    Moves: 
      1. Move a placed box to a new valid position.
      2. Swap positions of two placed boxes.
      3. Try to place an unplaced box.
    Objective: Maximize total volume filled.
    """
    W, H, D = container
    n = len(boxes)
    
    # State: list of (x, y, z, w, h, d) for each box, None if unplaced
    # We maintain a list of placed boxes for fast overlap checking
    # state[i] = (x, y, z, w, h, d) or None
    state = [None] * n
    placed_boxes = []  # List of (x, y, z, w, h, d)
    
    for i in range(n):
        p = initial_placements[i]
        if p[0] == 0:
            state[i] = (p[1], p[2], p[3], p[4], p[5], p[6])
            placed_boxes.append((p[1], p[2], p[3], p[4], p[5], p[6]))
    
    current_volume = sum(b[3]*b[4]*b[5] for b in placed_boxes)
    best_volume = current_volume
    best_state = [s[:] if s else None for s in state]
    
    # Cooling schedule
    T = 100.0  # Initial temperature
    alpha = 0.995  # Cooling rate
    min_T = 1e-6
    
    start_time = time.time()
    
    while T > min_T:
        if time.time() > deadline:
            break
            
        # Choose a random move
        move_type = rng.choice(['move', 'swap', 'insert'])
        
        if move_type == 'move':
            # Pick a random placed box and move it
            if not placed_boxes:
                continue
            idx = rng.randint(0, n-1)
            if state[idx] is None:
                continue
            
            old_x, old_y, old_z, old_w, old_h, old_d = state[idx]
            
            # Generate a new candidate position
            # Use extreme points from current placed boxes (excluding the one being moved)
            temp_placed = [b for b in placed_boxes if b != (old_x, old_y, old_z, old_w, old_h, old_d)]
            candidates = _extreme_points(temp_placed, container)
            
            if not candidates:
                candidates = [(0, 0, 0)]
            
            # Try a few random candidates
            rng.shuffle(candidates)
            new_pos = None
            new_dim = None
            
            # Try original orientation first, then others
            rots = _rotations(old_w, old_h, old_d)
            for (cx, cy, cz) in candidates[:10]:  # Limit candidates for speed
                for (rw, rh, rd) in rots:
                    if not _fits((rw, rh, rd), (cx, cy, cz), container):
                        continue
                    # Check overlap with other placed boxes
                    if _no_overlap((rw, rh, rd), (cx, cy, cz), temp_placed):
                        new_pos = (cx, cy, cz)
                        new_dim = (rw, rh, rd)
                        break
                if new_pos:
                    break
            
            if new_pos is None:
                # Try to find any valid spot
                for (cx, cy, cz) in candidates:
                    for (rw, rh, rd) in rots:
                        if not _fits((rw, rh, rd), (cx, cy, cz), container):
                            continue
                        if _no_overlap((rw, rh, rd), (cx, cy, cz), temp_placed):
                            new_pos = (cx, cy, cz)
                            new_dim = (rw, rh, rd)
                            break
                    if new_pos:
                        break
            
            if new_pos is None:
                continue
            
            new_x, new_y, new_z = new_pos
            new_w, new_h, new_d = new_dim
            
            # Calculate volume change
            vol_change = 0  # Same box, just moved
            
            # Acceptance
            if vol_change >= 0 or rng.random() < pow(2.718281828, vol_change / T):
                # Apply move
                placed_boxes.remove((old_x, old_y, old_z, old_w, old_h, old_d))
                new_box = (new_x, new_y, new_z, new_w, new_h, new_d)
                placed_boxes.append(new_box)
                state[idx] = new_box
                current_volume += vol_change
                
                if current_volume > best_volume:
                    best_volume = current_volume
                    best_state = [s[:] if s else None for s in state]
        
        elif move_type == 'swap':
            # Swap two placed boxes
            if len(placed_boxes) < 2:
                continue
            
            # Pick two random placed boxes
            idx1, idx2 = rng.sample(range(n), 2)
            if state[idx1] is None or state[idx2] is None:
                continue
            
            box1 = state[idx1]
            box2 = state[idx2]
            
            # Try swapping their positions (keep orientations)
            # Check if box1 fits at box2's position
            pos1 = (box1[0], box1[1], box1[2])
            pos2 = (box2[0], box2[1], box2[2])
            dim1 = (box1[3], box1[4], box1[5])
            dim2 = (box2[3], box2[4], box2[5])
            
            # Remove both from placed list
            temp_placed = []
            for b in placed_boxes:
                if b != box1 and b != box2:
                    temp_placed.append(b)
            
            # Check if swap is valid
            if (_fits(dim1, pos2, container) and _no_overlap(dim1, pos2, temp_placed) and
                _fits(dim2, pos1, container) and _no_overlap(dim2, pos1, temp_placed)):
                
                vol_change = 0  # Same total volume
                
                if vol_change >= 0 or rng.random() < pow(2.718281828, vol_change / T):
                    # Apply swap
                    placed_boxes.remove(box1)
                    placed_boxes.remove(box2)
                    new_box1 = (pos2[0], pos2[1], pos2[2], dim1[0], dim1[1], dim1[2])
                    new_box2 = (pos1[0], pos1[1], pos1[2], dim2[0], dim2[1], dim2[2])
                    placed_boxes.append(new_box1)
                    placed_boxes.append(new_box2)
                    state[idx1] = new_box1
                    state[idx2] = new_box2
                    current_volume += vol_change
                    
                    if current_volume > best_volume:
                        best_volume = current_volume
                        best_state = [s[:] if s else None for s in state]
        
        elif move_type == 'insert':
            # Try to insert an unplaced box
            unplaced = [i for i in range(n) if state[i] is None]
            if not unplaced:
                continue
            
            idx = rng.choice(unplaced)
            w, h, d = boxes[idx]
            rots = _rotations(w, h, d)
            
            candidates = _extreme_points(placed_boxes, container)
            if not candidates:
                candidates = [(0, 0, 0)]
            
            rng.shuffle(candidates)
            
            for (cx, cy, cz) in candidates[:20]:
                for (rw, rh, rd) in rots:
                    if not _fits((rw, rh, rd), (cx, cy, cz), container):
                        continue
                    if _no_overlap((rw, rh, rd), (cx, cy, cz), placed_boxes):
                        # Insert
                        new_box = (cx, cy, cz, rw, rh, rd)
                        placed_boxes.append(new_box)
                        state[idx] = new_box
                        vol_change = rw * rh * rd
                        current_volume += vol_change
                        
                        if current_volume > best_volume:
                            best_volume = current_volume
                            best_state = [s[:] if s else None for s in state]
                        
                        break
                else:
                    continue
                break
        
        T *= alpha
    
    # Convert best_state to placements format
    placements = []
    for i in range(n):
        if best_state[i] is not None:
            x, y, z, w, h, d = best_state[i]
            placements.append((0, x, y, z, w, h, d))
        else:
            w, h, d = boxes[i]
            placements.append((-1, 0, 0, 0, w, h, d))
    
    return placements, best_volume

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    W, H, D = container
    container_volume = W * H * D
    n = len(boxes)
    
    # Pre-compute box properties
    volumes = [b[0]*b[1]*b[2] for b in boxes]
    max_dims = [max(b) for b in boxes]
    
    # Generate multiple greedy seeds with different orderings
    seeds = []
    
    # Volume-based order
    vol_order = sorted(range(n), key=lambda i: volumes[i], reverse=True)
    placements, vol = _greedy_seed(boxes, container, vol_order, deadline)
    seeds.append((placements, vol))
    
    # Max dimension order
    maxdim_order = sorted(range(n), key=lambda i: max_dims[i], reverse=True)
    placements, vol = _greedy_seed(boxes, container, maxdim_order, deadline)
    seeds.append((placements, vol))
    
    # Random orders
    rng_main = random.Random(42)
    for _ in range(3):
        order = list(range(n))
        rng_main.shuffle(order)
        placements, vol = _greedy_seed(boxes, container, order, deadline)
        seeds.append((placements, vol))
    
    # Run SA on the best seeds
    best_placements = None
    best_volume = -1
    
    # Sort seeds by volume, try best first
    seeds.sort(key=lambda x: x[1], reverse=True)
    
    for placements, vol in seeds:
        if time.time() > deadline - 10:  # Leave time for final check
            break
        
        sa_placements, sa_vol = _simulated_annealing(boxes, container, placements, deadline, random.Random())
        
        if sa_vol > best_volume:
            best_volume = sa_vol
            best_placements = sa_placements
            if best_volume >= container_volume:
                break
    
    # If no SA result, use best greedy
    if best_placements is None:
        best_placements = seeds[0][0]
    
    return best_placements
