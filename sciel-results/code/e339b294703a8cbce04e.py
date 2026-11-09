import time
import random
import itertools
import math

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
    """
    placed: list of (x, y, z, w, h, d)
    """
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

def _greedy_init(boxes, container, order_idx, deadline):
    """
    Greedy extreme point approach to get a valid initial state.
    Returns:
    placed: list of (idx, x, y, z, w, h, d)
    placements: list of tuples (bin, x, y, z, w, h, d) for all boxes
    volume: total volume
    """
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
                score = (cz, cy, cx)
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
                    best_dim = (rw, rh, rd)
        
        if best_pos is not None:
            placed.append((orig_idx, best_pos[0], best_pos[1], best_pos[2],
                          best_dim[0], best_dim[1], best_dim[2]))
            placements[orig_idx] = (0, best_pos[0], best_pos[1], best_pos[2],
                                   best_dim[0], best_dim[1], best_dim[2])
        else:
            placements[orig_idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(len(boxes)):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
            
    volume = sum(p[4] * p[5] * p[6] for p in placements if p[0] == 0)
    return placed, placements, volume

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 145 # Safety margin for 150s limit
    
    W, H, D = container
    container_volume = W * H * D
    n = len(boxes)
    
    # 1. Initialize with a good greedy heuristic
    # Sort by volume descending, then by max dimension
    vol_order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    # Try a few different greedy orders to find a strong start
    best_init_placed = None
    best_init_placements = None
    best_init_vol = -1
    
    orders = [
        vol_order,
        sorted(range(n), key=lambda i: max(boxes[i]), reverse=True),
        sorted(range(n), key=lambda i: min(boxes[i])),
    ]
    
    rng = random.Random(42)
    for _ in range(3):
        o = list(range(n))
        rng.shuffle(o)
        orders.append(o)
        
    for order in orders:
        if time.time() > deadline - 10:
            break
        placed, placements, vol = _greedy_init(boxes, container, order, deadline - 10)
        if vol > best_init_vol:
            best_init_vol = vol
            best_init_placed = placed
            best_init_placements = placements
            
    if best_init_placed is None:
        # Fallback: empty
        best_init_placed = []
        best_init_placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        best_init_vol = 0

    # 2. Local Search / Simulated Annealing
    current_placed = list(best_init_placed)
    current_placements = list(best_init_placements)
    current_vol = best_init_vol
    
    best_placed = list(current_placed)
    best_placements = list(current_placements)
    best_vol = current_vol
    
    # Temperature parameters
    T = 100.0
    cooling = 0.995
    min_T = 0.1
    
    placed_indices = set([p[0] for p in current_placed])
    unplaced_indices = list(set(range(n)) - placed_indices)
    
    # Pre-compute rotations for speed
    box_rots = [_rotations(b[0], b[1], b[2]) for b in boxes]
    
    while time.time() < deadline - 5 and T > min_T:
        # Pick a move
        # 50% chance: remove a placed box
        # 50% chance: try to insert an unplaced box (if any)
        
        if unplaced_indices and (not current_placed or rng.random() < 0.5):
            # Try to insert a random unplaced box
            idx = rng.choice(unplaced_indices)
            w, h, d = boxes[idx]
            rots = box_rots[idx]
            
            # Generate candidate positions from extreme points of current state
            candidates = _extreme_points([(p[1], p[2], p[3], p[4], p[5], p[6]) for p in current_placed], container)
            
            best_pos = None
            best_dim = None
            best_score = None
            
            for (cx, cy, cz) in candidates:
                for (rw, rh, rd) in rots:
                    if not _fits((rw, rh, rd), (cx, cy, cz), container):
                        continue
                    if not _no_overlap((rw, rh, rd), (cx, cy, cz), [(p[1], p[2], p[3], p[4], p[5], p[6]) for p in current_placed]):
                        continue
                    score = (cz, cy, cx)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pos = (cx, cy, cz)
                        best_dim = (rw, rh, rd)
            
            if best_pos is not None:
                # Calculate new volume
                new_vol = current_vol + w * h * d
                delta_vol = new_vol - current_vol
                
                if delta_vol > 0 or rng.random() < math.exp(delta_vol / T):
                    # Accept move
                    new_entry = (idx, best_pos[0], best_pos[1], best_pos[2], best_dim[0], best_dim[1], best_dim[2])
                    current_placed.append(new_entry)
                    current_placements[idx] = (0, best_pos[0], best_pos[1], best_pos[2], best_dim[0], best_dim[1], best_dim[2])
                    current_vol = new_vol
                    placed_indices.add(idx)
                    unplaced_indices.remove(idx)
                    
                    if current_vol > best_vol:
                        best_vol = current_vol
                        best_placed = list(current_placed)
                        best_placements = list(current_placements)
                else:
                    # Rejected, do nothing
                    pass
            else:
                # Could not place, do nothing
                pass
                
        else:
            # Remove a random placed box
            if not current_placed:
                T *= cooling
                continue
                
            idx_to_remove = rng.choice(current_placed)
            # Find index in list to remove it properly
            remove_idx = -1
            for i, p in enumerate(current_placed):
                if p[0] == idx_to_remove[0]:
                    remove_idx = i
                    break
            if remove_idx != -1:
                current_placed.pop(remove_idx)
                # Update placements
                current_placements[idx_to_remove[0]] = (-1, 0, 0, 0, boxes[idx_to_remove[0]][0], boxes[idx_to_remove[0]][1], boxes[idx_to_remove[0]][2])
                # Update volume
                current_vol -= idx_to_remove[4] * idx_to_remove[5] * idx_to_remove[6]
                placed_indices.remove(idx_to_remove[0])
                unplaced_indices.append(idx_to_remove[0])
                
                # Always accept removal in SA context (it's a step towards exploration)
                # But we don't update best here as it's worse
                
        T *= cooling
        
    # Final cleanup: ensure all boxes are accounted for in placements
    # The local search modifies current_placements directly, so it should be fine.
    # But let's double check any None or missing entries just in case.
    for i in range(n):
        if current_placements[i] is None:
            w, h, d = boxes[i]
            current_placements[i] = (-1, 0, 0, 0, w, h, d)
            
    # Return the best found so far
    return best_placements
