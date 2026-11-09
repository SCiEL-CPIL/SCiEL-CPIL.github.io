"""Rewritten 3D container loading with multi-strategy greedy + local search."""
import time
import random
from itertools import permutations

# EVOLVE-BLOCK-START

def _rotations(w, h, d):
    """Return unique rotations of a box as (w, h, d) tuples."""
    seen = set()
    result = []
    for perm in permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result


def _fits_at(x, y, z, bw, bh, bd, placed, W, H, D):
    """Check if a box of size (bw,bh,bd) fits at (x,y,z) without overlap."""
    if x < 0 or y < 0 or z < 0:
        return False
    if x + bw > W or y + bh > H or z + bd > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        if not (x + bw <= px or px + pw <= x):
            if not (y + bh <= py or py + ph <= y):
                if not (z + bd <= pz or pz + pd <= z):
                    return False
    return True


def _candidate_positions(placed, W, H, D):
    """Generate candidate positions: origin and corners of placed boxes."""
    candidates = [(0, 0, 0)]
    for (px, py, pz, pw, ph, pd) in placed:
        for dx in (pw,):
            for dy in (ph,):
                for dz in (pd,):
                    nx, ny, nz = px + dx, py + dy, pz + dz
                    if nx < W and ny < H and nz < D:
                        candidates.append((nx, ny, nz))
    # Also add positions adjacent to each face
    for (px, py, pz, pw, ph, pd) in placed:
        # x+ face
        if px + pw < W:
            candidates.append((px + pw, py, pz))
            candidates.append((px + pw, py + ph, pz))
            candidates.append((px + pw, py, pz + pd))
            candidates.append((px + pw, py + ph, pz + pd))
        # y+ face
        if py + ph < H:
            candidates.append((px, py + ph, pz))
            candidates.append((px + pw, py + ph, pz))
            candidates.append((px, py + ph, pz + pd))
            candidates.append((px + pw, py + ph, pz + pd))
        # z+ face
        if pz + pd < D:
            candidates.append((px, py, pz + pd))
            candidates.append((px + pw, py, pz + pd))
            candidates.append((px, py + ph, pz + pd))
            candidates.append((px + pw, py + ph, pz + pd))
    # Deduplicate
    return list(set(candidates))


def _greedy_fill(boxes, container, strategy, rng):
    """
    Greedy fill with given strategy.
    strategy: 'largest_first', 'smallest_first', 'random', 'width_first', 'height_first', 'depth_first'
    Returns (placements_list, placed_boxes) where placed_boxes is list of (box_idx, x, y, z, bw, bh, bd)
    """
    W, H, D = container
    n = len(boxes)
    
    # Create list of (box_index, w, h, d)
    box_list = [(i, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    
    # Sort based on strategy
    if strategy == 'largest_first':
        box_list.sort(key=lambda b: -(b[1]*b[2]*b[3]))
    elif strategy == 'smallest_first':
        box_list.sort(key=lambda b: (b[1]*b[2]*b[3]))
    elif strategy == 'width_first':
        box_list.sort(key=lambda b: -b[1])
    elif strategy == 'height_first':
        box_list.sort(key=lambda b: -b[2])
    elif strategy == 'depth_first':
        box_list.sort(key=lambda b: -b[3])
    else:  # random
        rng.shuffle(box_list)
    
    placed = []  # (x, y, z, w, h, d)
    result = [None] * n  # Will be filled with (bin, x, y, z, w, h, d)
    
    for idx, w, h, d in box_list:
        rots = _rotations(w, h, d)
        best_pos = None
        best_vol = 0
        candidates = _candidate_positions(placed, W, H, D)
        
        for (bw, bh, bd) in rots:
            for (cx, cy, cz) in candidates:
                if _fits_at(cx, cy, cz, bw, bh, bd, placed, W, H, D):
                    # Prefer positions closer to origin for better packing
                    score = cx + cy + cz
                    if best_pos is None or score < best_vol:
                        best_pos = (cx, cy, cz, bw, bh, bd)
                        best_vol = score
        
        if best_pos:
            cx, cy, cz, bw, bh, bd = best_pos
            placed.append((cx, cy, cz, bw, bh, bd))
            result[idx] = (0, cx, cy, cz, bw, bh, bd)
        else:
            result[idx] = (-1, 0, 0, 0, w, h, d)
    
    return result, placed


def _local_search(placements, boxes, container, rng, max_iters=50):
    """
    Try to improve by moving boxes or swapping.
    placements: list of (bin, x, y, z, w, h, d)
    Returns improved placements or original if no improvement.
    """
    W, H, D = container
    n = len(boxes)
    current_vol = sum(p[4]*p[5]*p[6] for p in placements if p[0] == 0)
    best_placements = placements[:]
    best_vol = current_vol
    
    for _ in range(max_iters):
        improved = False
        # Try moving a random loaded box to a new position
        loaded = [i for i in range(n) if placements[i][0] == 0]
        if not loaded:
            break
        idx = rng.choice(loaded)
        old = placements[idx]
        bw, bh, bd = old[4], old[5], old[6]
        
        # Remove box
        placed = [p for i, p in enumerate(placements) if i != idx and p[0] == 0]
        
        # Try to place it at a better position
        candidates = _candidate_positions(placed, W, H, D)
        rots = _rotations(bw, bh, bd)
        best_new = None
        best_score = old[1] + old[2] + old[3]  # current position score
        
        for (rw, rh, rd) in rots:
            for (cx, cy, cz) in candidates:
                if _fits_at(cx, cy, cz, rw, rh, rd, placed, W, H, D):
                    score = cx + cy + cz
                    if score < best_score:
                        best_new = (0, cx, cy, cz, rw, rh, rd)
                        best_score = score
        
        if best_new:
            new_placements = placements[:]
            new_placements[idx] = best_new
            # Check if this allows another unplaced box to fit
            unplaced = [i for i in range(n) if new_placements[i][0] == -1]
            for uidx in unplaced:
                uw, uh, ud = boxes[uidx]
                urots = _rotations(uw, uh, ud)
                placed_now = [p for p in new_placements if p[0] == 0]
                ucands = _candidate_positions(placed_now, W, H, D)
                for (uw2, uh2, ud2) in urots:
                    for (cx, cy, cz) in ucands:
                        if _fits_at(cx, cy, cz, uw2, uh2, ud2, placed_now, W, H, D):
                            new_placements[uidx] = (0, cx, cy, cz, uw2, uh2, ud2)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if improved:
                best_placements = new_placements
                best_vol = sum(p[4]*p[5]*p[6] for p in best_placements if p[0] == 0)
            else:
                # Still accept if position is better (for future improvements)
                if best_new[1] + best_new[2] + best_new[3] < old[1] + old[2] + old[3]:
                    best_placements = new_placements
        else:
            # Could not move, keep original
            pass
    
    return best_placements


def load(boxes, container):
    W, H, D = container
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    n = len(boxes)
    rng = random.Random(42)
    
    strategies = [
        'largest_first', 'smallest_first', 'width_first', 
        'height_first', 'depth_first', 'random'
    ]
    
    best_result = None
    best_vol = -1
    
    for strat in strategies:
        if time.time() > deadline:
            break
        
        result, placed = _greedy_fill(boxes, container, strat, rng)
        vol = sum(p[4]*p[5]*p[6] for p in result if p[0] == 0)
        
        if vol > best_vol:
            best_vol = vol
            best_result = result[:]
    
    # Local search on best result
    if best_result is not None and time.time() < deadline:
        improved = _local_search(best_result, boxes, container, rng, max_iters=30)
        improved_vol = sum(p[4]*p[5]*p[6] for p in improved if p[0] == 0)
        if improved_vol > best_vol:
            best_result = improved
            best_vol = improved_vol
    
    # Additional random restarts if time permits
    if best_result is not None and time.time() < deadline:
        for _ in range(5):
            if time.time() > deadline:
                break
            result, placed = _greedy_fill(boxes, container, 'random', rng)
            vol = sum(p[4]*p[5]*p[6] for p in result if p[0] == 0)
            if vol > best_vol:
                best_vol = vol
                best_result = result[:]
    
    if best_result is None:
        # Fallback: simple shelf packing
        best_result = [(-1, 0, 0, 0, b[0], b[1], b[2]) for b in boxes]
    
    return best_result

# EVOLVE-BLOCK-END
