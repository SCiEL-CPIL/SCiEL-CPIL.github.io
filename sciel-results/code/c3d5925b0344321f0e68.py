import time
import itertools
import numpy as np

def _orientations(w, h, d):
    """Yield all 6 distinct orientations of a box with dims (w,h,d)."""
    seen = set()
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            yield perm

def _overlaps(x1, y1, z1, x2, y2, z2, w1, h1, d1, w2, h2, d2):
    """Check if two boxes overlap (strictly)."""
    if x1 + w1 <= x2 or x2 + w2 <= x1:
        return False
    if y1 + h1 <= y2 or y2 + h2 <= y1:
        return False
    if z1 + d1 <= z2 or z2 + d2 <= z1:
        return False
    return True

def _fits(placed, x, y, z, w, h, d, W, H, D):
    """Check if box at (x,y,z) with dims (w,h,d) fits in container and doesn't overlap."""
    if x < 0 or y < 0 or z < 0:
        return False
    if x + w > W or y + h > H or z + d > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        if _overlaps(x, y, z, px, py, pz, w, h, d, pw, ph, pd):
            return False
    return True

def _generate_candidates(placed, W, H, D):
    """
    Generate candidate (x,y,z) positions.
    We use the Extreme Point Heuristic: candidates are the corners of the bounding boxes
    formed by the intersection of planes extending from the faces of placed boxes.
    To keep it simple and fast, we use the faces of placed boxes and container boundaries.
    """
    xs = {0, W}
    ys = {0, H}
    zs = {0, D}
    
    for (px, py, pz, pw, ph, pd) in placed:
        xs.update([px, px + pw])
        ys.update([py, py + ph])
        zs.update([pz, pz + pd])
        
    # Filter to valid range
    xs = sorted([x for x in xs if 0 <= x <= W])
    ys = sorted([y for y in ys if 0 <= y <= H])
    zs = sorted([z for z in zs if 0 <= z <= D])
    
    cands = []
    for z in zs:
        for y in ys:
            for x in xs:
                cands.append((x, y, z))
                
    # Sort candidates to prefer bottom-front-left
    # This order helps in finding a "stable" base
    cands.sort(key=lambda pos: (pos[2], pos[1], pos[0]))
    return cands

def _try_pack(boxes, container, order, deadline):
    """Try to pack boxes in given order; return placements list and filled volume."""
    W, H, D = container
    placed = []  # list of (x, y, z, w, h, d)
    result = [None] * len(boxes)  # will store (bin, x, y, z, w, h, d)
    filled_vol = 0.0
    
    # Pre-compute orientations for each box index in the order to avoid repeated calculation
    # Actually, we can just compute them on the fly, it's fast for 60 boxes.
    
    for idx in order:
        if time.time() > deadline:
            break
        
        w, h, d = boxes[idx]
        best_pos = None
        best_orient = None
        best_score = float('inf')
        
        # Try each orientation
        for ow, oh, od in _orientations(w, h, d):
            # Generate candidate positions
            cands = _generate_candidates(placed, W, H, D)
            
            # We want to find the "best" position.
            # Heuristic: Prefer positions that are lower (z), then forward (y), then left (x).
            # The cands are already sorted this way.
            # However, we can also try to maximize the "support" or minimize overhang.
            # For now, let's stick to the sorted order but maybe check a few more if the first few fail?
            # No, the first valid one in the sorted order is usually a good "Best Fit" approximation 
            # for filling from the bottom up.
            
            for (cx, cy, cz) in cands:
                if _fits(placed, cx, cy, cz, ow, oh, od, W, H, D):
                    # Score the position. Lower is better.
                    # We can refine this score. 
                    # Current sort is (z, y, x).
                    # Let's just take the first one found in the sorted list.
                    best_pos = (cx, cy, cz)
                    best_orient = (ow, oh, od)
                    break
            
            if best_pos is not None:
                break
        
        if best_pos is not None:
            cx, cy, cz = best_pos
            ow, oh, od = best_orient
            placed.append((cx, cy, cz, ow, oh, od))
            result[idx] = (0, cx, cy, cz, ow, oh, od)
            filled_vol += w * h * d
        else:
            result[idx] = (-1, 0, 0, 0, w, h, d)
            
    # Fill in any None (in case of timeout)
    for j in range(len(boxes)):
        if result[j] is None:
            w, h, d = boxes[j]
            result[j] = (-1, 0, 0, 0, w, h, d)
    
    return result, filled_vol

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # Deadline guard
    start_time = time.time()
    # Leave 5s buffer for safety
    deadline = start_time + 115.0
    
    # Generate different sort orders
    orders = []
    
    # 1. Volume descending
    orders.append(sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True))
    
    # 2. Max dimension descending
    orders.append(sorted(range(n), key=lambda i: max(boxes[i]), reverse=True))
    
    # 3. Sum of dimensions descending
    orders.append(sorted(range(n), key=lambda i: sum(boxes[i]), reverse=True))
    
    # 4. "Flatness" (min dimension) descending - place flat boxes first?
    orders.append(sorted(range(n), key=lambda i: min(boxes[i]), reverse=True))
    
    # 5. Aspect ratio (most cube-like first)
    orders.append(sorted(range(n), key=lambda i: (max(boxes[i]) / min(boxes[i])), reverse=True))
    
    # 6. Randomized tie-breakers for volume
    # We can add a few random shuffles of the volume-sorted list to explore different local optima
    import random
    random.seed(42) # Deterministic for reproducibility in testing, but maybe not needed
    base_order = orders[0]
    for _ in range(3):
        # Shuffle boxes with similar volumes? Or just a full shuffle?
        # A full shuffle might be too chaotic.
        # Let's try shuffling the order slightly.
        shuffled = base_order.copy()
        random.shuffle(shuffled)
        orders.append(shuffled)
        
    best_result = None
    best_filled = 0.0
    
    for order in orders:
        if time.time() > deadline:
            break
        result, filled_vol = _try_pack(boxes, container, order, deadline)
        if filled_vol > best_filled:
            best_filled = filled_vol
            best_result = result
        # Early stop if we've filled the container completely (theoretical max)
        if best_filled >= W * H * D * 0.99:
            break

    if best_result is None:
        # Fallback: return all as left out
        best_result = []
        for w, h, d in boxes:
            best_result.append((-1, 0, 0, 0, w, h, d))
    
    return best_result
