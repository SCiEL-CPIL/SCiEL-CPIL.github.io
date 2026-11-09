"""3D Container Loading: maximize mean fraction of container volume filled."""
import time
import itertools
import random
import heapq

def _rotations(w, h, d):
    """Return all unique orientations (permutations) of a box."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _overlap(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                y1 + h1 <= y2 or y2 + h2 <= y1 or
                z1 + d1 <= z2 or z2 + d2 <= z1)

def _fits_in_container(x, y, z, w, h, d, W, H, D):
    return x >= 0 and y >= 0 and z >= 0 and x + w <= W and y + h <= H and z + d <= D

def _candidate_positions(placed, W, H, D):
    """Generate candidate positions from extreme points of placed boxes."""
    # Start with origin
    candidates = set([(0, 0, 0)])
    # For each placed box, add positions adjacent to its faces
    for (px, py, pz, pw, ph, pd) in placed:
        # Right face
        candidates.add((px + pw, py, pz))
        candidates.add((px + pw, py + ph, pz))
        candidates.add((px + pw, py, pz + pd))
        candidates.add((px + pw, py + ph, pz + pd))
        # Top face
        candidates.add((px, py + ph, pz))
        candidates.add((px + pw, py + ph, pz))
        candidates.add((px, py + ph, pz + pd))
        candidates.add((px + pw, py + ph, pz + pd))
        # Front face
        candidates.add((px, py, pz + pd))
        candidates.add((px + pw, py, pz + pd))
        candidates.add((px, py + ph, pz + pd))
        candidates.add((px + pw, py + ph, pz + pd))
    
    # Filter to valid positions within container
    valid = []
    for (cx, cy, cz) in candidates:
        if 0 <= cx < W and 0 <= cy < H and 0 <= cz < D:
            valid.append((cx, cy, cz))
    return valid

def _greedy_place(boxes, container, order, deadline=None):
    """
    Place boxes in given order using greedy algorithm.
    Returns list of (bin, x, y, z, w, h, d) tuples.
    """
    W, H, D = container
    placed = []  # (x, y, z, w, h, d)
    result = [(-1, 0, 0, 0, 0, 0, 0)] * len(boxes)
    
    for idx in order:
        if deadline and time.time() > deadline:
            break
        w, h, d = boxes[idx]
        rots = _rotations(w, h, d)
        
        best_pos = None
        best_vol = -1
        # Try all orientations and candidate positions
        for rw, rh, rd in rots:
            positions = _candidate_positions(placed, W, H, D)
            for (cx, cy, cz) in positions:
                if not _fits_in_container(cx, cy, cz, rw, rh, rd, W, H, D):
                    continue
                # Check overlap
                overlap = False
                for (px, py, pz, pw, ph, pd) in placed:
                    if _overlap(cx, cy, cz, rw, rh, rd, px, py, pz, pw, ph, pd):
                        overlap = True
                        break
                if not overlap:
                    # Score: prefer positions that use space well (lower z, then y, then x)
                    score = -(cz * 1000 + cy * 100 + cx)
                    if score > best_vol:
                        best_vol = score
                        best_pos = (cx, cy, cz, rw, rh, rd)
        
        if best_pos:
            cx, cy, cz, rw, rh, rd = best_pos
            placed.append((cx, cy, cz, rw, rh, rd))
            result[idx] = (0, cx, cy, cz, rw, rh, rd)
    
    return result

def _score(placement, boxes, container):
    """Calculate fraction of container volume filled."""
    W, H, D = container
    total_vol = W * H * D
    if total_vol == 0:
        return 0.0
    filled = 0
    for (b, x, y, z, w, h, d) in placement:
        if b == 0:
            filled += w * h * d
    return filled / total_vol

def _try_order(boxes, container, order, deadline):
    """Try a specific order of boxes and return placement."""
    return _greedy_place(boxes, container, order, deadline)

def load(boxes, container):
    """
    Place a subset of boxes inside one container without overlap.
    Returns list of (bin, x, y, z, w, h, d) per box.
    bin=0 for loaded, -1 for left out.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    W, H, D = container
    n = len(boxes)
    
    # Precompute volumes
    volumes = [w * h * d for w, h, d in boxes]
    
    # Try multiple ordering strategies
    best_placement = None
    best_score = -1
    
    # Strategy 1: Sort by volume descending
    order1 = sorted(range(n), key=lambda i: volumes[i], reverse=True)
    placement = _try_order(boxes, container, order1, deadline)
    score = _score(placement, boxes, container)
    if score > best_score:
        best_score = score
        best_placement = placement
    
    # Strategy 2: Sort by max dimension descending
    def max_dim(i):
        w, h, d = boxes[i]
        return max(w, h, d)
    order2 = sorted(range(n), key=max_dim, reverse=True)
    if time.time() < deadline:
        placement = _try_order(boxes, container, order2, deadline)
        score = _score(placement, boxes, container)
        if score > best_score:
            best_score = score
            best_placement = placement
    
    # Strategy 3: Sort by volume/max_dim ratio (prefer compact boxes)
    def vol_max_ratio(i):
        w, h, d = boxes[i]
        md = max(w, h, d)
        if md == 0:
            return 0
        return volumes[i] / md
    order3 = sorted(range(n), key=vol_max_ratio, reverse=True)
    if time.time() < deadline:
        placement = _try_order(boxes, container, order3, deadline)
        score = _score(placement, boxes, container)
        if score > best_score:
            best_score = score
            best_placement = placement
    
    # Strategy 4: Random shuffles
    random.seed(42)
    for attempt in range(50):
        if time.time() > deadline:
            break
        order = list(range(n))
        random.shuffle(order)
        placement = _try_order(boxes, container, order, deadline)
        score = _score(placement, boxes, container)
        if score > best_score:
            best_score = score
            best_placement = placement
    
    # Strategy 5: Sort by sum of dimensions
    def sum_dim(i):
        w, h, d = boxes[i]
        return w + h + d
    order5 = sorted(range(n), key=sum_dim, reverse=True)
    if time.time() < deadline:
        placement = _try_order(boxes, container, order5, deadline)
        score = _score(placement, boxes, container)
        if score > best_score:
            best_score = score
            best_placement = placement
    
    if best_placement is None:
        # Fallback: no boxes placed
        best_placement = [(-1, 0, 0, 0, 0, 0, 0)] * n
    
    return best_placement
