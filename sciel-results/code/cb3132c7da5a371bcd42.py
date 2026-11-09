"""3D Container Loading - optimized packer with spatial grid and random restarts."""
import time
import random
import itertools
import numpy as np

# EVOLVE-BLOCK-START

def _all_rotations(w, h, d):
    """Return all unique axis-aligned rotations of a box."""
    rots = set()
    for perm in itertools.permutations([w, h, d]):
        rots.add(perm)
    return list(rots)

def _build_grid(W, H, D, cell_size):
    """Build a 3D occupancy grid. cell_size is the resolution."""
    nx = max(1, int(np.ceil(W / cell_size)))
    ny = max(1, int(np.ceil(H / cell_size)))
    nz = max(1, int(np.ceil(D / cell_size)))
    grid = np.zeros((nx, ny, nz), dtype=np.uint8)
    return grid, nx, ny, nz, cell_size

def _mark_grid(grid, nx, ny, nz, cell_size, x, y, z, w, h, d):
    """Mark cells occupied by a box at (x,y,z) with dims (w,h,d)."""
    x0 = max(0, int(x // cell_size))
    x1 = min(nx, int(np.ceil((x + w) / cell_size)))
    y0 = max(0, int(y // cell_size))
    y1 = min(ny, int(np.ceil((y + h) / cell_size)))
    z0 = max(0, int(z // cell_size))
    z1 = min(nz, int(np.ceil((z + d) / cell_size)))
    if x1 > x0 and y1 > y0 and z1 > z0:
        grid[x0:x1, y0:y1, z0:z1] = 1

def _unmark_grid(grid, nx, ny, nz, cell_size, x, y, z, w, h, d):
    """Unmark cells for a box."""
    x0 = max(0, int(x // cell_size))
    x1 = min(nx, int(np.ceil((x + w) / cell_size)))
    y0 = max(0, int(y // cell_size))
    y1 = min(ny, int(np.ceil((y + h) / cell_size)))
    z0 = max(0, int(z // cell_size))
    z1 = min(nz, int(np.ceil((z + d) / cell_size)))
    if x1 > x0 and y1 > y0 and z1 > z0:
        grid[x0:x1, y0:y1, z0:z1] = 0

def _grid_check(grid, nx, ny, nz, cell_size, x, y, z, w, h, d):
    """Quick check if box fits in grid (no overlap with marked cells)."""
    x0 = max(0, int(x // cell_size))
    x1 = min(nx, int(np.ceil((x + w) / cell_size)))
    y0 = max(0, int(y // cell_size))
    y1 = min(ny, int(np.ceil((y + h) / cell_size)))
    z0 = max(0, int(z // cell_size))
    z1 = min(nz, int(np.ceil((z + d) / cell_size)))
    if x1 <= x0 or y1 <= y0 or z1 <= z0:
        return True
    if np.any(grid[x0:x1, y0:y1, z0:z1]):
        return False
    return True

def _exact_overlap_check(placed, x, y, z, w, h, d):
    """Check exact overlap with all placed boxes. placed is list of (x,y,z,w,h,d)."""
    for px, py, pz, pw, ph, pd in placed:
        if x < px + pw and x + w > px and y < py + ph and y + h > py and z < pz + pd and z + d > pz:
            return False
    return True

def _score_placement(placed, W, H, D):
    """Compute fraction of container volume filled."""
    if not placed:
        return 0.0
    total_vol = W * H * D
    if total_vol == 0:
        return 0.0
    filled = sum(w * h * d for _, _, _, w, h, d in placed)
    return filled / total_vol

def _greedy_pack(boxes, container, order, deadline):
    """
    Greedy pack boxes in given order using corner-based candidate generation.
    Returns list of (bin, x, y, z, w, h, d) for each box.
    """
    W, H, D = container
    start_time = time.time()
    
    # Sort boxes by volume descending for better packing (but respect 'order' for randomization)
    # Actually, 'order' is the permutation of box indices. We'll pack in that order.
    # Precompute rotations for each box
    rotations = [_all_rotations(w, h, d) for w, h, d in boxes]
    
    placed = []  # list of (x, y, z, w, h, d)
    result = [None] * len(boxes)
    
    # Candidate positions: start with origin
    candidates = [(0, 0, 0)]
    
    for idx in order:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        best_pos = None
        best_score = -1
        best_rot = None
        
        # Try each rotation
        for rw, rh, rd in rotations[idx]:
            # Try each candidate position
            for cx, cy, cz in candidates:
                if cx + rw > W or cy + rh > H or cz + rd > D:
                    continue
                # Quick check: does it fit without overlapping placed boxes?
                if not _exact_overlap_check(placed, cx, cy, cz, rw, rh, rd):
                    continue
                # Score: prefer lower z, then lower y, then lower x (compact)
                # Also prefer positions that create fewer new candidates
                score = -(cz * 10000 + cy * 100 + cx)
                if score > best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
                    best_rot = (rw, rh, rd)
        
        if best_pos is not None:
            bx, by, bz = best_pos
            bw, bh, bd = best_rot
            placed.append((bx, by, bz, bw, bh, bd))
            result[idx] = (0, bx, by, bz, bw, bh, bd)
            
            # Generate new candidate positions from this placement
            # Right face, front face, top face
            new_cands = []
            # Right: (bx + bw, by, bz)
            if bx + bw <= W:
                new_cands.append((bx + bw, by, bz))
            # Front: (bx, by + bh, bz)
            if by + bh <= H:
                new_cands.append((bx, by + bh, bz))
            # Top: (bx, by, bz + bd)
            if bz + bd <= D:
                new_cands.append((bx, by, bz + bd))
            
            # Deduplicate and filter
            for nc in new_cands:
                if nc not in candidates:
                    candidates.append(nc)
            
            # Limit candidates to avoid explosion
            if len(candidates) > 200:
                # Keep only the most promising (lowest z, y, x)
                candidates.sort(key=lambda c: (c[2], c[1], c[0]))
                candidates = candidates[:200]
        else:
            result[idx] = (-1, 0, 0, 0, w, h, d)
    
    # Fill in any unplaced boxes
    for i in range(len(boxes)):
        if result[i] is None:
            w, h, d = boxes[i]
            result[i] = (-1, 0, 0, 0, w, h, d)
    
    return result

def _improve_with_reinsertion(placement, boxes, container, deadline):
    """
    Try to reinsert left-out boxes by removing some placed boxes and repacking.
    """
    W, H, D = container
    if time.time() > deadline:
        return placement
    
    # Identify left-out boxes
    left_out = [i for i, p in enumerate(placement) if p[0] == -1]
    if not left_out:
        return placement
    
    # Try to place each left-out box by removing one placed box
    placed_indices = [i for i, p in enumerate(placement) if p[0] == 0]
    
    for lo_idx in left_out:
        if time.time() > deadline:
            break
        w, h, d = boxes[lo_idx]
        rotations = _all_rotations(w, h, d)
        
        # Try placing at all candidate positions
        best = None
        best_score = -1
        
        # Build list of placed boxes
        placed_boxes = [(placement[i][1], placement[i][2], placement[i][3], 
                        placement[i][4], placement[i][5], placement[i][6]) 
                       for i in placed_indices]
        
        for rw, rh, rd in rotations:
            # Try grid of positions
            step = max(1, min(W, H, D) // 50)
            for x in range(0, W - rw + 1, step):
                for y in range(0, H - rh + 1, step):
                    for z in range(0, D - rd + 1, step):
                        if _exact_overlap_check(placed_boxes, x, y, z, rw, rh, rd):
                            score = -(z * 10000 + y * 100 + x)
                            if score > best_score:
                                best_score = score
                                best = (x, y, z, rw, rh, rd)
        
        if best is not None:
            x, y, z, rw, rh, rd = best
            placement[lo_idx] = (0, x, y, z, rw, rh, rd)
            # Update placed_boxes
            placed_boxes.append((x, y, z, rw, rh, rd))
    
    return placement

def load(boxes, container):
    """
    Main entry point. Tries multiple random orderings and returns the best packing.
    """
    W, H, D = container
    n = len(boxes)
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin: 110s out of 120s
    
    # Precompute box volumes
    volumes = [w * h * d for w, h, d in boxes]
    total_vol = W * H * D
    
    # Sort indices by volume descending (base order)
    base_order = sorted(range(n), key=lambda i: -volumes[i])
    
    best_placement = None
    best_score = -1.0
    
    # Deterministic first pass
    placement = _greedy_pack(boxes, container, base_order, deadline)
    score = _score_placement(
        [(p[1], p[2], p[3], p[4], p[5], p[6]) for p in placement if p[0] == 0],
        W, H, D
    )
    if score > best_score:
        best_score = score
        best_placement = placement
    
    if time.time() > deadline:
        return best_placement
    
    # Random restarts with perturbed orders
    rng = random.Random(42)
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:  # Reserve 5s for final improvement
            break
        
        # Create perturbed order: mostly sorted by volume, with some swaps
        order = list(base_order)
        # Randomly swap a few pairs
        n_swaps = rng.randint(1, max(1, n // 10))
        for _ in range(n_swaps):
            i, j = rng.sample(range(n), 2)
            order[i], order[j] = order[j], order[i]
        
        placement = _greedy_pack(boxes, container, order, deadline)
        score = _score_placement(
            [(p[1], p[2], p[3], p[4], p[5], p[6]) for p in placement if p[0] == 0],
            W, H, D
        )
        if score > best_score:
            best_score = score
            best_placement = placement
    
    # Final improvement pass on the best placement
    if best_placement is not None and time.time() < deadline - 1.0:
        best_placement = _improve_with_reinsertion(best_placement, boxes, container, deadline)
    
    # Ensure all boxes have a result
    if best_placement is None:
        best_placement = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Verify no box is missing
    while len(best_placement) < n:
        best_placement.append((-1, 0, 0, 0, 0, 0, 0))
    
    return best_placement

# EVOLVE-BLOCK-END
