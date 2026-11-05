"""3D Container Loading with volume-aware distance field search."""
# EVOLVE-BLOCK-START
import time
import itertools
import sys
import math

def _orientations(w, h, d):
    """Generate all unique orientations of a box."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _fits(box, pos, placed, W, H, D):
    """Check if box at pos doesn't overlap with placed boxes and fits in container."""
    x, y, z = pos
    bw, bh, bd = box
    if x < 0 or y < 0 or z < 0:
        return False
    if x + bw > W or y + bh > H or z + bd > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        # Check overlap: boxes overlap if they intersect in all 3 axes
        if not (x + bw <= px or px + pw <= x or
                y + bh <= py or py + ph <= y or
                z + bd <= pz or pz + pd <= z):
            return False
    return True

def _build_distance_field(placed, W, H, D, grid_res=8):
    """
    Build a 3D distance field on a coarse grid.
    Returns a 3D array where each cell contains the distance to the nearest
    obstacle (placed box or container wall). Higher values = more free space.
    """
    nx = max(1, int(W / grid_res) + 1)
    ny = max(1, int(H / grid_res) + 1)
    nz = max(1, int(D / grid_res) + 1)
    
    # Initialize with large distance
    INF = 1e9
    dist = [[ [INF for _ in range(nx)] for _ in range(ny)] for _ in range(nz)]
    
    # Mark occupied cells
    for (px, py, pz, pw, ph, pd) in placed:
        x0 = max(0, int(px / grid_res))
        x1 = min(nx, int((px + pw) / grid_res) + 1)
        y0 = max(0, int(py / grid_res))
        y1 = min(ny, int((py + ph) / grid_res) + 1)
        z0 = max(0, int(pz / grid_res))
        z1 = min(nz, int((pz + pd) / grid_res) + 1)
        for z in range(z0, z1):
            for y in range(y0, y1):
                for x in range(x0, x1):
                    dist[z][y][x] = 0
    
    # Mark container walls as obstacles (distance 0 at boundaries)
    for z in range(nz):
        for y in range(ny):
            dist[z][y][0] = min(dist[z][y][0], 0)
            dist[z][y][nx-1] = min(dist[z][y][nx-1], 0)
    for z in range(nz):
        for x in range(nx):
            dist[z][0][x] = min(dist[z][0][x], 0)
            dist[z][ny-1][x] = min(dist[z][ny-1][x], 0)
    for y in range(ny):
        for x in range(nx):
            dist[0][y][x] = min(dist[0][y][x], 0)
            dist[nz-1][y][x] = min(dist[nz-1][y][x], 0)
    
    # Two-pass distance transform (approximate Euclidean)
    # Forward pass
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if dist[z][y][x] > 0:
                    candidates = [dist[z][y][x]]
                    if x > 0: candidates.append(dist[z][y][x-1] + 1)
                    if y > 0: candidates.append(dist[z][y-1][x] + 1)
                    if z > 0: candidates.append(dist[z-1][y][x] + 1)
                    if x > 0 and y > 0: candidates.append(dist[z][y-1][x-1] + 2)
                    if x > 0 and z > 0: candidates.append(dist[z-1][y][x-1] + 2)
                    if y > 0 and z > 0: candidates.append(dist[z-1][y-1][x] + 2)
                    if x > 0 and y > 0 and z > 0: candidates.append(dist[z-1][y-1][x-1] + 3)
                    dist[z][y][x] = min(candidates)
    
    # Backward pass
    for z in range(nz-1, -1, -1):
        for y in range(ny-1, -1, -1):
            for x in range(nx-1, -1, -1):
                if dist[z][y][x] > 0:
                    candidates = [dist[z][y][x]]
                    if x < nx-1: candidates.append(dist[z][y][x+1] + 1)
                    if y < ny-1: candidates.append(dist[z][y+1][x] + 1)
                    if z < nz-1: candidates.append(dist[z+1][y][x] + 1)
                    if x < nx-1 and y < ny-1: candidates.append(dist[z][y+1][x+1] + 2)
                    if x < nx-1 and z < nz-1: candidates.append(dist[z+1][y][x+1] + 2)
                    if y < ny-1 and z < nz-1: candidates.append(dist[z+1][y+1][x] + 2)
                    if x < nx-1 and y < ny-1 and z < nz-1: candidates.append(dist[z+1][y+1][x+1] + 3)
                    dist[z][y][x] = min(candidates)
    
    return dist, nx, ny, nz, grid_res

def _largest_empty_space_score(box, pos, dist_field, nx, ny, nz, grid_res, W, H, D):
    """
    Score a box placement by the minimum distance field value across its volume.
    Higher score = box is in a larger contiguous empty space.
    """
    x, y, z = pos
    bw, bh, bd = box
    if x < 0 or y < 0 or z < 0:
        return -1
    if x + bw > W or y + bh > H or z + bd > D:
        return -1
    
    # Sample the distance field across the box volume
    x0 = max(0, int(x / grid_res))
    x1 = min(nx, int((x + bw) / grid_res) + 1)
    y0 = max(0, int(y / grid_res))
    y1 = min(ny, int((y + bh) / grid_res) + 1)
    z0 = max(0, int(z / grid_res))
    z1 = min(nz, int((z + bd) / grid_res) + 1)
    
    min_dist = float('inf')
    for z in range(z0, z1):
        for y in range(y0, y1):
            for x in range(x0, x1):
                if dist_field[z][y][x] < min_dist:
                    min_dist = dist_field[z][y][x]
                    if min_dist == 0:
                        return 0
    
    if min_dist == float('inf'):
        return 0
    return min_dist

def _candidate_positions_v2(placed, W, H, D, max_candidates=500):
    """Generate candidate positions: corners of placed boxes + grid sampling."""
    candidates = set()
    candidates.add((0, 0, 0))
    
    for (px, py, pz, pw, ph, pd) in placed:
        candidates.add((px + pw, py, pz))
        candidates.add((px, py + ph, pz))
        candidates.add((px, py, pz + pd))
        # Also try positions offset by box dimensions
        candidates.add((px + pw, py + ph, pz))
        candidates.add((px + pw, py, pz + pd))
        candidates.add((px, py + ph, pz + pd))
    
    # Add some grid-based candidates for exploration
    step_x = max(1, W // 10)
    step_y = max(1, H // 10)
    step_z = max(1, D // 10)
    for x in range(0, W, step_x):
        for y in range(0, H, step_y):
            for z in range(0, D, step_z):
                candidates.add((x, y, z))
    
    # Filter to valid positions
    valid = []
    for (x, y, z) in candidates:
        if 0 <= x < W and 0 <= y < H and 0 <= z < D:
            valid.append((x, y, z))
    
    if len(valid) <= max_candidates:
        return valid
    
    # Sort by z (floor first), then x+y
    valid.sort(key=lambda p: (p[2], p[0] + p[1]))
    return valid[:max_candidates]

def _best_fit_volume_aware(boxes, container, deadline, max_candidates=500, grid_res=8):
    """Best-Fit with volume-aware distance field scoring."""
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Sort boxes by volume descending
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    placed = []
    placement_map = {}
    
    for i, w, h, d in indexed_boxes:
        if time.time() > deadline:
            break
        
        orients = _orientations(w, h, d)
        
        # Build distance field for current placement
        dist_field, nx, ny, nz, _ = _build_distance_field(placed, W, H, D, grid_res)
        
        candidates = _candidate_positions_v2(placed, W, H, D, max_candidates)
        
        best_pos = None
        best_score = -1
        best_orient = None
        
        for orient in orients:
            for pos in candidates:
                if _fits(orient, pos, placed, W, H, D):
                    # Score: prefer placements in large empty spaces
                    # Combine distance field score with "tightness" (lower coords)
                    df_score = _largest_empty_space_score(orient, pos, dist_field, nx, ny, nz, grid_res, W, H, D)
                    tightness = pos[0] + pos[1] + pos[2]
                    # We want high df_score (large empty space) and low tightness
                    # Use a weighted combination
                    combined = df_score * 10 - tightness
                    if combined > best_score:
                        best_score = combined
                        best_pos = pos
                        best_orient = orient
        
        if best_pos is not None:
            x, y, z = best_pos
            bw, bh, bd = best_orient
            placed.append((x, y, z, bw, bh, bd))
            placement_map[i] = (x, y, z, bw, bh, bd)
    
    result = []
    for i, (w, h, d) in enumerate(boxes):
        if i in placement_map:
            x, y, z, bw, bh, bd = placement_map[i]
            result.append((0, x, y, z, bw, bh, bd))
        else:
            result.append((-1, 0, 0, 0, w, h, d))
    return result

def _best_fit_decreasing(boxes, container, deadline, max_candidates=300, score_func=None):
    """Best-Fit-Decreasing with corner-based candidate generation."""
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Sort boxes by volume descending
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    placed = []
    placement_map = {}
    
    for i, w, h, d in indexed_boxes:
        if time.time() > deadline:
            break
        
        orients = _orientations(w, h, d)
        best_pos = None
        best_score = float('inf')
        best_orient = None
        
        # Generate candidate positions from corners
        candidates = set()
        candidates.add((0, 0, 0))
        for (px, py, pz, pw, ph, pd) in placed:
            candidates.add((px + pw, py, pz))
            candidates.add((px, py + ph, pz))
            candidates.add((px, py, pz + pd))
        
        valid = [(x, y, z) for (x, y, z) in candidates if 0 <= x < W and 0 <= y < H and 0 <= z < D]
        if len(valid) > max_candidates:
            valid.sort(key=lambda p: (p[2], p[0] + p[1]))
            valid = valid[:max_candidates]
        
        for orient in orients:
            for pos in valid:
                if _fits(orient, pos, placed, W, H, D):
                    score = pos[0] + pos[1] + pos[2]
                    if score < best_score:
                        best_score = score
                        best_pos = pos
                        best_orient = orient
        
        if best_pos is not None:
            x, y, z = best_pos
            bw, bh, bd = best_orient
            placed.append((x, y, z, bw, bh, bd))
            placement_map[i] = (x, y, z, bw, bh, bd)
    
    result = []
    for i, (w, h, d) in enumerate(boxes):
        if i in placement_map:
            x, y, z, bw, bh, bd = placement_map[i]
            result.append((0, x, y, z, bw, bh, bd))
        else:
            result.append((-1, 0, 0, 0, w, h, d))
    return result

def _volume_filling(boxes, container):
    """Calculate volume fraction filled."""
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return 0.0
    container_vol = W * H * D
    if container_vol == 0:
        return 0.0
    filled = 0
    for (bin_id, x, y, z, w, h, d) in boxes:
        if bin_id == 0:
            filled += w * h * d
    return filled / container_vol

def _improve_placement(result, boxes, container, deadline, max_iters=40):
    """Local search: try to reposition boxes to improve packing."""
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return result
    
    placed = []
    placement_map = {}
    for i, (bin_id, x, y, z, w, h, d) in enumerate(result):
        if bin_id == 0:
            placed.append((x, y, z, w, h, d))
            placement_map[i] = (x, y, z, w, h, d)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        best_gain = 0
        best_swap = None
        
        for i in list(placement_map.keys()):
            if time.time() > deadline:
                break
            x, y, z, w, h, d = placement_map[i]
            remaining = [p for p in placed if p != (x, y, z, w, h, d)]
            del placement_map[i]
            placed = remaining
            
            orients = _orientations(w, h, d)
            candidates = set()
            candidates.add((0, 0, 0))
            for (px, py, pz, pw, ph, pd) in placed:
                candidates.add((px + pw, py, pz))
                candidates.add((px, py + ph, pz))
                candidates.add((px, py, pz + pd))
            
            valid = [(cx, cy, cz) for (cx, cy, cz) in candidates if 0 <= cx < W and 0 <= cy < H and 0 <= cz < D]
            valid.sort(key=lambda p: (p[2], p[0] + p[1]))
            valid = valid[:200]
            
            best_pos = None
            best_score = float('inf')
            best_orient = None
            
            for orient in orients:
                for pos in valid:
                    if _fits(orient, pos, placed, W, H, D):
                        score = pos[0] + pos[1] + pos[2]
                        if score < best_score:
                            best_score = score
                            best_pos = pos
                            best_orient = orient
            
            if best_pos is not None:
                old_score = x + y + z
                if best_score < old_score - 1e-9:
                    gain = old_score - best_score
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = (i, best_pos, best_orient)
            
            placement_map[i] = (x, y, z, w, h, d)
            placed = list(remaining) + [(x, y, z, w, h, d)]
        
        if best_swap is not None:
            i, new_pos, new_orient = best_swap
            x, y, z, w, h, d = placement_map[i]
            placed = [p for p in placed if p != (x, y, z, w, h, d)]
            bx, by, bz = new_pos
            bw, bh, bd = new_orient
            placed.append((bx, by, bz, bw, bh, bd))
            placement_map[i] = (bx, by, bz, bw, bh, bd)
        else:
            break
    
    result = []
    for i, (w, h, d) in enumerate(boxes):
        if i in placement_map:
            x, y, z, bw, bh, bd = placement_map[i]
            result.append((0, x, y, z, bw, bh, bd))
        else:
            result.append((-1, 0, 0, 0, w, h, d))
    return result

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second margin before 150s limit
    
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Strategy 1: Best-Fit-Decreasing (sum-of-coords scoring)
    result1 = _best_fit_decreasing(boxes, container, deadline, max_candidates=300)
    score1 = _volume_filling(result1, container)
    
    # Strategy 2: Volume-aware with distance field
    result2 = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    score2 = 0.0
    if time.time() < deadline - 20.0:
        result2 = _best_fit_volume_aware(boxes, container, deadline, max_candidates=400, grid_res=10)
        score2 = _volume_filling(result2, container)
    
    # Pick the best initial result
    best = max([(score1, result1), (score2, result2)], key=lambda t: t[0])
    best_score = best[0]
    best_result = best[1]
    
    # Try to improve with local search if time allows
    if time.time() < deadline - 15.0:
        improved = _improve_placement(best_result, boxes, container, deadline, max_iters=50)
        improved_score = _volume_filling(improved, container)
        if improved_score > best_score:
            best_result = improved
            best_score = improved_score
    
    return best_result
# EVOLVE-BLOCK-END
