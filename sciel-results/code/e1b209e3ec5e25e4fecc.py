"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def _all_rotations(w, h, d):
    """Return unique rotations of a box (w,h,d)."""
    dims = (w, h, d)
    seen = set()
    rots = []
    for perm in itertools.permutations(dims):
        if perm not in seen:
            seen.add(perm)
            rots.append(perm)
    return rots

def _fits(pos, dims, W, H, D):
    x, y, z = pos
    w, h, d = dims
    return (x + w <= W and y + h <= H and z + d <= D)

def _overlaps(pos, dims, placed):
    """Check if box at pos with dims overlaps any placed box."""
    x, y, z = pos
    w, h, d = dims
    for (px, py, pz, pw, ph, pd) in placed:
        if not (x + w <= px or px + pw <= x):
            if not (y + h <= py or py + ph <= y):
                if not (z + d <= pz or pz + pd <= z):
                    return True
    return False

def _generate_positions(placed, W, H, D):
    """Generate candidate positions based on placed boxes and origin."""
    xs = {0}
    ys = {0}
    zs = {0}
    for (px, py, pz, pw, ph, pd) in placed:
        xs.add(px + pw)
        ys.add(py + ph)
        zs.add(pz + pd)
    # Also add positions based on the "floor" of each placed box
    for (px, py, pz, pw, ph, pd) in placed:
        # Positions adjacent to each face
        xs.add(px)
        ys.add(py)
        zs.add(pz)
    positions = []
    for x in xs:
        for y in ys:
            for z in zs:
                if x < W and y < H and z < D:
                    positions.append((x, y, z))
    return positions

def _place_box_ffd(boxes, container, deadline):
    """First-fit decreasing with rotation, using position generation."""
    W, H, D = container
    placed = []  # list of (x, y, z, w, h, d, box_index)
    remaining = list(enumerate(boxes))
    
    # Sort by volume descending
    remaining.sort(key=lambda idx_box: boxes[idx_box[0]][0]*boxes[idx_box[0]][1]*boxes[idx_box[0]][2], reverse=True)
    
    placements = [None] * len(boxes)
    
    for idx, (w, h, d) in remaining:
        if time.time() > deadline:
            break
        rots = _all_rotations(w, h, d)
        positions = _generate_positions(placed, W, H, D)
        
        found = False
        # Try positions in a heuristic order: prefer lower z, then y, then x
        positions.sort(key=lambda p: (p[2], p[1], p[0]))
        
        for pos in positions:
            if time.time() > deadline:
                break
            for rot in rots:
                if _fits(pos, rot, W, H, D) and not _overlaps(pos, rot, placed):
                    placed.append((pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], idx))
                    placements[idx] = (0, pos[0], pos[1], pos[2], rot[0], rot[1], rot[2])
                    found = True
                    break
            if found:
                break
        
        if not found:
            placements[idx] = (-1, 0, 0, 0, w, h, d)
    
    # Fill in any remaining None entries
    for i in range(len(boxes)):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    return placements

def _local_search(placements, boxes, container, deadline):
    """Try to improve by re-inserting left-out boxes and moving loaded ones."""
    W, H, D = container
    n = len(boxes)
    
    # Build list of loaded and unloaded
    loaded = [i for i in range(n) if placements[i][0] == 0]
    unloaded = [i for i in range(n) if placements[i][0] == -1]
    
    if not unloaded:
        return placements
    
    best_placements = placements[:]
    best_volume = sum(boxes[i][0]*boxes[i][1]*boxes[i][2] for i in loaded)
    
    # Try to insert unloaded boxes by removing some loaded boxes
    # Simple approach: try each unloaded box, see if we can place it by removing one loaded box
    for ui in unloaded:
        if time.time() > deadline:
            break
        uw, uh, ud = boxes[ui]
        urots = _all_rotations(uw, uh, ud)
        
        # Generate positions
        placed_list = []
        for li in loaded:
            p = placements[li]
            placed_list.append((p[1], p[2], p[3], p[4], p[5], p[6], li))
        
        positions = _generate_positions(placed_list, W, H, D)
        positions.sort(key=lambda p: (p[2], p[1], p[0]))
        
        # Try to place ui directly
        for pos in positions:
            if time.time() > deadline:
                break
            for rot in urots:
                if _fits(pos, rot, W, H, D) and not _overlaps(pos, rot, placed_list):
                    # Can place ui without removing anything
                    new_placements = best_placements[:]
                    new_placements[ui] = (0, pos[0], pos[1], pos[2], rot[0], rot[1], rot[2])
                    new_loaded = [i for i in range(n) if new_placements[i][0] == 0]
                    new_volume = sum(boxes[i][0]*boxes[i][1]*boxes[i][2] for i in new_loaded)
                    if new_volume > best_volume:
                        best_volume = new_volume
                        best_placements = new_placements
                        loaded = new_loaded
                    break
            else:
                continue
            break
    
    return best_placements

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    W, H, D = container
    n = len(boxes)
    
    # Initial FFD placement
    placements = _place_box_ffd(boxes, container, deadline)
    
    # Local search if time permits
    if time.time() < deadline - 10:
        placements = _local_search(placements, boxes, container, deadline)
    
    # Ensure all placements are valid
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    return placements
# EVOLVE-BLOCK-END
