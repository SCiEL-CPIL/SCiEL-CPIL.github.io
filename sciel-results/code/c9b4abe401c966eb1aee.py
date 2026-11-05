"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 100.0  # 20s safety margin before 120s limit
    
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Preprocess: compute volume and sort by volume descending
    indexed_boxes = [(w, h, d, i) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[0]*x[1]*x[2], reverse=True)
    
    # State: list of placed boxes as (x, y, z, w, h, d)
    placed = []
    # For each original index, store (bin, x, y, z, w, h, d)
    result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    def orientations(w, h, d):
        """Generate all 6 unique orientations."""
        orients = set()
        for perm in itertools.permutations([w, h, d]):
            if perm not in orients:
                orients.add(perm)
        return list(orientations) if False else list(orients)
    
    def get_orientations(w, h, d):
        """Generate all 6 unique orientations as tuples."""
        orients = set()
        for perm in itertools.permutations([w, h, d]):
            orients.add(perm)
        return list(orients)
    
    def overlaps(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        """Check if two boxes overlap."""
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                    y1 + h1 <= y2 or y2 + h2 <= y1 or
                    z1 + d1 <= z2 or z2 + d2 <= z1)
    
    def fits(x, y, z, w, h, d):
        """Check if a box at (x,y,z) with dims (w,h,d) fits in container without overlap."""
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in placed:
            if overlaps(x, y, z, w, h, d, px, py, pz, pw, ph, pd):
                return False
        return True
    
    def candidate_positions():
        """Generate candidate positions for a new box."""
        # Start with origin and all faces of existing boxes
        positions = set()
        positions.add((0, 0, 0))
        
        for (px, py, pz, pw, ph, pd) in placed:
            # Positions on each face
            positions.add((px + pw, py, pz))
            positions.add((px, py + ph, pz))
            positions.add((px, py, pz + pd))
            # Also try combinations with other boxes' extents for better packing
            # For efficiency, limit to direct face contacts
        
        # Filter to valid positions within container
        valid_positions = []
        for (x, y, z) in positions:
            if 0 <= x < W and 0 <= y < H and 0 <= z < D:
                valid_positions.append((x, y, z))
        
        # Sort by a heuristic: prefer positions closer to origin (bottom-left-front)
        valid_positions.sort(key=lambda p: (p[0], p[1], p[2]))
        
        return valid_positions
    
    def best_fit_position(w, h, d):
        """Find the best position for a box with given dimensions using best-fit decreasing."""
        best_pos = None
        best_score = float('inf')
        
        for (x, y, z) in candidate_positions():
            if not fits(x, y, z, w, h, d):
                continue
            # Score: prefer positions that minimize the max extent (bounding box)
            # or prefer positions closer to origin
            # Use a combination: minimize (x+y+z) as a proxy for compactness
            score = x + y + z
            if score < best_score:
                best_score = score
                best_pos = (x, y, z)
        
        return best_pos
    
    # Main placement loop
    for (w, h, d, orig_idx) in indexed_boxes:
        if time.time() > deadline:
            break
        
        orients = get_orientations(w, h, d)
        
        best_placement = None
        best_score = float('inf')
        
        for (ow, oh, od) in orients:
            pos = best_fit_position(ow, oh, od)
            if pos is not None:
                x, y, z = pos
                # Score: prefer placements that minimize the resulting bounding box
                # or minimize the sum of coordinates
                score = x + y + z
                if score < best_score:
                    best_score = score
                    best_placement = (x, y, z, ow, oh, od)
        
        if best_placement is not None:
            x, y, z, ow, oh, od = best_placement
            placed.append((x, y, z, ow, oh, od))
            result[orig_idx] = (0, x, y, z, ow, oh, od)
    
    return result
# EVOLVE-BLOCK-END
