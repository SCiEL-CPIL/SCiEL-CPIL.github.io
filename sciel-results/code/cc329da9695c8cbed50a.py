import time
import itertools

def _check_overlap(cx, cy, cz, rw, rh, rd, placed):
    """Check if box at (cx,cy,cz) with dims (rw,rh,rd) overlaps any placed box."""
    for px, py, pz, pw, ph, pd in placed:
        if not (cx + rw <= px or px + pw <= cx or
                cy + rh <= py or py + ph <= cy or
                cz + rd <= pz or pz + pd <= cz):
            return True
    return False

def _score_position(cx, cy, cz, placed, W, H, D):
    """Score a position: prefer low z, then compact (close to existing boxes)."""
    # Primary: minimize z (fill from bottom)
    # Secondary: minimize distance to nearest placed box (compactness)
    # Tertiary: minimize x, then y
    if not placed:
        return (cz, cx, cy)
    
    # Find minimum distance to any placed box (Manhattan-like)
    min_dist = float('inf')
    for px, py, pz, pw, ph, pd in placed:
        # Distance between box centers or edges
        dx = max(0, max(cx - (px + pw), px - (cx + rw))) if False else 0  # simplified
        # Use simple metric: sum of absolute differences in starting coords
        dist = abs(cx - px) + abs(cy - py) + abs(cz - pz)
        if dist < min_dist:
            min_dist = dist
    
    return (cz, min_dist, cx, cy)

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for safety
    
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    if n == 0:
        return []
        
    # Sort boxes by volume descending (largest first)
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: x[1][0]*x[1][1]*x[1][2], reverse=True)
    
    placed = [] # list of (x, y, z, w, h, d)
    placements = [None] * n # (bin, x, y, z, w, h, d)
    
    # Candidate points: use a set to avoid duplicates
    candidate_set = {(0, 0, 0)}
    
    for i, (orig_idx, (w, h, d)) in enumerate(indexed_boxes):
        if time.time() > deadline:
            break
            
        box_rotations = list(set(itertools.permutations([w, h, d])))
        
        best_pos = None
        best_rot = None
        best_score = None
        
        # Try all rotations
        for rw, rh, rd in box_rotations:
            # Check if rotation fits in container
            if rw > W or rh > H or rd > D:
                continue
                
            # Try all candidate points
            for cx, cy, cz in list(candidate_set):
                # Check bounds
                if cx + rw > W or cy + rh > H or cz + rd > D:
                    continue
                    
                # Check overlap with placed boxes
                if _check_overlap(cx, cy, cz, rw, rh, rd, placed):
                    continue
                
                # Score the position
                score = _score_position(cx, cy, cz, placed, W, H, D)
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
                    best_rot = (rw, rh, rd)
                        
        if best_pos is not None:
            x, y, z = best_pos
            rw, rh, rd = best_rot
            placed.append((x, y, z, rw, rh, rd))
            placements[orig_idx] = (0, x, y, z, rw, rh, rd)
            
            # Add new candidate points (corners of the placed box)
            new_points = [
                (x, y, z + rd),
                (x + rw, y, z + rd),
                (x, y + rh, z + rd),
                (x + rw, y + rh, z + rd),
                (x + rw, y, z),
                (x, y + rh, z),
                (x + rw, y + rh, z),
            ]
            for pt in new_points:
                if pt not in candidate_set:
                    candidate_set.add(pt)
        else:
            placements[orig_idx] = (-1, 0, 0, 0, w, h, d)
            
    # Fill any remaining None (in case of timeout)
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
            
    return placements
