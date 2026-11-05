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
    if not placed:
        return (cz, cx, cy)
    
    min_dist = float('inf')
    for px, py, pz, pw, ph, pd in placed:
        dist = abs(cx - px) + abs(cy - py) + abs(cz - pz)
        if dist < min_dist:
            min_dist = dist
    
    return (cz, min_dist, cx, cy)

def _pack(boxes, container, sort_key, deadline):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return [], 0
    
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=sort_key, reverse=True)
    
    placed = []
    placements = [None] * n
    candidate_set = {(0, 0, 0)}
    total_volume = 0
    
    for i, (orig_idx, (w, h, d)) in enumerate(indexed_boxes):
        if time.time() > deadline:
            break
            
        box_rotations = list(set(itertools.permutations([w, h, d])))
        
        best_pos = None
        best_rot = None
        best_score = None
        
        for rw, rh, rd in box_rotations:
            if rw > W or rh > H or rd > D:
                continue
                
            for cx, cy, cz in list(candidate_set):
                if cx + rw > W or cy + rh > H or cz + rd > D:
                    continue
                    
                if _check_overlap(cx, cy, cz, rw, rh, rd, placed):
                    continue
                
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
            total_volume += rw * rh * rd
            
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
            
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
            
    return placements, total_volume

def load(boxes, container):
    start_time = time.time()
    # Reserve 5 seconds for safety
    deadline = start_time + 115.0
    
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    if not boxes:
        return []
        
    # Define multiple sorting strategies
    strategies = [
        lambda x: x[1][0]*x[1][1]*x[1][2], # Volume
        lambda x: max(x[1]),                # Max dimension
        lambda x: x[1][2],                  # Height (Z)
        lambda x: x[1][0],                  # Width (X)
        lambda x: x[1][1],                  # Depth (Y)
    ]
    
    best_placements = None
    best_volume = -1
    
    for strategy in strategies:
        if time.time() > deadline:
            break
        placements, vol = _pack(boxes, container, strategy, deadline)
        if vol > best_volume:
            best_volume = vol
            best_placements = placements
            
    if best_placements is None:
        # Fallback if no strategy ran or all failed
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    return best_placements
