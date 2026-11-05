import time
import itertools

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 100.0  # 100s margin
    
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    if n == 0:
        return []
        
    # Generate unique rotations for a box
    def get_rotations(w, h, d):
        rots = set()
        for perm in itertools.permutations((w, h, d)):
            rots.add(perm)
        return list(rots)
    
    # Sorting strategies
    def sort_key_max_dim(b):
        return -max(b)
        
    def sort_key_volume(b):
        return -b[0]*b[1]*b[2]
        
    def sort_key_min_dim(b):
        return -min(b)
        
    def sort_key_aspect(b):
        return -(b[0]+b[1]+b[2])

    strategies = [
        ("max_dim", sort_key_max_dim),
        ("volume", sort_key_volume),
        ("min_dim", sort_key_min_dim),
        ("sum_dim", sort_key_aspect),
    ]
    
    best_placements = None
    best_volume = -1
    
    for name, key_func in strategies:
        if time.time() > deadline:
            break
            
        # Sort boxes
        indexed_boxes = list(enumerate(boxes))
        indexed_boxes.sort(key=lambda x: key_func(x[1]))
        
        placements = [None] * n
        placed_boxes = []  # List of (x, y, z, w, h, d) for placed boxes
        candidates = [(0, 0, 0)]
        
        for orig_idx, box in indexed_boxes:
            w, h, d = box
            placed = False
            best_pos = None
            best_score = None
            
            rots = get_rotations(w, h, d)
            
            for rw, rh, rd in rots:
                # Try all candidate positions
                for cx, cy, cz in candidates:
                    # Check bounds
                    if cx + rw > W or cy + rh > H or cz + rd > D:
                        continue
                    
                    # Check overlap with all placed boxes
                    overlap = False
                    for px, py, pz, pw, ph, pd in placed_boxes:
                        # Check if [cx, cx+rw] overlaps [px, px+pw] in X
                        if cx < px + pw and cx + rw > px:
                            # Check Y
                            if cy < py + ph and cy + rh > py:
                                # Check Z
                                if cz < pz + pd and cz + rd > pz:
                                    overlap = True
                                    break
                    if overlap:
                        continue
                    
                    # Score: prefer lower x, then lower y, then lower z
                    score = (cx, cy, cz)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pos = (cx, cy, cz, rw, rh, rd)
            
            if best_pos is not None:
                x, y, z, rw, rh, rd = best_pos
                placements[orig_idx] = (0, x, y, z, rw, rh, rd)
                placed_boxes.append((x, y, z, rw, rh, rd))
                
                # Generate new candidates
                # (x+rw, y, z), (x, y+rh, z), (x, y, z+rd)
                new_candidates = [
                    (x + rw, y, z),
                    (x, y + rh, z),
                    (x, y, z + rd)
                ]
                # Add to candidates if not already present (optional optimization, but O(N^2) is fine)
                candidates.extend(new_candidates)
            else:
                placements[orig_idx] = (-1, 0, 0, 0, w, h, d)
                
        # Calculate volume
        vol = 0
        for p in placements:
            if p[0] == 0:
                vol += p[4] * p[5] * p[6]
        
        if vol > best_volume:
            best_volume = vol
            best_placements = [tuple(p) for p in placements]
            
    if best_placements is None:
        # Fallback: simple greedy (should not happen if logic is correct, but just in case)
        placements = []
        for w, h, d in boxes:
            placements.append((-1, 0, 0, 0, w, h, d))
        return placements
        
    return best_placements
