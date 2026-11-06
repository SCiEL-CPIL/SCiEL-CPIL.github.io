"""Strong 3D Container Loading: skyline-based placement with deadline guard."""
import time

# EVOLVE-BLOCK-START
def load(boxes, container):
    t_start = time.time()
    deadline = t_start + 110.0  # safety margin vs 120s limit

    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    # Precompute orientations for each box, sorted by priority
    box_data = []
    for idx, (w, h, d) in enumerate(boxes):
        # Generate all unique orientations
        seen = set()
        orients = []
        for perm in [(w, h, d), (w, d, h), (h, w, d), (h, d, w), (d, w, h), (d, h, w)]:
            if perm not in seen:
                seen.add(perm)
                orients.append(perm)
        
        # Filter to only orientations that fit in container
        valid_orients = [(bw, bh, bd) for bw, bh, bd in orients if bw <= W and bh <= H and bd <= D]
        if not valid_orients:
            valid_orients = [(w, h, d)]  # will be marked as not placed
        
        # Sort orientations: prefer those with smaller max dimension (more "flat")
        valid_orients.sort(key=lambda o: (max(o), o[0], o[1], o[2]))
        
        volume = w * h * d
        box_data.append((idx, valid_orients, volume, w, h, d))
    
    # Sort boxes by volume descending (largest first)
    box_data.sort(key=lambda x: x[2], reverse=True)
    
    placed = []  # list of (x, y, z, bw, bh, bd)
    results = [None] * len(boxes)
    
    def get_candidates():
        """Generate candidate positions from placed boxes and origin."""
        cx_set = {0}
        cy_set = {0}
        cz_set = {0}
        for (px, py, pz, pw, ph, pd) in placed:
            cx_set.add(px)
            cx_set.add(px + pw)
            cy_set.add(py)
            cy_set.add(py + ph)
            cz_set.add(pz)
            cz_set.add(pz + pd)
        cx_list = sorted(cx_set)
        cy_list = sorted(cy_set)
        cz_list = sorted(cz_set)
        return cx_list, cy_list, cz_list
    
    def check_no_overlap(cx, cy, cz, bw, bh, bd):
        """Check if a box at (cx,cy,cz) with dims (bw,bh,bd) doesn't overlap any placed box."""
        for (px, py, pz, pw, ph, pd) in placed:
            # No overlap if separated in any dimension
            if not ((cx + bw <= px) or (px + pw <= cx) or
                    (cy + bh <= py) or (py + ph <= cy) or
                    (cz + bd <= pz) or (pz + pd <= cz)):
                return False
        return True
    
    def try_place_box(idx, orients, volume, ow, oh, od):
        """Try to place a box using best-fit decreasing with all candidate corners."""
        cx_list, cy_list, cz_list = get_candidates()
        
        # Generate all candidate positions sorted by (z, y, x) to prefer lowest positions
        candidates = []
        for cz in cz_list:
            for cy in cy_list:
                for cx in cx_list:
                    candidates.append((cx, cy, cz))
        candidates.sort(key=lambda c: (c[2], c[1], c[0]))
        
        best_pos = None
        best_orient = None
        best_score = None
        
        for (cx, cy, cz) in candidates:
            for (bw, bh, bd) in orients:
                # Check bounds
                if cx + bw > W or cy + bh > H or cz + bd > D:
                    continue
                
                # Check overlap
                if not check_no_overlap(cx, cy, cz, bw, bh, bd):
                    continue
                
                # Score: prefer positions that are lower, more to the left, more to the front
                # Also prefer orientations that leave more room (smaller max dimension)
                score = (cz, cy, cx, max(bw, bh, bd))
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
                    best_orient = (bw, bh, bd)
        
        if best_pos is not None:
            cx, cy, cz = best_pos
            bw, bh, bd = best_orient
            placed.append((cx, cy, cz, bw, bh, bd))
            results[idx] = (0, cx, cy, cz, bw, bh, bd)
            return True
        else:
            results[idx] = (-1, 0, 0, 0, ow, oh, od)
            return False
    
    # Phase 1: Place all boxes using best-fit decreasing
    for idx, orients, volume, ow, oh, od in box_data:
        if time.time() > deadline:
            # Time out - mark remaining as not placed
            for j in range(len(boxes)):
                if results[j] is None:
                    w_i, h_i, d_i = boxes[j]
                    results[j] = (-1, 0, 0, 0, w_i, h_i, d_i)
            break
        
        try_place_box(idx, orients, volume, ow, oh, od)
    
    # Phase 2: Try to fill gaps by re-examining unplaced boxes with more aggressive search
    # Collect unplaced boxes
    unplaced = []
    for idx, orients, volume, ow, oh, od in box_data:
        if results[idx][0] == -1:
            unplaced.append((idx, orients, volume, ow, oh, od))
    
    # Sort unplaced by volume ascending (try small boxes first to fill gaps)
    unplaced.sort(key=lambda x: x[2])
    
    for idx, orients, volume, ow, oh, od in unplaced:
        if time.time() > deadline:
            break
        
        # Try to place this unplaced box
        try_place_box(idx, orients, volume, ow, oh, od)
    
    # Fill any remaining None entries
    for i in range(len(boxes)):
        if results[i] is None:
            w_i, h_i, d_i = boxes[i]
            results[i] = (-1, 0, 0, 0, w_i, h_i, d_i)
    
    return results
# EVOLVE-BLOCK-END
