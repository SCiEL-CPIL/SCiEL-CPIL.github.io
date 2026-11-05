"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second margin before 120s kill
    
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    container_vol = W * H * D
    
    # Precompute all 6 orientations for each box
    # orientation: (w, h, d)
    def get_orientations(w, h, d):
        dims = sorted([w, h, d], reverse=True)
        # Generate all permutations of dims
        orients = set()
        for perm in itertools.permutations(dims):
            orients.add(perm)
        return list(orients)
    
    # For each box, store its dimensions and all valid orientations
    box_orients = []
    for w, h, d in boxes:
        box_orients.append(get_orientations(w, h, d))
    
    # Sort boxes by volume descending (largest first) - FFD approach
    box_indices = list(range(n))
    box_indices.sort(key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)
    
    def check_overlaps(placement, placed_list):
        """Check if a new placement overlaps with any existing placement."""
        _, x, y, z, w, h, d = placement
        for _, px, py, pz, pw, ph, pd in placed_list:
            # Check overlap in all 3 dimensions
            if not (x + w <= px or px + pw <= x):
                if not (y + h <= py or py + ph <= y):
                    if not (z + d <= pz or pz + pd <= z):
                        return True
        return False
    
    def try_place_box(box_idx, placed_list, W, H, D):
        """Try to place a box in the best position. Returns placement or None."""
        w, h, d = boxes[box_idx]
        best_pos = None
        best_score = None
        
        # Generate candidate positions
        # Strategy: Try corners and edges of existing boxes
        # Also try the origin
        
        candidates = set()
        
        # Always try origin
        candidates.add((0, 0, 0))
        
        # Try positions based on existing placed boxes
        for _, px, py, pz, pw, ph, pd in placed_list:
            # Try to place adjacent to existing boxes
            # Right of box
            if px + pw + w <= W:
                candidates.add((px + pw, py, pz))
            # Left of box (if possible, but usually we pack from origin)
            if px - w >= 0:
                candidates.add((px - w, py, pz))
            # Above box
            if py + ph + h <= H:
                candidates.add((px, py + ph, pz))
            if py - h >= 0:
                candidates.add((px, py - h, pz))
            # In front of box
            if pz + pd + d <= D:
                candidates.add((px, py, pz + pd))
            if pz - d >= 0:
                candidates.add((px, py, pz - d))
        
        # Also try some grid-based candidates for better packing
        # But keep it sparse to avoid O(n^3)
        # Let's add a few more strategic points:
        # Max x, y, z extents
        max_x = 0
        max_y = 0
        max_z = 0
        for _, px, py, pz, pw, ph, pd in placed_list:
            max_x = max(max_x, px + pw)
            max_y = max(max_y, py + ph)
            max_z = max(max_z, pz + pd)
        
        candidates.add((max_x, 0, 0))
        candidates.add((0, max_y, 0))
        candidates.add((0, 0, max_z))
        
        # Try all orientations for each candidate position
        for orient in box_orients[box_idx]:
            ow, oh, od = orient
            for cx, cy, cz in candidates:
                if cx + ow > W or cy + oh > H or cz + od > D:
                    continue
                
                # Check overlaps
                placement = (0, cx, cy, cz, ow, oh, od)
                if not check_overlaps(placement, placed_list):
                    # Score: prefer positions closer to origin (bottom-left-front)
                    # to fill corners
                    score = cx + cy + cz
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pos = placement
        
        return best_pos
    
    # Run multiple strategies and pick the best
    strategies = []
    
    # Strategy 1: FFD with candidate positions
    def strategy_ffd():
        placed = []
        result = [-1] * n
        for i in box_indices:
            placement = try_place_box(i, placed, W, H, D)
            if placement is not None:
                placed.append(placement)
                result[i] = placement
            else:
                result[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
        
        # Calculate volume filled
        vol = sum(w * h * d for _, _, _, _, w, h, d in result if _ == 0)
        return result, vol
    
    # Strategy 2: Shelf packing (simpler but might be faster and sometimes better)
    def strategy_shelf():
        placed = []
        result = [-1] * n
        x, y, z = 0, 0, 0
        shelf_h = 0
        layer_d = 0
        
        for i in box_indices:
            w, h, d = boxes[i]
            # Try orientations to minimize height increase
            best_orient = None
            best_h = float('inf')
            for ow, oh, od in box_orients[i]:
                if x + ow <= W and z + od <= D:
                    if oh < best_h:
                        best_h = oh
                        best_orient = (ow, oh, od)
            
            if best_orient is None:
                # Try next shelf
                x = 0
                shelf_h = 0
                for ow, oh, od in box_orients[i]:
                    if x + ow <= W and z + od <= D:
                        if oh < best_h:
                            best_h = oh
                            best_orient = (ow, oh, od)
            
            if best_orient is None:
                # Try next layer
                x, y = 0, y + shelf_h
                shelf_h = 0
                layer_d = 0
                for ow, oh, od in box_orients[i]:
                    if x + ow <= W and z + od <= D:
                        if oh < best_h:
                            best_h = oh
                            best_orient = (ow, oh, od)
            
            if best_orient is None:
                # Try next layer in z
                x, y, z = 0, 0, z + layer_d
                shelf_h = 0
                layer_d = 0
                for ow, oh, od in box_orients[i]:
                    if x + ow <= W and z + od <= D:
                        if oh < best_h:
                            best_h = oh
                            best_orient = (ow, oh, od)
            
            if best_orient is None:
                result[i] = (-1, 0, 0, 0, w, h, d)
                continue
            
            ow, oh, od = best_orient
            placement = (0, x, y, z, ow, oh, od)
            placed.append(placement)
            result[i] = placement
            x += ow
            shelf_h = max(shelf_h, oh)
            layer_d = max(layer_d, od)
        
        vol = sum(w * h * d for _, _, _, _, w, h, d in result if _ == 0)
        return result, vol
    
    # Run strategies
    best_result = None
    best_vol = -1
    
    # Try FFD
    if time.time() < deadline:
        res, vol = strategy_ffd()
        if vol > best_vol:
            best_vol = vol
            best_result = res
    
    # Try Shelf
    if time.time() < deadline:
        res, vol = strategy_shelf()
        if vol > best_vol:
            best_vol = vol
            best_result = res
    
    if best_result is None:
        # Fallback: all left out
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_result

# EVOLVE-BLOCK-END
