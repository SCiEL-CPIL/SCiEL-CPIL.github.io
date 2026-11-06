"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    deadline = time.time() + 130  # safety margin before 150s kill
    
    # Try multiple sorting strategies, keep best result
    best_placements = None
    best_score = float('inf')
    
    # Define sorting keys
    sort_keys = [
        lambda b: -(b[0]*b[1]*b[2]),  # volume desc
        lambda b: -(b[0]+b[1]+b[2]),  # sum of dims desc
        lambda b: -(max(b[0],b[1],b[2])),  # max dim desc
        lambda b: -(b[0]*b[1]*b[2]),  # volume desc (duplicate for stability)
    ]
    
    def compute_score(placements):
        """Compute B + 0.1*(1 - last_bin_occupied_vol / container_vol)"""
        if not placements:
            return 0.0
        bins = {}
        for (b, x, y, z, w, h, d) in placements:
            if b not in bins:
                bins[b] = 0
            bins[b] += w*h*d
        B = len(bins)
        last_bin = max(bins.keys())
        last_vol = bins[last_bin]
        container_vol = W*H*D
        frac = last_vol / container_vol if container_vol > 0 else 0
        return B + 0.1*(1 - frac)
    
    def try_bfd(sorted_boxes):
        """Run BFD on sorted boxes, return placements or None if timeout."""
        bins = []  # list of lists of (x,y,z,w,h,d)
        
        def can_place(bin_boxes, x, y, z, w, h, d):
            if x + w > W or y + h > H or z + d > D:
                return False
            for (px, py, pz, pw, ph, pd) in bin_boxes:
                # Check overlap
                if (x < px + pw and x + w > px and
                    y < py + ph and y + h > py and
                    z < pz + pd and z + d > pz):
                    return False
            return True
        
        def find_best_position(bin_boxes, w, h, d):
            """Find position minimizing x+y+z (lower is better for compaction)."""
            # Candidate positions: try placing at various (x,y,z)
            # Use a grid of candidate x, y, z values
            best_pos = None
            best_key = float('inf')
            
            # Candidate x values: 0, and right edge of each placed box
            xs = [0]
            ys = [0]
            zs = [0]
            for (px, py, pz, pw, ph, pd) in bin_boxes:
                if px + pw <= W:
                    xs.append(px + pw)
                if py + ph <= H:
                    ys.append(py + ph)
                if pz + pd <= D:
                    zs.append(pz + pd)
            
            xs = sorted(set(x for x in xs if x + w <= W))
            ys = sorted(set(y for y in ys if y + h <= H))
            zs = sorted(set(z for z in zs if z + d <= D))
            
            if not xs or not ys or not zs:
                return None
            
            for x in xs:
                for y in ys:
                    for z in zs:
                        if can_place(bin_boxes, x, y, z, w, h, d):
                            key = x + y + z  # minimize sum of coordinates
                            if key < best_key:
                                best_key = key
                                best_pos = (x, y, z)
            return best_pos
        
        placements = []
        box_indices = list(range(len(sorted_boxes)))
        
        for idx in box_indices:
            if time.time() > deadline:
                return None  # timeout, discard this attempt
            w, h, d = sorted_boxes[idx]
            
            placed = False
            # Try existing bins first (best-fit: try all, pick best)
            best_bin = -1
            best_pos = None
            best_key = float('inf')
            
            for bi in range(len(bins)):
                pos = find_best_position(bins[bi], w, h, d)
                if pos is not None:
                    key = pos[0] + pos[1] + pos[2]
                    if key < best_key:
                        best_key = key
                        best_bin = bi
                        best_pos = pos
            
            if best_bin >= 0:
                x, y, z = best_pos
                bins[best_bin].append((x, y, z, w, h, d))
                placements.append((best_bin, x, y, z, w, h, d))
            else:
                # New bin
                new_bin_id = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                placements.append((new_bin_id, 0, 0, 0, w, h, d))
        
        return placements
    
    # Try different sorting orders
    for key_func in sort_keys:
        if time.time() > deadline:
            break
        sorted_boxes = sorted(boxes, key=key_func)
        placements = try_bfd(sorted_boxes)
        if placements is None:
            continue  # timeout
        score = compute_score(placements)
        if score < best_score:
            best_score = score
            best_placements = placements
    
    # If all timed out, fall back to simple shelf
    if best_placements is None:
        placements, b, x, y, z, row_h, layer_d = [], 0, 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            placements.append((b, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        best_placements = placements
    
    return best_placements
# EVOLVE-BLOCK-END
