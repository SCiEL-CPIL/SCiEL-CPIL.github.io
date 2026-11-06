"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    deadline = time.time() + 110.0  # safety margin before 120s limit
    
    # Precompute unique orientations for each box
    def get_orientations(w, h, d):
        opts = set()
        for perm in itertools.permutations((w, h, d)):
            opts.add(perm)
        return list(opts)
    
    # Check if a box of size (bw, bh, bd) at position (px, py, pz) fits without overlap
    def fits(bw, bh, bd, px, py, pz, placed):
        if px + bw > W or py + bh > H or pz + bd > D:
            return False
        for (qx, qy, qz, qw, qh, qd) in placed:
            # Check overlap in all 3 dimensions
            if (px < qx + qw and px + bw > qx and
                py < qy + qh and py + bh > qy and
                pz < qz + qd and pz + bd > qz):
                return False
        return True
    
    # Generate candidate anchor points for placement
    def get_anchors(placed, bw, bh, bd):
        candidates = set()
        # Origin
        candidates.add((0, 0, 0))
        
        # For each placed box, generate faces
        for (qx, qy, qz, qw, qh, qd) in placed:
            # Right face
            candidates.add((qx + qw, qy, qz))
            # Left face
            candidates.add((qx, qy, qz))
            # Back face
            candidates.add((qx, qy, qz))
            # Front face
            candidates.add((qx, qy + qh, qz))
            # Top face
            candidates.add((qx, qy, qz + qd))
            # Bottom face
            candidates.add((qx, qy, qz))
            
            # Also try corners adjacent to placed boxes
            candidates.add((qx + qw, qy, qz))
            candidates.add((qx, qy + qh, qz))
            candidates.add((qx, qy, qz + qd))
            candidates.add((qx + qw, qy + qh, qz))
            candidates.add((qx + qw, qy, qz + qd))
            candidates.add((qx, qy + qh, qz + qd))
            candidates.add((qx + qw, qy + qh, qz + qd))
        
        # Filter to valid positions within container
        valid = []
        for (px, py, pz) in candidates:
            if 0 <= px <= W - bw and 0 <= py <= H - bh and 0 <= pz <= D - bd:
                valid.append((px, py, pz))
        
        # Sort by priority: lower z, then lower y, then lower x (back-bottom-left preference)
        valid.sort(key=lambda p: (p[2], p[1], p[0]))
        return valid
    
    # Try to pack boxes using greedy best-fit decreasing with multiple strategies
    def try_pack(box_list, order_key, max_anchors=50):
        placed = []  # list of (x, y, z, w, h, d)
        result = [None] * len(box_list)  # result[i] = (bin, x, y, z, w, h, d) or None
        
        # Sort boxes by the given key
        indexed_boxes = [(i, box_list[i]) for i in range(len(box_list))]
        indexed_boxes.sort(key=lambda x: order_key(x[1]), reverse=True)
        
        for idx, (orig_idx, (bw_orig, bh_orig, bd_orig)) in indexed_boxes:
            if time.time() > deadline:
                break
            
            orientations = get_orientations(bw_orig, bh_orig, bd_orig)
            
            best_pos = None
            best_orient = None
            best_score = None
            
            for (bw, bh, bd) in orientations:
                anchors = get_anchors(placed, bw, bh, bd)
                # Limit anchors for speed
                if len(anchors) > max_anchors:
                    anchors = anchors[:max_anchors]
                
                for (px, py, pz) in anchors:
                    if fits(bw, bh, bd, px, py, pz, placed):
                        # Score: prefer lower positions (back, bottom, left)
                        score = (px, py, pz)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_pos = (px, py, pz)
                            best_orient = (bw, bh, bd)
            
            if best_pos is not None:
                px, py, pz = best_pos
                bw, bh, bd = best_orient
                placed.append((px, py, pz, bw, bh, bd))
                result[orig_idx] = (0, px, py, pz, bw, bh, bd)
            else:
                result[orig_idx] = (-1, 0, 0, 0, bw_orig, bh_orig, bd_orig)
        
        return result
    
    # Define different ordering strategies
    def volume_key(box):
        return box[0] * box[1] * box[2]
    
    def max_dim_key(box):
        return max(box)
    
    def min_dim_key(box):
        return min(box)
    
    def surface_key(box):
        w, h, d = box
        return 2 * (w*h + w*d + h*d)
    
    strategies = [
        ("volume_desc", volume_key),
        ("max_dim_desc", max_dim_key),
        ("min_dim_asc", lambda b: -min(b)),
        ("surface_desc", surface_key),
    ]
    
    best_result = None
    best_fill = 0
    
    for name, key_func in strategies:
        if time.time() > deadline:
            break
        
        result = try_pack(boxes, key_func)
        
        # Calculate fill ratio
        filled_volume = 0
        for r in result:
            if r is not None and r[0] == 0:
                _, _, _, _, w, h, d = r
                filled_volume += w * h * d
        
        total_volume = W * H * D
        fill = filled_volume / total_volume if total_volume > 0 else 0
        
        if fill > best_fill:
            best_fill = fill
            best_result = result
        
        if time.time() > deadline:
            break
    
    # If we haven't found a result yet, fall back to simple shelf method
    if best_result is None:
        placements = []
        x, y, z, row_h, layer_d = 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d, x, row_h = 0, z + layer_d, 0, 0, 0
            if z + d > D:
                placements.append((-1, 0, 0, 0, w, h, d))
                continue
            placements.append((0, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        best_result = placements
    
    return best_result
# EVOLVE-BLOCK-END
