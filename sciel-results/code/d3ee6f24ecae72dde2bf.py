"""3D Bin Packing: BFD with 2D Skyline (height-map) representation for efficient placement."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin for 120s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume descending (largest first) for better packing
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: -x[4])
    
    # Each bin maintains:
    #   height_map: 2D numpy array of shape (W, H) storing max Z-height at each (x,y) cell
    #   placed_boxes: list of (x, y, z, w, h, d) for reference
    bins = []  # list of dicts: {'height_map': 2D array, 'placed': [(x,y,z,w,h,d), ...]}
    
    def create_bin():
        return {
            'height_map': np.zeros((W, H), dtype=np.int32),
            'placed': []
        }
    
    def can_place_in_bin(bin_data, x, y, z, w, h, d):
        """Check if box fits at (x,y,z) using height map."""
        if x + w > W or y + h > H or z + d > D:
            return False
        # Check that the height map in the rectangle [x, x+w) x [y, y+h) 
        # has max height <= z
        region = bin_data['height_map'][x:x+w, y:y+h]
        return np.max(region) <= z
    
    def update_height_map(bin_data, x, y, z, w, h, d):
        """Update height map after placing a box."""
        bin_data['height_map'][x:x+w, y:y+h] = np.maximum(
            bin_data['height_map'][x:x+w, y:y+h], z + d
        )
    
    def find_best_position(bin_data, w, h, d):
        """
        Find the best (x, y, z) position for a box in this bin using skyline/height-map.
        For each candidate (x, y), the lowest valid z is max height in the footprint.
        We score positions to prefer low z, low x, low y (corner packing).
        """
        height_map = bin_data['height_map']
        
        # Candidate x positions: 0 and right edges of existing boxes
        xs = [0]
        ys = [0]
        for (px, py, pz, pw, ph, pd) in bin_data['placed']:
            if px + pw <= W:
                xs.append(px + pw)
            if py + ph <= H:
                ys.append(py + ph)
        xs = sorted(set(xs))
        ys = sorted(set(ys))
        
        best_score = float('inf')
        best_pos = None
        
        for x in xs:
            if x + w > W:
                continue
            for y in ys:
                if y + h > H:
                    continue
                # The lowest z where this box can be placed is the max height in its footprint
                region = height_map[x:x+w, y:y+h]
                z = int(np.max(region)) if region.size > 0 else 0
                
                if z + d > D:
                    continue
                
                # Verify no overlap (height map check is sufficient for axis-aligned boxes
                # placed at their lowest valid z, but let's be safe)
                if not can_place_in_bin(bin_data, x, y, z, w, h, d):
                    continue
                
                # Score: prefer low z (stacking), low x, low y (corner)
                # Use weighted sum; z is most important for fragmentation
                score = 1.0 * z + 0.5 * x + 0.3 * y
                if score < best_score:
                    best_score = score
                    best_pos = (x, y, z)
        
        return best_pos
    
    def bin_fill_fraction(bin_data):
        """Compute volume fraction of bin occupied."""
        if not bin_data['placed']:
            return 0.0
        total_vol = sum(pw * ph * pd for (_, _, _, pw, ph, pd) in bin_data['placed'])
        return total_vol / (W * H * D)
    
    def place_box(w, h, d):
        """Place a box in the best existing bin or create a new one."""
        best_bin = -1
        best_score = float('inf')
        best_pos = None
        
        for b_idx, bin_data in enumerate(bins):
            pos = find_best_position(bin_data, w, h, d)
            if pos is not None:
                x, y, z = pos
                fill = bin_fill_fraction(bin_data)
                # Prefer bins with higher fill (consolidation) and lower placement height
                score = z - 3.0 * fill * D
                if score < best_score:
                    best_score = score
                    best_bin = b_idx
                    best_pos = pos
        
        if best_bin == -1:
            bins.append(create_bin())
            best_bin = len(bins) - 1
            best_pos = (0, 0, 0)
        
        x, y, z = best_pos
        bins[best_bin]['placed'].append((x, y, z, w, h, d))
        update_height_map(bins[best_bin], x, y, z, w, h, d)
        return best_bin, x, y, z
    
    # Place boxes in sorted order
    placements = [None] * n
    for idx, w, h, d, vol in indexed_boxes:
        if time.time() > deadline:
            # Emergency: place remaining boxes in new bins at origin
            for i in range(idx, n):
                _, w2, h2, d2, _ = indexed_boxes[i]
                bin_idx = len(bins)
                bins.append(create_bin())
                bins[bin_idx]['placed'].append((0, 0, 0, w2, h2, d2))
                update_height_map(bins[bin_idx], 0, 0, 0, w2, h2, d2)
                placements[i] = (bin_idx, 0, 0, 0, w2, h2, d2)
            break
        
        bin_idx, x, y, z = place_box(w, h, d)
        placements[idx] = (bin_idx, x, y, z, w, h, d)
    
    # Consolidation polish: try to move boxes from later bins into earlier bins
    if time.time() < deadline - 10.0 and len(bins) > 1:
        for b_idx in range(len(bins) - 1, 0, -1):
            if time.time() > deadline - 10.0:
                break
            if not bins[b_idx]['placed']:
                continue
            # Try to move each box in this bin to an earlier bin
            moved_boxes = []
            for box_idx in range(len(bins[b_idx]['placed']) - 1, -1, -1):
                if time.time() > deadline - 10.0:
                    break
                px, py, pz, pw, ph, pd = bins[b_idx]['placed'][box_idx]
                # Remove from current bin
                bins[b_idx]['placed'].pop(box_idx)
                # Rebuild height map for this bin (simpler than decrementing)
                bins[b_idx]['height_map'] = np.zeros((W, H), dtype=np.int32)
                for (qx, qy, qz, qw, qh, qd) in bins[b_idx]['placed']:
                    bins[b_idx]['height_map'][qx:qx+qw, qy:qy+qh] = np.maximum(
                        bins[b_idx]['height_map'][qx:qx+qw, qy:qy+qh], qz + qd
                    )
                
                # Try to place in earlier bins
                moved = False
                for target_bin in range(b_idx):
                    if not bins[target_bin]['placed']:
                        continue
                    pos = find_best_position(bins[target_bin], pw, ph, pd)
                    if pos is not None:
                        tx, ty, tz = pos
                        bins[target_bin]['placed'].append((tx, ty, tz, pw, ph, pd))
                        update_height_map(bins[target_bin], tx, ty, tz, pw, ph, pd)
                        # Update placement
                        for orig_idx in range(n):
                            p = placements[orig_idx]
                            if p is not None and p[0] == b_idx and p[1] == px and p[2] == py and p[3] == pz and p[4] == pw and p[5] == ph and p[6] == pd:
                                placements[orig_idx] = (target_bin, tx, ty, tz, pw, ph, pd)
                                break
                        moved = True
                        break
                if not moved:
                    # Put back
                    bins[b_idx]['placed'].append((px, py, pz, pw, ph, pd))
                    update_height_map(bins[b_idx], px, py, pz, pw, ph, pd)
    
    return placements
# EVOLVE-BLOCK-END
