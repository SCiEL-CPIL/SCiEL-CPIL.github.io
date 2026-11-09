import time
import random
import math

def pack(boxes, container):
    W, H, D = container
    if not boxes:
        return []
    
    n = len(boxes)
    container_vol = W * H * D
    if container_vol == 0:
        return [(0, 0, 0, w, h, d) for (w, h, d) in boxes]

    start_time = time.time()
    DEADLINE = start_time + 145.0

    # BSP Node representation
    # A node is either:
    # 1. An empty box: (x, y, z, w, h, d)
    # 2. A split node: (node1, node2)
    # We will represent the state of a bin as a list of "empty boxes" initially, 
    # but to be a true BSP, we should maintain the tree structure.
    # However, for 3D packing with "no rotation" and "axis aligned", 
    # maintaining a list of maximal empty boxes is equivalent to the leaves of the BSP.
    # Let's implement a function that takes a list of empty boxes and a new box to place,
    # and returns the new list of empty boxes.
    
    def split_boxes(empty_boxes, box):
        """
        Takes a list of empty boxes and a box to place (x, y, z, w, h, d).
        Returns the new list of empty boxes after placing the box.
        Assumes the box fits exactly into one of the empty boxes (or union of them, 
        but our candidate generation ensures it fits into a single empty box).
        """
        bx, by, bz, bw, bh, bd = box
        new_empty = []
        
        for ex, ey, ez, ew, eh, ed in empty_boxes:
            # Check if the new box is fully contained in this empty box
            if bx >= ex and by >= ey and bz >= ez and \
               bx + bw <= ex + ew and by + bh <= ey + eh and bz + bd <= ez + ed:
                
                # Split this empty box
                # We split along the axes to create up to 6 new empty boxes (actually 3 pairs, but some might be zero volume)
                
                # 1. Left/Right (X-axis)
                # Left part: x from ex to bx
                if bx > ex:
                    new_empty.append((ex, ey, ez, bx - ex, eh, ed))
                # Right part: x from bx+bw to ex+ew
                if bx + bw < ex + ew:
                    new_empty.append((bx + bw, ey, ez, ex + ew - (bx + bw), eh, ed))
                
                # 2. Down/Up (Y-axis) - Only for the part of the X-span occupied by the new box? 
                # No, standard BSP splitting:
                # If we split X, we have two regions: Left and Right.
                # The "Middle" (where the box is) is removed.
                # But the box might not span the full Y or Z of the empty box.
                # So we need to split the "Middle" region further?
                # Actually, the standard "subtractive" BSP works like this:
                # The empty box is divided into up to 6 non-overlapping boxes that do not intersect the new box.
                
                # Let's do it properly:
                # Region 1: x in [ex, bx), y in [ey, ey+eh), z in [ez, ez+ed)
                if bx > ex:
                    new_empty.append((ex, ey, ez, bx - ex, eh, ed))
                
                # Region 2: x in [bx, bx+bw), y in [ey, by), z in [ez, ez+ed)
                if by > ey:
                    new_empty.append((bx, ey, ez, bw, by - ey, ed))
                
                # Region 3: x in [bx, bx+bw), y in [by, by+bh), z in [ez, bz)
                if bz > ez:
                    new_empty.append((bx, by, ez, bw, bh, bz - ez))
                
                # Region 4: x in [bx, bx+bw), y in [by, by+bh), z in [bz+bd, ez+ed)
                if bz + bd < ez + ed:
                    new_empty.append((bx, by, bz + bd, bw, bh, ez + ed - (bz + bd)))
                
                # Region 5: x in [bx, bx+bw), y in [by+bh, ey+eh), z in [ez, ez+ed)
                if by + bh < ey + eh:
                    new_empty.append((bx, by + bh, ez, bw, ey + eh - (by + bh), ed))
                
                # Region 6: x in [bx+bw, ex+ew), y in [ey, ey+eh), z in [ez, ez+ed)
                if bx + bw < ex + ew:
                    new_empty.append((bx + bw, ey, ez, ex + ew - (bx + bw), eh, ed))
                
            else:
                # The new box is not in this empty box, so keep it
                new_empty.append((ex, ey, ez, ew, eh, ed))
                
        return new_empty

    def find_best_position(empty_boxes, w, h, d, W, H, D):
        """
        Finds the best position (x, y, z) for a box of size (w, h, d) 
        within the given list of empty boxes.
        Returns (x, y, z, index_of_empty_box) or None.
        Heuristic: Minimize (x, y, z) lexicographically to keep packing compact.
        """
        best_pos = None
        best_empty_idx = -1
        best_key = None
        
        for i, (ex, ey, ez, ew, eh, ed) in enumerate(empty_boxes):
            if ew >= w and eh >= h and ed >= d:
                # The box fits in this empty space.
                # We place it at the "corner" (ex, ey, ez) to maximize compactness.
                # Is this always optimal? No, but it's a good heuristic.
                # Alternative: Check if we can place it at (ex, ey, ez)
                # Since we are splitting, placing at the corner is the standard BSP insertion.
                
                # Heuristic: Prefer lower Z, then Y, then X
                key = (ez, ey, ex)
                
                if best_key is None or key < best_key:
                    best_key = key
                    best_pos = (ex, ey, ez)
                    best_empty_idx = i
                    
        if best_pos is not None:
            return best_pos, best_empty_idx
        return None, -1

    def pack_instance(sort_key, candidate_key):
        indexed_boxes = list(enumerate(boxes))
        indexed_boxes.sort(key=lambda t: sort_key(t[1]))
        
        bins = [] # List of dicts: {'empty_boxes': [...], 'items': [...]}
        placement_map = {}
        
        for idx, (w, h, d) in indexed_boxes:
            if w > W or h > H or d > D:
                continue # Can't fit, skip (should not happen per problem statement usually, but safe)
                
            placed = False
            
            # Try existing bins
            # Heuristic: Try bins that are "fuller" first to minimize B? 
            # Or try bins where the box fits "tightest"?
            # Let's sort bins by occupied volume descending (fill full bins first)
            bin_indices = list(range(len(bins)))
            if bin_indices:
                # Calculate volume for sorting
                def bin_vol(bi):
                    # Sum of volumes of items in bin
                    return sum(item[3]*item[4]*item[5] for item in bins[bi]['items'])
                bin_indices.sort(key=bin_vol, reverse=True)
            
            for bi in bin_indices:
                bin_data = bins[bi]
                empty_boxes = bin_data['empty_boxes']
                
                pos, empty_idx = find_best_position(empty_boxes, w, h, d, W, H, D)
                
                if pos is not None:
                    x, y, z = pos
                    # Update empty boxes
                    new_empty = split_boxes(empty_boxes, (x, y, z, w, h, d))
                    bin_data['empty_boxes'] = new_empty
                    bin_data['items'].append((x, y, z, w, h, d))
                    placement_map[idx] = (bi, x, y, z)
                    placed = True
                    break
            
            if not placed:
                # Open new bin
                bi = len(bins)
                bins.append({
                    'empty_boxes': [(0, 0, 0, W, H, D)],
                    'items': [(0, 0, 0, w, h, d)]
                })
                placement_map[idx] = (bi, 0, 0, 0)
        
        if not bins:
            return []
        
        # Re-index bins to maximize last-bin volume
        # Score: B + 0.1 * (1 - last_bin_vol / total_vol)
        # We want to minimize B. If B is fixed, we want to maximize last_bin_vol.
        # "Last bin" is defined by the largest label.
        # So we should assign the largest label to the bin with the largest volume.
        
        bin_volumes = []
        for bi in range(len(bins)):
            vol = sum(item[3]*item[4]*item[5] for item in bins[bi]['items'])
            bin_volumes.append(vol)
        
        # Sort bins by volume descending
        sorted_bin_indices = sorted(range(len(bins)), key=lambda i: -bin_volumes[i])
        
        # Map old index to new index
        # The bin with the largest volume gets the highest index (len(bins)-1)
        # The bin with the smallest volume gets index 0
        new_index_map = {}
        for new_idx, old_idx in enumerate(sorted_bin_indices):
            new_index_map[old_idx] = new_idx
            
        final_placements = []
        for i in range(n):
            if i in placement_map:
                old_bi, x, y, z = placement_map[i]
                new_bi = new_index_map[old_bi]
                w, h, d = boxes[i]
                final_placements.append((new_bi, x, y, z, w, h, d))
            else:
                # Fallback for unplaceable boxes (shouldn't happen)
                final_placements.append((0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]))
        
        return final_placements

    def evaluate(placements):
        if not placements:
            return float('inf')
        
        bins_used = set(p[0] for p in placements)
        B = len(bins_used)
        last_bin = max(bins_used)
        
        last_vol = 0
        for p in placements:
            if p[0] == last_bin:
                last_vol += p[4] * p[5] * p[6]
                
        frac = last_vol / container_vol
        if frac > 1.0: frac = 1.0
        if frac < 0.0: frac = 0.0
        
        score = B + 0.1 * (1 - frac)
        return score

    # Strategies
    def sort_vol_desc(box):
        return -(box[0] * box[1] * box[2])
    
    def sort_max_dim_desc(box):
        return -max(box)
    
    def sort_surface_desc(box):
        w, h, d = box
        return -(2*(w*h + h*d + w*d))
    
    def sort_min_dim_desc(box):
        return -min(box)
    
    def sort_aspect_desc(box):
        w, h, d = box
        if min(w,h,d) == 0: return 0
        return -max(w,h,d)/min(w,h,d)

    strategies = [
        sort_vol_desc,
        sort_max_dim_desc,
        sort_surface_desc,
        sort_min_dim_desc,
        sort_aspect_desc,
    ]
    
    best_placements = None
    best_score = float('inf')
    
    for sort_key in strategies:
        if time.time() > DEADLINE - 5:
            break
        try:
            placements = pack_instance(sort_key, None)
            score = evaluate(placements)
            if score < best_score:
                best_score = score
                best_placements = placements
        except Exception:
            pass

    # Randomized restarts with jittered sorting
    if time.time() < DEADLINE - 5:
        rng = random.Random(42)
        attempts = 0
        while time.time() < DEADLINE - 2:
            # Jitter the sorting by adding noise to the volume
            def jittered_sort(box):
                vol = box[0] * box[1] * box[2]
                # Add small noise relative to max volume? 
                # Just use uniform noise to shuffle close volumes
                noise = rng.uniform(-0.05, 0.05) * vol
                return -(vol + noise)
            
            try:
                placements = pack_instance(jittered_sort, None)
                score = evaluate(placements)
                if score < best_score:
                    best_score = score
                    best_placements = placements
            except Exception:
                pass
            attempts += 1

    if best_placements is None:
        # Fallback
        best_placements = []
        for i, (w, h, d) in enumerate(boxes):
            best_placements.append((0, 0, 0, w, h, d))
            
    return best_placements
