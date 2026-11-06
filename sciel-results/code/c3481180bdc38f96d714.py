"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import heapq

# EVOLVE-BLOCK-START

def load(boxes, container):
    W, H, D = container
    deadline = time.time() + 115.0  # 5s safety margin before 120s kill
    
    # Precompute all unique orientations for each box
    def get_orientations(w, h, d):
        opts = set()
        for perm in itertools.permutations((w, h, d)):
            opts.add(perm)
        return list(opts)
    
    # Floor-map based packing using a 3D grid of occupied cells
    # For efficiency with large containers, we use a coordinate-compressed approach
    # but for typical problem sizes, a direct grid may work. Let's use a list-based
    # approach with candidate position generation.
    
    def floor_map_pack(sorted_boxes, orientations_list, W, H, D):
        """
        Pack boxes using a floor-map (skyline) approach.
        We maintain a list of placed boxes and for each new box, generate candidate
        positions from the "skyline" of existing boxes.
        
        Candidate positions are:
        - (0, 0, 0)
        - For each placed box, positions at (px+pw, py, pz), (px, py+ph, pz), (px, py, pz+pd)
          and their combinations, filtered to valid positions.
        """
        placed = []  # list of (x, y, z, w, h, d)
        placements = []
        
        # To speed up overlap checking, we maintain a set of occupied "corners"
        # and use a spatial index (simple grid or list)
        
        for i, (bw, bh, bd) in enumerate(sorted_boxes):
            if time.time() > deadline:
                # Return what we have so far
                for j in range(i, len(sorted_boxes)):
                    w, h, d = sorted_boxes[j]
                    placements.append((-1, 0, 0, 0, w, h, d))
                break
            
            orients = orientations_list[i]
            best_pos = None
            best_score = None
            
            # Generate candidate positions
            candidates = set()
            candidates.add((0, 0, 0))
            
            for px, py, pz, pw, ph, pd in placed:
                # Positions adjacent to each face of each placed box
                candidates.add((px + pw, py, pz))
                candidates.add((px, py + ph, pz))
                candidates.add((px, py, pz + pd))
                # Also try combinations of two offsets
                candidates.add((px + pw, py + ph, pz))
                candidates.add((px + pw, py, pz + pd))
                candidates.add((px, py + ph, pz + pd))
                candidates.add((px + pw, py + ph, pz + pd))
            
            # Filter candidates to those within container bounds
            valid_candidates = set()
            for (cx, cy, cz) in candidates:
                if cx >= 0 and cy >= 0 and cz >= 0:
                    valid_candidates.add((cx, cy, cz))
            
            for ow, oh, od in orients:
                if ow > W or oh > H or od > D:
                    continue
                
                for (cx, cy, cz) in valid_candidates:
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue
                    
                    # Check for overlap with all placed boxes
                    valid = True
                    for px, py, pz, pw, ph, pd in placed:
                        if (cx < px + pw and cx + ow > px and
                            cy < py + ph and cy + oh > py and
                            cz < pz + pd and cz + od > pz):
                            valid = False
                            break
                    
                    if valid:
                        # Score: prefer lower z, then lower y, then lower x
                        score = (cz, cy, cx)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_pos = (cx, cy, cz, ow, oh, od)
            
            if best_pos is not None:
                x, y, z, ow, oh, od = best_pos
                placed.append((x, y, z, ow, oh, od))
                placements.append((0, x, y, z, ow, oh, od))
            else:
                placements.append((-1, 0, 0, 0, bw, bh, bd))
        
        return placements
    
    def compute_score(placements):
        total_vol = sum(w * h * d for b, x, y, z, w, h, d in placements if b == 0)
        return total_vol / (W * H * D)
    
    # Generate different sorting strategies
    def sort_by_volume_desc(boxes):
        return sorted(range(len(boxes)), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    def sort_by_max_dim_desc(boxes):
        return sorted(range(len(boxes)), key=lambda i: max(boxes[i]), reverse=True)
    
    def sort_by_min_dim_desc(boxes):
        return sorted(range(len(boxes)), key=lambda i: min(boxes[i]), reverse=True)
    
    def sort_by_surface_desc(boxes):
        return sorted(range(len(boxes)), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)
    
    def sort_by_aspect_vol(boxes):
        def key(i):
            w, h, d = boxes[i]
            aspect = max(w, h, d) / min(w, h, d)
            vol = w * h * d
            return (aspect, -vol)
        return sorted(range(len(boxes)), key=key)
    
    def sort_by_volume_then_max(boxes):
        return sorted(range(len(boxes)), key=lambda i: (boxes[i][0]*boxes[i][1]*boxes[i][2], max(boxes[i])), reverse=True)
    
    strategies = [
        ("vol_desc", sort_by_volume_desc),
        ("max_dim_desc", sort_by_max_dim_desc),
        ("min_dim_desc", sort_by_min_dim_desc),
        ("surface_desc", sort_by_surface_desc),
        ("aspect_vol", sort_by_aspect_vol),
        ("vol_max", sort_by_volume_then_max),
    ]
    
    # Precompute orientations for all boxes
    orientations_list = []
    for w, h, d in boxes:
        orientations_list.append(get_orientations(w, h, d))
    
    best_placements = None
    best_score = 0
    
    for strat_name, sort_func in strategies:
        if time.time() > deadline:
            break
        
        order = sort_func(boxes)
        sorted_boxes = [boxes[i] for i in order]
        sorted_orients = [orientations_list[i] for i in order]
        
        placements = floor_map_pack(sorted_boxes, sorted_orients, W, H, D)
        
        # Reorder placements back to original box order
        final_placements = [None] * len(boxes)
        for k, orig_idx in enumerate(order):
            if k < len(placements):
                final_placements[orig_idx] = placements[k]
        
        # Any boxes not placed get -1
        for i in range(len(boxes)):
            if final_placements[i] is None:
                w, h, d = boxes[i]
                final_placements[i] = (-1, 0, 0, 0, w, h, d)
        
        score = compute_score(final_placements)
        if score > best_score:
            best_score = score
            best_placements = final_placements
    
    # If no strategy completed, use a fallback
    if best_placements is None:
        placements = []
        for w, h, d in boxes:
            placements.append((-1, 0, 0, 0, w, h, d))
        best_placements = placements
    
    return best_placements

# EVOLVE-BLOCK-END
