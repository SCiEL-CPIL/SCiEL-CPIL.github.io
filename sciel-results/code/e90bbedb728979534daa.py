"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sorting strategies
    def sort_by_volume(b):
        return sorted(b, key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    def sort_by_max_dim(b):
        return sorted(b, key=lambda x: max(x[1], x[2], x[3]), reverse=True)
        
    def sort_by_surface_area(b):
        return sorted(b, key=lambda x: 2*(x[1]*x[2] + x[2]*x[3] + x[1]*x[3]), reverse=True)

    strategies = [sort_by_volume, sort_by_max_dim, sort_by_surface_area]
    
    def run_packing(sorted_boxes):
        """
        sorted_boxes: list of (orig_idx, w, h, d)
        Returns: (placements, num_bins, last_bin_fill_ratio)
        Uses Best Fit Decreasing (BFD) with expanded candidate generation.
        """
        bins = [] # List of lists of (w, h, d, x, y, z)
        placements = [None] * n
        
        def can_place(bin_boxes, w, h, d, x, y, z):
            if x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
            for (bw, bh, bd, bx, by, bz) in bin_boxes:
                if (x < bx + bw and x + w > bx and
                    y < by + bh and y + h > by and
                    z < bz + bd and z + d > bz):
                    return False
            return True
            
        def find_best_position(bin_boxes, w, h, d):
            # Generate candidates: Origin + Faces + Corners of existing boxes
            candidates = set()
            candidates.add((0, 0, 0))
            
            for (bw, bh, bd, bx, by, bz) in bin_boxes:
                # Faces
                candidates.add((bx + bw, by, bz))     # Right
                candidates.add((bx, by + bh, bz))     # Back
                candidates.add((bx, by, bz + bd))     # Top
                
                # Corners (combinations of face extensions)
                candidates.add((bx + bw, by + bh, bz))
                candidates.add((bx + bw, by, bz + bd))
                candidates.add((bx, by + bh, bz + bd))
                candidates.add((bx + bw, by + bh, bz + bd))
                
                # Also consider the "inner" corners? No, usually we place *on* top/side.
                # But sometimes sliding *along* a wall is better?
                # The face candidates cover the "next" position.
                
            valid_candidates = []
            for c in candidates:
                if 0 <= c[0] <= W-w and 0 <= c[1] <= H-h and 0 <= c[2] <= D-d:
                    valid_candidates.append(c)
                    
            best_pos = None
            best_key = None
            for (x, y, z) in valid_candidates:
                if can_place(bin_boxes, w, h, d, x, y, z):
                    # Heuristic: Minimize Z, then Y, then X
                    key = (z, y, x)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_pos = (x, y, z)
                        
            return best_pos

        for orig_idx, w, h, d in sorted_boxes:
            best_bin_idx = -1
            best_pos = None
            best_key = None
            
            # Try to place in existing bins (Best Fit)
            for b_idx, bin_boxes in enumerate(bins):
                pos = find_best_position(bin_boxes, w, h, d)
                if pos is not None:
                    x, y, z = pos
                    key = (z, y, x)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_pos = pos
                        best_bin_idx = b_idx
                        
            if best_bin_idx != -1:
                x, y, z = best_pos
                bins[best_bin_idx].append((w, h, d, x, y, z))
                placements[orig_idx] = (best_bin_idx, x, y, z, w, h, d)
            else:
                # Start new bin
                bins.append([(w, h, d, 0, 0, 0)])
                placements[orig_idx] = (len(bins)-1, 0, 0, 0, w, h, d)
                
        if not bins:
            return placements, 0, 0.0
            
        num_bins = len(bins)
        last_bin_idx = max(p[0] for p in placements if p is not None)
        
        last_bin_vol = 0
        for b_idx, x, y, z, w, h, d in placements:
            if b_idx == last_bin_idx:
                last_bin_vol += w * h * d
                
        container_vol = W * H * D
        fill_ratio = last_bin_vol / container_vol if container_vol > 0 else 0.0
        
        return placements, num_bins, fill_ratio

    best_result = None
    best_score = float('inf')
    
    for strategy in strategies:
        indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
        sorted_boxes = strategy(indexed_boxes)
        placements, num_bins, fill_ratio = run_packing(sorted_boxes)
        
        score = num_bins + 0.1 * (1 - fill_ratio)
        
        if score < best_score:
            best_score = score
            best_result = placements
            
    return best_result
# EVOLVE-BLOCK-END
